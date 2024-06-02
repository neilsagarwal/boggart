import os
import json
import pickle
from collections import defaultdict
from operator import itemgetter

import numpy as np
import pandas as pd
from scipy import optimize

from configs import BackgroundConfig, TrajectoryConfig, query_results_dir, boggart_results_dir
from ingest import IngestTimeProcessing
from ModelProcessor import ModelProcessor
from utils import (calculate_bbox_accuracy, calculate_count_accuracy,
                   get_ioda_matrix, calculate_binary_accuracy)
from VideoData import VideoData

class QueryProcessor:

    def __init__(self, query_type, video_data, model, query_class, query_conf, mfs_approach, bg_conf, traj_conf, ioda, query_segment_size):
        self.query_type = query_type
        self.video_data = video_data
        self.fps = traj_conf.fps
        self.modelProcessor: ModelProcessor = ModelProcessor(model, video_data, query_class, query_conf, self.fps)
        self.mfs_approach = mfs_approach
        self.ioda = ioda
        self.bg_conf: BackgroundConfig = bg_conf
        self.traj_conf : TrajectoryConfig = traj_conf
        self.query_segment_size : int = query_segment_size

        assert self.query_segment_size <= self.traj_conf.chunk_size

        qtypes = ["binary", "count", "bbox"]

        assert query_type in qtypes

        self.results_folder = {qtype : query_results_dir.format(video_dir=self.video_data.video_dir, query_type=qtype) for qtype in qtypes}
        self.boggart_results_folder = {qtype : boggart_results_dir.format(video_dir=self.video_data.video_dir, query_type=qtype) for qtype in qtypes}
        for q in qtypes:
            os.makedirs(self.results_folder[q], exist_ok=True)
            os.makedirs(self.boggart_results_folder[q], exist_ok=True)

    def get_results_fname(self, chunk_start, segment_start, get_save_info=False, query_type=None):

        query = self.query_type if query_type is None else query_type

        colnames, cols = self._get_label_cols(chunk_start, segment_start, query)
        results_fname = f"{self.results_folder[query]}{'_'.join(list(map(str, cols[3:])))}.csv" # vid/hour/query in results folder
        if get_save_info:
            return results_fname, colnames, cols
        return results_fname
    
    def get_boggart_results_fname(self, chunk_start, segment_start, query_type=None):
        query = self.query_type if query_type is None else query_type
        _, cols = self._get_label_cols(chunk_start, segment_start, query)
        results_fname = f"{self.boggart_results_folder[query]}{'_'.join(list(map(str, cols[3:])))}.json" # vid/hour/query in results folder
        return results_fname

    def _get_label_cols(self, chunk_start, segment_start, query):

        vid = self.video_data.vid_label
        hour = self.video_data.hour
        chunk_size = self.traj_conf.chunk_size
        segment_size = self.query_segment_size
        peak_thresh = self.bg_conf.peak_thresh
        diff_thresh = self.traj_conf.diff_thresh
        query_class = self.modelProcessor.query_class
        model = self.modelProcessor.model
        query_conf = self.modelProcessor.query_conf

        colnames = ['vid', 'hour', 'query', 'chunk_start', 'chunk_size', 'seg_start', 'seg_size', 'peak_thresh', 'diff_thresh', 'model', 'class', 'conf', 'fps', 'mfs_approach', 'ioda']
        cols = [vid, hour, query, chunk_start, chunk_size, segment_start, segment_size, peak_thresh, diff_thresh, model, query_class, query_conf, self.fps, self.mfs_approach, self.ioda]

        return colnames, cols

    def get_tracking_info(self, chunk_start, query_segment_start):
        ingestTimeProcessor = IngestTimeProcessing(self.video_data, self.bg_conf, self.traj_conf)
        results = ingestTimeProcessor.run_tracker(chunk_start, load=True)
        if type(results) is not pd.core.frame.DataFrame and results in [None, -1]:
            return results
        results['TS'] = chunk_start + (results['TS']-chunk_start) * 30/self.fps
        results = results[(results.TS >= query_segment_start) & (results.TS < query_segment_start + self.query_segment_size)]
        return results

    def prepare_tracking_results(self, df, query_segment_start):
        prepared_results = dict()
        if self.modelProcessor.crop_region is not None:
            x1, y1, x2, y2 = self.modelProcessor.crop_region
            # toss only if completely inside crop region
            # might be better way to implement...
            toss_idx = df.loc[(x1 <= df.x1) & (df.x1 <= x2) & (y1 <= df.y1) & (df.y1 <= y2) & (x1 <= df.x2) & (df.x2 <= x2) & (y1 <= df.y2) & (df.y2 <= y2)].index
            df = df.loc[~df.index.isin(toss_idx)]
        for i in range(query_segment_start, query_segment_start + self.query_segment_size, int(30/self.fps)):
            prepared_results[i] = df[df.TS == i][['x1', 'y1', 'x2', 'y2', 'ObjId', 'bstate']].values.tolist()
        return prepared_results


    def get_min_frame_set(self, data, query_segment_start):

        fps_factor = int(30/self.fps)
        edge_slack = 2 * fps_factor
        reward = 5
        skip_amt = fps_factor

        thresh = self.mfs_approach
        if thresh < 0:
            t = abs(thresh) * fps_factor
            return list(range(query_segment_start, query_segment_start + self.query_segment_size, t))

        thresh = thresh * fps_factor

         # id -> frame
        candidates = list()                 # obj id for each obj != ESTIMATED
        candidates_by_frames = list()       # frame no for each obj != ESTIMATED
        trajs_by_frames = dict()            # frame no -> [obj id]
        trajs = dict()                      # obj id -> [frame no]

        for k in data.keys():
            for elem in data[k]:
                if elem[5] != "ObjectState.ESTIMATED":
                    candidates.append(elem[4])
                    candidates_by_frames.append(k)
                if elem[4] not in trajs:
                    trajs[elem[4]] = []
                trajs[elem[4]].append(k)
                if k not in trajs_by_frames:
                    trajs_by_frames[k] = []
                trajs_by_frames[k].append(elem[4])

        traj_bounds = dict()            # obj id -> [min frame, max frame]
        for t, fras in trajs.items():
            traj_bounds[t] = [min(fras), max(fras)]

        delegate_frames = set()
        prev_traj_len = None
        same_traj_counter = 0
        prefer_candidates = True

        temp_dict = dict()              # frame no -> [traj ids that it covers + [-1] * reward if frame near thresh]

        for k in set(data.keys()).intersection(set(trajs_by_frames.keys())):
            for t in set(trajs_by_frames[k]):
                add_amt = list([t for i in trajs[t] if k-thresh <= i <= k + thresh])
                if len(add_amt) > 0:
                    if k not in temp_dict:
                        temp_dict[k] = []
                    temp_dict[k].extend(add_amt)
                    if k-traj_bounds[t][0] <= edge_slack or traj_bounds[t][1]-k <= edge_slack:
                        temp_dict[k].extend([-1] * reward)
        del trajs
        del traj_bounds

        while len(temp_dict) > 0:
            prefer_candidates = prefer_candidates and (prev_traj_len is None or same_traj_counter < 5)
            if len(temp_dict) == prev_traj_len:
                same_traj_counter += 1
            else:
                same_traj_counter = 0
            prev_traj_len = len(temp_dict)

            # pruned set of candidates
            rel_keys = set(temp_dict.keys()).intersection(candidates_by_frames)
            if not prefer_candidates or len(rel_keys) == 0:
                prefer_candidates = False
                rel_keys = temp_dict.keys()

            # if frame not in rel_keys, multiply by 0 so ignored by max
            max_frame = max(temp_dict.items(), key=lambda x: len(x[1]) * int(x[0] in rel_keys))

            max_frame = max_frame[0]
            delegate_frames.add(max_frame)

            # frames within +- threshold of max_frame
            affected_frames = set(range(max_frame-thresh, max_frame+thresh+skip_amt, skip_amt)).intersection(trajs_by_frames.keys())
            relevant_trajs = trajs_by_frames[max_frame]
            for af in affected_frames.intersection(temp_dict.keys()):
                temp_dict[af] = [i for i in temp_dict[af] if i not in relevant_trajs or i == -1]
                if len(temp_dict[af]) == 0 or (-1 * len(temp_dict[af]) == sum(temp_dict[af])):
                    del temp_dict[af]

        return delegate_frames

    # key frame에서의 traejctory와 gt_bbox 이용해서 객체 검출하기
    def _get_data_per_track(self, detections, tracking_results, ioda_threshold, query):
        assert len(detections) != 0
        key_frame_information = dict()
        for frame_no, dets in detections:
            key_frame_information[frame_no] = dict()
            key_frame_information[frame_no]["tracks"] = dict()
            # key_frame_information[frame_no]["traj_ids"] = []
            key_frame_information[frame_no]["transforms"] = dict()
            key_frame_information[frame_no]["missed_dets"] = 0
            key_frame_information[frame_no]["missed_dets_bboxes"] = []

            dets = np.array(dets)
            trackers_with_ids = np.array(tracking_results[frame_no])
            trackers = trackers_with_ids[..., :-2].astype(np.int32)
            key_frame_information[frame_no]["count"] = len(dets)
            key_frame_information[frame_no]["dets"] = dets
            for t in trackers_with_ids:
                key_frame_information[frame_no]["transforms"][int(t[-2])] = [] if query == "bbox" else 0
            if len(dets) == 0:
                continue
            if len(trackers) == 0:
                for i in range(len(dets)):
                    key_frame_information[frame_no]["missed_dets"] += 1
                    key_frame_information[frame_no]["missed_dets_bboxes"].append(dets[i])
            else:
                ioda_matrix = get_ioda_matrix(dets, trackers)
                for i in range(len(dets)):
                    idx_of_max = np.argmax(ioda_matrix[:, i])
                    if ioda_matrix[idx_of_max, i] >= ioda_threshold:
                        tracking_id, state = trackers_with_ids[idx_of_max][-2:]
                        traj_bbox = np.array(trackers_with_ids[idx_of_max][:-2])
                        if int(tracking_id) not in key_frame_information[frame_no]["tracks"]:
                            if query == "bbox":
                                key_frame_information[frame_no]["transforms"][int(tracking_id)] = []
                            else:
                                key_frame_information[frame_no]["tracks"][int(tracking_id)] = 0
                        if query == "bbox":
                            # using tracks to store corresp traj boxes
                            key_frame_information[frame_no]["tracks"][int(tracking_id)] = traj_bbox.astype(np.int16)
                            key_frame_information[frame_no]["transforms"][int(tracking_id)].append(dets[i].astype(np.int16))
                            # temp_idx = len(key_frame_information[frame_no]["transforms"][int(tracking_id)])-1
                            # key_frame_information[frame_no]["traj_ids"].append([temp_idx, int(tracking_id)]) # so [det_no, tracking_id]
                        else:
                            key_frame_information[frame_no]["tracks"][int(tracking_id)] += 1
                    else:
                        key_frame_information[frame_no]["missed_dets"] += 1
                        key_frame_information[frame_no]["missed_dets_bboxes"].append(dets[i])
        assert len(key_frame_information) != 0
        return key_frame_information

    def _prep_bounds(self, key_frame_info, start_frame, num_frames):
        key_frames = sorted(key_frame_info.items())
        markers = []
        for i in range(len(key_frames)-1):
            curr_frame, _ = key_frames[i]
            next_frame, _ = key_frames[i+1]
            marker = curr_frame + int((next_frame-curr_frame)/2)
            markers.append(marker)
        markers.append(start_frame + num_frames)
        return markers
    
    def load_boggart_results(self, chunk_start, query_segment_start):
        boggart_results_fname = self.get_boggart_results_fname(chunk_start, query_segment_start)
        with open(boggart_results_fname, "r") as f:
            return json.load(f)

    # return None if no video for this minute...
    def execute(self, chunk_start, query_segment_start, check_only=True, get_results_df=False, get_mfs=False):

        assert chunk_start <= query_segment_start <= query_segment_start + self.query_segment_size <= chunk_start + self.traj_conf.chunk_size

        results_fname, colnames, cols = self.get_results_fname(chunk_start, query_segment_start, get_save_info=True)

        if os.path.exists(results_fname):
            if check_only:
                return
            if get_results_df:
                return pd.read_csv(results_fname)

        return_dictionary = dict()

        trajectories_df = self.get_tracking_info(chunk_start, query_segment_start)
        if type(trajectories_df) is not pd.core.frame.DataFrame and trajectories_df == -1:
            return None
        gt_bboxes, gt_counts = self.modelProcessor.get_ground_truth(query_segment_start, query_segment_start + self.query_segment_size)
        return_dictionary["gt_bboxes"] = gt_bboxes
        # print(f"gt bboxes:{gt_bboxes}")

        return_dictionary["trajectory_df"] = trajectories_df.copy() if trajectories_df is not None else None

        no_tracking = trajectories_df is None
        if trajectories_df is not None:
            mot_results = self.prepare_tracking_results(trajectories_df.copy(), query_segment_start)
            no_tracking = np.sum([len(e) for e in list(mot_results.values())]) == 0

        if no_tracking:
            # get middle frame
            min_frames_set = [query_segment_start + int(self.query_segment_size/2/(30/self.fps))]
            mfs_dets = [(frame_no, gt_bboxes[int((frame_no-query_segment_start) * self.fps/30)]) for frame_no in min_frames_set]
            return_dictionary["mfs_size"] = len(mfs_dets)
            if self.query_type in ["count", "binary"]:
                curr_result = len(mfs_dets[0][1])
            else:
                curr_result = mfs_dets[0][1]

            query_results = [curr_result for _ in range(self.query_segment_size)]

        else:
            min_frames_set = self.get_min_frame_set(mot_results.copy(), query_segment_start)
            mfs_dets = [(frame_no, gt_bboxes[int((frame_no-query_segment_start) * self.fps/30)]) for frame_no in min_frames_set]
            return_dictionary["mfs_size"] = len(mfs_dets)

            assert len(mfs_dets) > 0

            key_frame_info = self._get_data_per_track(mfs_dets, mot_results.copy(), self.ioda, self.query_type)

            markers = self._prep_bounds(key_frame_info, query_segment_start, self.query_segment_size)

            if self.query_type in ["count", "binary"]:
                results_data = self.execute_count(markers, key_frame_info, mot_results)
            else:
                # propagate detection results
                kps_loc_template = self.traj_conf.get_kps_loc_template(self.video_data.video_dir, self.bg_conf.peak_thresh)
                kps_matches_template = self.traj_conf.get_kps_matches_template(self.video_data.video_dir, self.bg_conf.peak_thresh)
                results_data = self.execute_bbox(markers, key_frame_info, mot_results, kps_loc_template, kps_matches_template, query_segment_start, query_segment_start + self.query_segment_size)
                return_dictionary["minimization_errors"] = results_data["minimization_errors"]
            query_results = results_data["query_results"]
            # return_dictionary["distances"] = results_data["distances"]

        # a bit messy implementation because want to avoid duplicate work for count/binary
        if self.query_type == "bbox":
            return_dictionary["query_results"] = query_results
            scores = []
            for bbox_gt, sr in zip(gt_bboxes, query_results):
                scores.append(calculate_bbox_accuracy(bbox_gt, sr))
            return_dictionary["scores"] = scores

            if get_mfs:
                return_dictionary["mfs"] = list(map(itemgetter(0), mfs_dets))

            colnames.extend(["score", "min_frames"])
            cols.extend([round(np.mean(scores), 3), len(mfs_dets)])

            df = pd.DataFrame([cols], columns=colnames)
            df.to_csv(results_fname, index=False)
            
            boggart_results_fname = self.get_boggart_results_fname(chunk_start, query_segment_start)
            with open(boggart_results_fname, "w") as f:
                json.dump([[r.tolist() for r in result] for result in return_dictionary["query_results"]], f)

            if get_results_df:
                return df #, return_dictionary
            return return_dictionary

        query_results_binary = [int(elem > 0) for elem in query_results]
        query_results_count = query_results

        return_dictionary["query_results"] = query_results_binary if self.query_type == "binary" else query_results_count

        binary_scores = []
        count_scores = []

        for gt, sr_count, sr_binary in zip(gt_counts, query_results_count, query_results_binary):
            count_scores.append(calculate_count_accuracy(gt, sr_count))
            binary_scores.append(calculate_binary_accuracy(int(gt>0), sr_binary))

        return_dictionary["scores"] = binary_scores if self.query_type == "binary" else count_scores

        for query_type, scores in zip(["binary", "count"], [binary_scores, count_scores]):
            results_fname, colnames, cols = self.get_results_fname(chunk_start, query_segment_start, get_save_info=True, query_type=query_type)
            colnames.extend(["score", "min_frames"])
            cols.extend([round(np.mean(scores), 3), len(mfs_dets)])
            df = pd.DataFrame([cols], columns=colnames)
            df.to_csv(results_fname, index=False)

            if query_type == "binary":
                binary_df = df
            if query_type == "count":
                count_df = df
                
            boggart_results_fname = self.get_boggart_results_fname(chunk_start, query_segment_start, query_type=query_type)
            with open(boggart_results_fname, "w") as f:
                json.dump(return_dictionary["query_results"], f)

        if get_results_df:
            df = count_df if self.query_type == "count" else binary_df
            return df #, return_dictionary["distances"]

        return return_dictionary

    def get_corresp_key_frames(self, markers, frame_no, key_frames):
        x = np.where(np.array(markers) >= frame_no)[0]
        x = len(markers) if len(x) == 0 else x[0]
        assert frame_no <= markers[x]
        relevant_key = key_frames[x]
        if relevant_key > frame_no:
            other_key = key_frames[x-1] if x != 0 else None
        else:
            other_key = key_frames[x+1] if x != len(key_frames)-1 else None
        return relevant_key, other_key

    def execute_count(self, markers, key_frame_information, tracking_results):
        query_results = []
        key_frames = sorted(key_frame_information.keys())
        for key, value in sorted(tracking_results.items(), key=lambda x: int(x[0])):
            if key in key_frames:
                query_results.append(key_frame_information[key]["count"])
            elif len(key_frames) == 0:
                assert False
                query_results.append(0)
            else:
                ids = list(map(itemgetter(4), value))
                traj_boxes = np.array(list(map(lambda x : x[:4], value))).astype(np.int16)
                x = 0
                while key > markers[x]:
                    x += 1
                assert key <= markers[x]
                relevant_key = key_frames[x]
                frame_info = key_frame_information[relevant_key]
                curr_results = 0
                for obj_id, traj_box in zip(ids, traj_boxes):
                    if obj_id in frame_info["tracks"]:
                        curr_results += frame_info["tracks"].get(obj_id)
                # only get missed dets from closest frame...
                curr_results += frame_info["missed_dets"]
                query_results.append(curr_results)


        return_dictionary = dict()
        # return_dictionary["distances"] = smart_distances
        return_dictionary["query_results"] = query_results

        return return_dictionary

    def execute_bbox(self, markers, key_frame_information, tracking_results, kps_loc_template, kps_matches_template, start, end):
        fps_factor = int(30/self.fps)

        def opt_fn(x, kps_and_anchors):
            return sum([((x[1]-kpx)/(x[1]-x[0]) - ax) ** 2 for kpx, ax in kps_and_anchors])
        def opt_fn_scalar(x, kps_and_anchors, length):
            return sum([((x+length-kpx)/length - ax) ** 2 for kpx, ax in kps_and_anchors])

        frame_height, frame_width = self.video_data.get_frame_bounds()

        key_frames = sorted(key_frame_information.keys())
        query_results = defaultdict(list)
        good_tracks = defaultdict(list)
        kp_counts = defaultdict(set)

        # kp_paths: key frame -> dist from key frame -> path idx -> curr frame kp idx
        # det_paths: key_frame -> track -> det no -> path idx
        kp_paths = dict()
        det_paths = dict()
        key_frame_anchors = dict()

        for key_frame_no in key_frames:
            kp_paths[key_frame_no] = {0 : dict()}
            key_frame_anchors[key_frame_no] = dict()
            kps = None

            if len(key_frame_information[key_frame_no]["dets"]) > 0:
                query_results[key_frame_no].extend(key_frame_information[key_frame_no]["dets"])
            good_tracks[key_frame_no].extend(list(key_frame_information[key_frame_no]["tracks"].keys()))


            # track t -> gt bounding boxes
            for t, dets in sorted(key_frame_information[key_frame_no]["transforms"].items()):
                if len(dets) > 0:
                    for det_no, b in enumerate(dets):
                        if kps is None:
                            kps = np.array(pickle.load(open(kps_loc_template.format(frame_no=key_frame_no), 'rb')))
                        if len(kps) > 0:
                            for p in np.where((b[0] <= kps[:,0]) &  (kps[:,0] <= b[2]) & (b[1] <= kps[:,1]) &  (kps[:,1] <= b[3]).astype(int))[0]:
                                kp_paths[key_frame_no][0][p] = [p, t]

                                tmp = det_paths.get(key_frame_no, None)
                                if tmp is None:
                                    det_paths[key_frame_no] = {t : {det_no : [p]}}
                                else:
                                    tmp = tmp.get(t, None)
                                    if tmp is None:
                                        det_paths[key_frame_no][t] = {det_no: [p]}
                                    else:
                                        tmp = tmp.get(det_no, None)
                                        if tmp is None:
                                            det_paths[key_frame_no][t][det_no] = [p]
                                        else:
                                            det_paths[key_frame_no][t][det_no].append(p)
                                kpx, kpy = kps[p]
                                kp_counts[key_frame_no].add(p)
                                if det_no not in key_frame_anchors[key_frame_no]:
                                    key_frame_anchors[key_frame_no][det_no] = {p: [(b[2]-kpx)/(b[2]-b[0]), (b[3]-kpy)/(b[3]-b[1]), b]}
                                else:
                                    key_frame_anchors[key_frame_no][det_no][p] = [(b[2]-kpx)/(b[2]-b[0]), (b[3]-kpy)/(b[3]-b[1]), b]
        for key_frame_no in key_frames:
            for dir in [1*fps_factor, -1*fps_factor]:
                i = dir
                keep_going = True
                while keep_going:
                    rel_key, _ = self.get_corresp_key_frames(markers, key_frame_no +i, key_frames)
                    if key_frame_no == rel_key and start <= key_frame_no + i < end:
                        if i < 0:
                            prev_matches = None
                        else:
                            curr_matches = None

                        if i-dir in kp_paths[key_frame_no]:
                            for path_idx, prev_kp_idx in list(kp_paths[key_frame_no][i-dir].items()):
                                if i-dir == 0:
                                    prev_kp_idx = prev_kp_idx[0]
                                if i < 0:
                                    if prev_matches is None:
                                        prev_matches = pickle.load(open(kps_matches_template.format(frame_no=key_frame_no+i+fps_factor), 'rb'))
                                        temp_idx = []
                                    if len(prev_matches) > 0:
                                        temp_idx = prev_matches[0][np.where(np.in1d(prev_matches[1], prev_kp_idx))[0]]
                                else:
                                    if curr_matches is None:
                                        if not os.path.exists(kps_matches_template.format(frame_no=key_frame_no+i)):
                                            curr_matches = []
                                            keep_going=False
                                        else:
                                            curr_matches = pickle.load(open(kps_matches_template.format(frame_no=key_frame_no+i), 'rb'))
                                    temp_idx = []
                                    if len(curr_matches) > 0:
                                        temp_idx = curr_matches[1][np.where(np.in1d(curr_matches[0], prev_kp_idx))[0]]
                                if len(temp_idx) > 0:
                                    assert len(temp_idx) == 1
                                    curr_kp_idx = temp_idx[0]
                                    if i not in kp_paths[key_frame_no]:
                                        kp_paths[key_frame_no][i] = {path_idx : curr_kp_idx}
                                    else:
                                        kp_paths[key_frame_no][i][path_idx] = curr_kp_idx
                    else:
                        keep_going = False
                    i += dir


        minimization_errors = dict()

        for key_frame_no in key_frames:
            for dir in [1*fps_factor, -1*fps_factor]:
                i = dir
                keep_going = True
                while keep_going:
                    rel_key, other_key = self.get_corresp_key_frames(markers, key_frame_no +i, key_frames)
                    other_frame_info = key_frame_information[other_key] if other_key is not None else None
                    frame_info = key_frame_information[rel_key]
                    if key_frame_no == rel_key and start <= key_frame_no + i < end:

                        trajs = {x[4] : x[:4] for x in tracking_results[key_frame_no+i]}

                        curr_query_results = []
                        not_enough_kps = []

                        if i in kp_paths[key_frame_no]:
                            curr_kps = pickle.load(open(kps_loc_template.format(frame_no=key_frame_no+i), 'rb'))
                            # find all kps and tracks relevant; then find the ones specific to that det
                            all_kps_in_play = set(list(kp_paths[key_frame_no][i].keys())) # kps
                            all_tracks_in_play = list([kp_paths[key_frame_no][0][kp_idx][1] for kp_idx in all_kps_in_play]) # tracks
                            for t in set(all_tracks_in_play):
                                if t in trajs:
                                    traj_bbox = trajs[t]
                                    for det_no, curr_det_paths in sorted(det_paths[key_frame_no][t].items()):

                                        # all kps present in this trajectory
                                        kps_in_play = np.array(list(set(curr_det_paths).intersection(all_kps_in_play))) # paths of kps in that det!
                                        tracks_in_play = list([kp_paths[key_frame_no][0][kp_idx][1] for kp_idx in kps_in_play])
                                        rel_kp_paths = kps_in_play[np.where(np.in1d(tracks_in_play, t))[0]]
                                        rel_kp_indices_in_curr_frame = [kp_paths[key_frame_no][i][p] for p in rel_kp_paths]
                                        anchors = [key_frame_anchors[key_frame_no][det_no][p] for p in rel_kp_paths]
                                        kps_and_anchors_x = []
                                        kps_and_anchors_y = []
                                        gt_bbox = []
                                        for kp_idx, anchor in zip(rel_kp_indices_in_curr_frame, anchors):
                                            kpx, kpy = curr_kps[kp_idx]
                                            if traj_bbox[0] - 100 <= kpx <= traj_bbox[2] + 100 and traj_bbox[1] - 100 <= kpy <= traj_bbox[3] + 100:
                                                ax, ay, gt_bbox = anchor
                                                kps_and_anchors_x.append([kpx, ax])
                                                kps_and_anchors_y.append([kpy, ay])
                                                kp_counts[key_frame_no+i].add(kp_idx)


                                        if len(gt_bbox) != 0:

                                            new_w = None
                                            new_h = None

                                            if other_frame_info is not None and t in other_frame_info['transforms']  and len(other_frame_info['transforms'].get(t)) > 0 and len(other_frame_info['transforms'].get(t)) == len(frame_info['transforms'].get(t)):
                                                ox1, oy1, ox2, oy2 = other_frame_info['transforms'].get(t)[det_no]
                                                num_frames_between_key_frames = float(abs(rel_key-other_key))
                                                frame_idx_between_key_frames = float(abs(other_key-key_frame_no))
                                                o_dims = np.array([ox2-ox1, oy2-oy1]).astype(np.float32)
                                                rel_dims = np.array([gt_bbox[2]-gt_bbox[0], gt_bbox[3]-gt_bbox[1]]).astype(np.float32)
                                                new_w, new_h = np.array(o_dims + (rel_dims-o_dims) * (frame_idx_between_key_frames/num_frames_between_key_frames)).astype(np.int16)

                                            scale_factor_max = min(1 + abs(i) * .25, 2)
                                            scale_factor_min = max(1 - abs(i) * .25, 0)
                                            gt_width = gt_bbox[2]-gt_bbox[0]
                                            gt_height = gt_bbox[3]-gt_bbox[1]
                                            conx1 = optimize.LinearConstraint(np.array([-1, 1]), max(scale_factor_min * gt_width, 1),  scale_factor_max * gt_width)
                                            cony1 = optimize.LinearConstraint(np.array([-1, 1]), max(scale_factor_min * gt_height, 1),  scale_factor_max * gt_height)
                                            conx2 = optimize.LinearConstraint(np.identity(2), 0, frame_width)
                                            cony2 = optimize.LinearConstraint(np.identity(2), 0, frame_height)

                                            consx = [conx1, conx2]
                                            consy = [cony1, cony2]

                                            if new_w is not None and new_h is not None:
                                                xr = optimize.minimize_scalar(opt_fn_scalar, bounds=(-1, frame_width+1), args=(kps_and_anchors_x, new_w), method='bounded', options={'xatol':0})
                                                yr = optimize.minimize_scalar(opt_fn_scalar, bounds=(-1, frame_height+1), args=(kps_and_anchors_y, new_h), method='bounded', options={'xatol':0})
                                                err = (xr.fun + yr.fun)/len(kps_and_anchors_x)
                                                xr, yr = xr.x, yr.x
                                                curr_query_results.append([err, [xr, yr, xr + new_w, yr+new_h]])
                                        
                                            else:
                                                xr = optimize.minimize(opt_fn, [gt_bbox[0], gt_bbox[2]], args=kps_and_anchors_x, method = "SLSQP", constraints=consx) #, options={'catol':0})
                                                yr = optimize.minimize(opt_fn, [gt_bbox[1], gt_bbox[3]], args=kps_and_anchors_y, method = "SLSQP", constraints=consy) #, options={'catol':0})

                                                err = (xr.fun + yr.fun)/len(kps_and_anchors_x)
                                                xr, yr = xr.x, yr.x
                                                curr_query_results.append([err, [xr[0], yr[0], xr[1], yr[1]]])
                                            good_tracks[key_frame_no+i].append(t)
                                            if abs(i) not in minimization_errors:
                                                minimization_errors[abs(i)] = [err]
                                            else:
                                                minimization_errors[abs(i)].append(err)
                                            # print(err, i)

                                        else:
                                            not_enough_kps.append(frame_info['transforms'].get(t)[det_no]) # if not enough kps, using key frame one

                        if len(curr_query_results) > 0:
                            curr_query_results = list(map(itemgetter(1), sorted(curr_query_results, key=itemgetter(0))))
                            query_results[key_frame_no+i].extend(curr_query_results)

                        if len(key_frame_information[key_frame_no]["missed_dets_bboxes"]) > 0:
                            query_results[key_frame_no+i].extend(key_frame_information[key_frame_no]["missed_dets_bboxes"])

                        if len(not_enough_kps) > 0:
                            query_results[key_frame_no+i].extend(not_enough_kps)

                    else:
                        keep_going = False
                    i += dir

        for j in tracking_results:
            if j not in query_results:
                query_results[j] = []
            else:
                query_results[j] = np.clip(np.array(query_results[j]).astype(np.int16), [0, 0, 0, 0], [frame_width, frame_height, frame_width, frame_height])
            if j not in good_tracks:
                good_tracks[j] = []

        query_results = list(map(itemgetter(1), sorted(query_results.items())))
        good_tracks = list(map(itemgetter(1), sorted(good_tracks.items())))

        return_dictionary = dict()
        return_dictionary["query_results"] = query_results
        return_dictionary["good_tracks"] = good_tracks
        return_dictionary["minimization_errors"] = minimization_errors
        return return_dictionary