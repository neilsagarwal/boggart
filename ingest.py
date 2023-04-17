import os
import pickle
from operator import itemgetter

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from ut_tracker import Tracker
from VideoData import VideoData, NoMoreVideo
from configs import BackgroundConfig, TrajectoryConfig


class IngestTimeProcessing:

    def __init__(self, video_data_config:VideoData, bg_config:BackgroundConfig=None, traj_config:TrajectoryConfig=None, fps:int=30):

        self.vd: VideoData = video_data_config
        self.video_dir = self.vd.video_dir
        os.makedirs(self.video_dir, exist_ok=True)
        self.fps = fps

        if bg_config is not None:
            self.bg_config : BackgroundConfig = bg_config
            self.bg_dir = self.bg_config.get_bg_dir(self.video_dir)
            os.makedirs(self.bg_dir, exist_ok=True)

        if traj_config is not None:
            self.traj_config: TrajectoryConfig = traj_config
            self.traj_dir = self.traj_config.get_traj_dir(self.video_dir)
            os.makedirs(self.traj_dir, exist_ok=True)

    def __str__(self):
        return f"IngestProcessor@[{self.vd.db_vid}{self.vd.hour}]"

    def get_base_background(self, bg_start):

        frames = []

        frame_generator = self.vd.get_frames_by_bounds(bg_start, bg_start + self.bg_config.bg_dur, self.bg_config.sample_rate)

        for i in trange(bg_start, bg_start + self.bg_config.bg_dur):
            if i % self.bg_config.sample_rate != 0:
                continue
            img = next(frame_generator)
            if img is None:
                print(f"{self}: Last frame in background @ {i}")
                break

            h, w , _ = img.shape
            h /= 2 # scale down
            w /= 2 # scale down
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (int(w/self.bg_config.box_length), int(h/self.bg_config.box_length)))
            img = img.astype(np.uint8).flatten().reshape(-1,1)
            frames.append(img)

        vals = np.concatenate(frames, axis=1)

        return vals

    def get_frequencies(self, vals):
        max_count = []
        max_values = []
        second_max_values = []
        second_max_count = []

        for v in tqdm(vals):
            srtd = sorted(list(zip(*np.unique(v//self.bg_config.quant, return_counts=True))), key=itemgetter(1))[::-1]
            max_count.append(srtd[0][1])
            max_values.append(srtd[0][0])
            if len(srtd) > 1:
                second_max_values.append(srtd[1][0])
                second_max_count.append(srtd[1][1])
            else:
                second_max_count.append(0)
                second_max_values.append(0)

        max_values = np.array(max_values).reshape(-1, 1) * self.bg_config.quant + (self.bg_config.quant/2)
        second_max_values = np.array(second_max_values).reshape(-1, 1) * self.bg_config.quant + (self.bg_config.quant/2)
        max_count = np.array(max_count).reshape(-1, 1).astype(np.int16)
        second_max_count = np.array(second_max_count).reshape(-1, 1).astype(np.int16)

        return max_values.astype(np.int16), second_max_values.astype(np.int16), max_count, second_max_count

    def generate_background(self, bg_start, load=False):
        bg_proper_fname = self.bg_config.get_proper_bg_fname(self.video_dir, bg_start)
        try:
            if os.path.exists(bg_proper_fname):
                if load:
                    with open(bg_proper_fname, "rb") as f:
                        bg_data = pickle.load(f)
                        return [bg_data['bg_max'], bg_data['bg_max2']]
                return
        except pickle.UnpicklingError:
            pass

        bg_before_vals = self.get_base_background(bg_start-self.bg_config.bg_dur) if bg_start - self.bg_config.bg_dur >= 0 else None

        try:
            bg_after_vals = self.get_base_background(bg_start+self.bg_config.bg_dur)
        except NoMoreVideo:
            bg_after_vals = None

        bg_curr_vals = self.get_base_background(bg_start)

        thresh = self.bg_config.peak_thresh * self.bg_config.bg_dur/self.bg_config.sample_rate

        curr_max_val, curr_max2_val, curr_max_count, curr_max2_count = self.get_frequencies(bg_curr_vals)

        bg_max  = np.zeros(curr_max_count.shape).astype(np.int16)
        bg_max2 = np.zeros(curr_max_count.shape).astype(np.int16)

        prominent_peak_mask = (curr_max_count - curr_max2_count >= thresh)
        np.place(bg_max, prominent_peak_mask, curr_max_val[prominent_peak_mask])
            
        if bg_after_vals is not None and bg_before_vals is not None:

            curr_and_after_vals = np.concatenate([bg_curr_vals, bg_after_vals], axis=1)
            after_max_val, after_max2_val, after_max_count, after_max2_count = self.get_frequencies(curr_and_after_vals)
            bg_after_mask = ~prominent_peak_mask & (after_max_val == curr_max_val) & (after_max2_val == curr_max2_val)
            np.place(bg_max, bg_after_mask, curr_max_val[bg_after_mask])
            np.place(bg_max2, bg_after_mask, curr_max2_val[bg_after_mask])

            all_vals = np.concatenate([bg_before_vals, curr_and_after_vals], axis=1)
            before_max_val, before_max2_val, before_max_count, before_max2_count = self.get_frequencies(all_vals)
            bg_before_mask = ~bg_after_mask & (after_max_val==before_max_val)
            np.place(bg_max, bg_before_mask, before_max_val[bg_before_mask])

        else:
            other_vals = bg_after_vals if bg_after_vals is not None else bg_before_vals
            curr_and_other_vals = np.concatenate([bg_curr_vals, other_vals], axis=1)
            other_max_val, other_max2_val, other_max_count, other_max2_count = self.get_frequencies(curr_and_other_vals)
            bg_other_mask = ~prominent_peak_mask & (other_max_val == curr_max_val) & (other_max2_val == curr_max2_val)
            np.place(bg_max, bg_other_mask, curr_max_val[bg_other_mask])
            np.place(bg_max2, bg_other_mask, curr_max2_val[bg_other_mask])

        bg_data = dict()
        bg_data['bg_max'] = bg_max
        bg_data['bg_max2'] = bg_max2

        with open(bg_proper_fname, "wb") as f:
            pickle.dump(bg_data, f)

        if load:
            return [bg_max, bg_max2]

    def get_foreground(self, bg_max, bg_max2, f):
        h, w = f.shape
        f = cv2.resize(f, (int(w/self.bg_config.box_length), int(h/self.bg_config.box_length)))
        f = f.astype(np.uint8).flatten().reshape(-1,1)
        result = np.zeros(f.shape)
        result.fill(255)
        result[((bg_max - self.traj_config.diff_thresh) < f) & (f < (bg_max + self.traj_config.diff_thresh)) & (bg_max != 0)] = 0
        result[((bg_max2 -  self.traj_config.diff_thresh) < f) & (f < (bg_max2 + self.traj_config.diff_thresh)) & (bg_max2 != 0)] = 0
        result = result.reshape(int(h/self.bg_config.box_length), int(w/self.bg_config.box_length))
        result = cv2.resize(result, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        result = cv2.medianBlur(result, self.traj_config.blur_amt)
        return result


    # return None means no trajectory, return -1 means no video!
    def run_tracker(self, chunk_start, load=False):

        traj_fname = self.traj_config.get_traj_fname(self.video_dir, chunk_start, self.bg_config)

        if os.path.exists(traj_fname):
            if load:
                try:
                    df = pd.read_csv(traj_fname)
                    if df.empty: # tracking was never done
                        return -1
                    return df
                except pd.errors.EmptyDataError: # only file name was created
                    return None
            return

        t = Tracker(chunk_start)
        t.kps_matches_template = self.traj_config.get_kps_matches_template(self.video_dir, self.bg_config.peak_thresh)
        t.kps_loc_template = self.traj_config.get_kps_loc_template(self.video_dir, self.bg_config.peak_thresh)

        bg_start = self.bg_config.get_bg_start(chunk_start)
        try:
            _temp_bg = self.generate_background(bg_start, load=True)
        except NoMoreVideo:
            print(f"{self}: No video for tracking @ {bg_start}")
            pd.DataFrame().to_csv(traj_fname) # saving empty frame
            return -1
        
        bg_max, bg_max2 = _temp_bg

        frame_generator = self.vd.get_frames_by_bounds(chunk_start, chunk_start+self.traj_config.chunk_size, int(30/self.traj_config.fps))
        for i in trange(chunk_start, chunk_start+self.traj_config.chunk_size, int(30/self.traj_config.fps), leave=False, desc=f"{chunk_start}_{self.traj_config.chunk_size}"):
            f = next(frame_generator)
            if f is None:
                print(f"skipping frame {i}")
                # assert i > chunk_start+self.traj_config.chunk_size - 250, i
                continue

            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

            h, w, = f.shape
            h /= 2 # scale down
            w /= 2 # scale down

            f = cv2.resize(f, (int(w), int(h)))

            o = f.copy()

            result = self.get_foreground(bg_max, bg_max2, f)

            t.process_frame(result, o, save_ts=i)
            
        t.save_results(traj_fname)

        if load:
            try:
                return pd.read_csv(traj_fname)
            except pd.errors.EmptyDataError:
                return None

    def get_feature_vector(self, chunk_start, query_seg_start, query_seg_size, skip_no_traj=False):
        results_df = self.run_tracker(chunk_start, load=True)
        if type(results_df) == int and results_df == -1:
            return None

        if results_df is None:
            if skip_no_traj:
                return None
            return [0, 0, 0, 0, 0]

        results_df.loc[:, 'TS'] = chunk_start + (results_df['TS']-chunk_start) * 30/self.traj_config.fps
        a = results_df[(results_df.TS >= query_seg_start) & (results_df.TS < query_seg_start + query_seg_size)].copy()

        if len(a) == 0:
            if skip_no_traj:
                return None
            return [0, 0, 0, 0, 0]

        a.loc[:, "area"] = (a.y2-a.y1) * (a.x2-a.x1)

        dur = []
        dists = []
        for objId in a.ObjId.unique():
            rel_df = a[a.ObjId == objId]
            mn = rel_df[rel_df.TS == min(rel_df.TS)][['x1', 'y1']]
            mx = rel_df[rel_df.TS == max(rel_df.TS)][['x1', 'y1']]
            if len(rel_df) > 1:
                dists.append(np.sum(abs(mn.values-mx.values)))
                dur.append((max(rel_df.TS)-min(rel_df.TS)))
        new_speed = float(sum(dists))/sum(dur) if sum(dur) != 0 else 0
        temp = a.groupby("ObjId").count()["TS"]
        good_tracks = temp[temp > 1].index.values
        a_prime = a[a.ObjId.isin(good_tracks)]
        med_x12 = (a_prime.x1 + a_prime.x2).median()/2 if len(a_prime) != 0 else 0
        med_y12 = (a_prime.y1 + a_prime.y2).median()/2 if len(a_prime) != 0 else 0
        area_var = np.sum(a_prime.groupby("ObjId").agg({"area": np.var}))[0]

        return np.array([new_speed, float(sum(dists)), med_x12, med_y12, area_var])

    @staticmethod
    def cluster(vecs, n_clusters=3):
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        scaler = StandardScaler()
        vecs = scaler.fit_transform(vecs)
        kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(vecs)
        distances = kmeans.transform(vecs)
        clusters = kmeans.predict(vecs)
        centroids = np.array([np.argmin(distances[:, i]) for i in range(n_clusters)])
        return centroids, clusters

    @staticmethod
    def cluster_profile(vecs, n_clusters=3):

        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        scaler = StandardScaler()
        vecs = scaler.fit_transform(vecs)

        kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(vecs)

        distances = kmeans.transform(vecs)
        clusters = kmeans.predict(vecs)

        centroids = np.array([np.argmin(distances[:, i]) for i in range(n_clusters)]) 

        per_chunk_closest_centroid = centroids[clusters]
        assert np.array_equal(per_chunk_closest_centroid, centroids[[np.argsort(distances[i])[0] for i in range(len(clusters))]])
        per_chunk_2nd_closest_centroid = centroids[[np.argsort(distances[i])[1] for i in range(len(clusters))]]

        return distances, clusters, centroids, per_chunk_closest_centroid, per_chunk_2nd_closest_centroid