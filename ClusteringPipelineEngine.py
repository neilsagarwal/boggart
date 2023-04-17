from operator import itemgetter

import numpy as np
import pandas as pd

from configs import BackgroundConfig, TrajectoryConfig
from VideoData import VideoData
from ingest import IngestTimeProcessing
from utils import parallelize_update_dictionary
from QueryProcessor import QueryProcessor

class ClusteringPipelineEngine:
    def __init__(self, vid_label, bg_conf:BackgroundConfig=None, traj_conf:TrajectoryConfig=None, fps:int=30, query_conf=0.3, query_seg_size=1800, skip_no_traj=False):

        self.vid_label = vid_label

        self.bg_conf = BackgroundConfig(peak_thresh=0.1) if bg_conf is None else bg_conf
        self.traj_conf =  TrajectoryConfig(diff_thresh=16, chunk_size=1800, fps=fps) if traj_conf is None else traj_conf
        assert self.traj_conf.fps == fps

        self.fps = fps
        self.query_conf = query_conf
        self.query_seg_size = query_seg_size
        self.chunk_size = self.traj_conf.chunk_size

        self.skip_no_traj = skip_no_traj

        self.total_frames_per_hour = 60 * 60 * 30 # min/hour * sec/min * frames/sec

        self.mfs_sweep = [900, 450, 300, 200, 100, 50, 20, 10, 5, 2, 1, 0, -900, -300, -30, -10, -3, -2]

        self.all_vecs = None
        self._all_vecs = None
        self.hours = None # parameterize all vecs
        self._hours = None

        self.qp_sweep = None

    def _sweep_helper(self, sweep_idx):
        chunk_start, query_seg_start, qp = self.qp_sweep[sweep_idx]
        return qp.execute(chunk_start, query_seg_start, check_only=False, get_results_df=True)

    def _get_vec(self, query_seg_start):
        chunk_start = (query_seg_start//self.chunk_size) * self.chunk_size
        assert self.ip is not None
        return self.ip.get_feature_vector(chunk_start, query_seg_start, self.query_seg_size, self.skip_no_traj)

    def get_hour_vecs(self, hour, get_skips=False, get_df=False):
        vd = VideoData(self.vid_label, hour)
        self.ip = IngestTimeProcessing(vd, self.bg_conf, self.traj_conf)
        vecs = parallelize_update_dictionary(self._get_vec, range(0, self.total_frames_per_hour, self.query_seg_size), total_cpus=60, max_workers=30)
        self.ip = None

        skips = [k for k,v in vecs.items() if v is None]
        query_starts, vecs = zip(*[(k, v) for k, v in sorted(vecs.items()) if v is not None])

        vecs = np.array(vecs)

        assert not (get_skips and get_df)
        if get_skips:
            return vecs, skips

        if get_df:
            df = pd.DataFrame(query_starts, columns=["seg_start"])
            df["chunk_start"] = (df["seg_start"]//self.chunk_size) * self.chunk_size
            df["hour"] = hour
            return vecs, df

        return vecs

    def execute(self, hours, query_type, model, query_class, acc_target, percent_clusters, ioda):
        if type(hours) == int:
            hours = list(hours)

        if self._hours is None or self._hours != hours:
            self._hours = hours
            self._all_vecs, self._all_dfs = zip(*[self.get_hour_vecs(h, get_df=True) for h in hours])
            self._all_vecs = np.vstack(self._all_vecs)
            self._all_dfs = pd.concat(self._all_dfs).reset_index().drop(columns=["index"])

        self.hours = self._hours.copy()
        self.all_vecs = self._all_vecs.copy()
        self.all_dfs = self._all_dfs.copy()

        n_clusters = max(2, int(percent_clusters * len(self.all_vecs)))

        _, clusters, centroids, _, _ = IngestTimeProcessing.cluster_profile(self.all_vecs.copy(), n_clusters=n_clusters)

        if not np.array_equal(centroids[clusters][centroids], centroids):
            clusters[centroids] = list(range(n_clusters))

        self.all_dfs["cluster"] = clusters

        centroids_df = self.all_dfs.loc[centroids].copy()

        mfs_sweep_index = 0
        sweep_chunk_length = 8

        entire_df = None

        full_df = None

        while len(centroids_df) > 0:

            centroid_qps = []
            for (_, (query_seg_start, chunk_start, hour, cluster)) in centroids_df.iterrows():

                vd = VideoData(self.vid_label, hour)
                for mfs in self.mfs_sweep[mfs_sweep_index * sweep_chunk_length : (mfs_sweep_index + 1) * sweep_chunk_length]:
                    qp = QueryProcessor(query_type, vd, model, query_class, self.query_conf, mfs, self.bg_conf, self.traj_conf, ioda, self.query_seg_size)
                    centroid_qps.append([chunk_start, query_seg_start, qp])

            if len(centroid_qps) > 0:

                self.qp_sweep = centroid_qps
                c_sweep_res = parallelize_update_dictionary(self._sweep_helper, range(len(centroid_qps)), total_cpus=60, max_workers=min(60, len(centroid_qps)))
                self.qp_sweep = None
                _tempDF = list(map(itemgetter(1), sorted(c_sweep_res.items(), key=itemgetter(0))))
                c_sweep_res = pd.concat(_tempDF).reset_index().drop(columns="index")
                entire_df = pd.concat([entire_df, c_sweep_res.copy()]) if entire_df is not None else c_sweep_res
                c_sweep_res = c_sweep_res[c_sweep_res.score >= acc_target].sort_values(["hour", "seg_start"])
                c_sweep_res = c_sweep_res.loc[c_sweep_res.groupby(["hour", "seg_start"]).mfs_approach.idxmax()]

                centroids_df = centroids_df.merge(c_sweep_res, how='outer', indicator=True).loc[lambda x : x['_merge']=='left_only'][["seg_start", "chunk_start", "hour", "cluster"]]
                mfs_sweep_index += 1

            else:
                temp_df = centroids_df.merge(entire_df, on=["seg_start", "chunk_start", "hour"])
                c_sweep_res = temp_df.loc[temp_df.groupby(["seg_start", "chunk_start", "hour"]).score.idxmax()].drop(columns=["cluster"])
                centroids_df = []
            full_df = pd.concat([c_sweep_res, full_df]) if full_df is not None else c_sweep_res

        full_df = full_df.merge(self.all_dfs.loc[centroids], on=["hour", "chunk_start", "seg_start"]).sort_values("cluster").reset_index().drop(columns=["index"])

        assert len(full_df) == len(centroids)

        self.all_dfs["mfs_approach"] = full_df.mfs_approach.values[self.all_dfs["cluster"]]

        assert len(full_df.merge(self.all_dfs, how='outer', on=["chunk_start", "seg_start", "hour", "cluster"], indicator=True).loc[lambda x : x['_merge']=='both'].loc[lambda x : x['mfs_approach_x'] != x['mfs_approach_y']]) == 0

        remaining_segments_df = full_df.merge(self.all_dfs, how='outer', on=["chunk_start", "seg_start", "hour", "cluster", "mfs_approach"], indicator=True).loc[lambda x : x['_merge']=='right_only'][["hour", "chunk_start", "seg_start", "mfs_approach", "cluster"]]

        remaining_qps = []

        for (_, (hour, chunk_start, seg_start, mfs_approach, cluster)) in remaining_segments_df.iterrows():
            vd = VideoData(self.vid_label, hour)
            qp = QueryProcessor(query_type, vd, model, query_class, self.query_conf, mfs_approach, self.bg_conf, self.traj_conf, ioda, self.query_seg_size)
            remaining_qps.append([chunk_start, seg_start, qp])

        self.qp_sweep = remaining_qps
        remaining_query_results = parallelize_update_dictionary(self._sweep_helper, range(len(remaining_qps)), total_cpus=72, max_workers=36)
        self.qp_sweep = None

        _tempDF = list(map(itemgetter(1), sorted(remaining_query_results.items())))
        remaining_query_results = pd.concat(_tempDF).reset_index().drop(columns="index")

        mfs = remaining_query_results.min_frames.sum() + len(centroids) * self.query_seg_size * self.fps / 30
        full_results = pd.concat([full_df, remaining_query_results]).sort_values(["hour", "seg_start"]).reset_index().drop(columns="index")
        score = np.round(full_results.score.mean(), 4)

        assert len(full_results) == len(self.all_dfs)

        total_frames = len(full_results) * self.query_seg_size * self.fps / 30

        print(score, mfs, total_frames, np.round(mfs/total_frames, 4), n_clusters)

        return score, mfs, total_frames, np.round(mfs/total_frames, 4), n_clusters