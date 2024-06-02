from operator import itemgetter

import numpy as np
import pandas as pd
from tqdm import tqdm

from configs import BackgroundConfig, TrajectoryConfig
from VideoData import VideoData
from ingest import IngestTimeProcessing
from utils import parallelize_update_dictionary
from QueryProcessor import QueryProcessor
from visuallize import ResultProcessor
# from object_detection.utils import coco_evaluation
from utils import calculate_bbox_accuracy

vid_label = "lausanne_pont_bassieres"
hour =10
query_type = 'bbox'
model = 'yolov5'
query_class = 2
query_conf = 0.7
mfs = 900
bg_conf = BackgroundConfig(peak_thresh=0.1)
traj_conf =TrajectoryConfig(diff_thresh=16, chunk_size=1800, fps=30)
ioda = 0.1
query_seg_size = 1800
vd = VideoData(vid_label, hour)
qp = QueryProcessor(query_type, vd, model, query_class, query_conf, mfs, bg_conf, traj_conf, ioda, query_seg_size)
# trajectories_df = self.get_tracking_info(chunk_start, query_segment_start)

query_segment_start = 0

gt_bboxes, gt_counts = qp.modelProcessor.get_ground_truth(query_segment_start, query_segment_start + qp.query_segment_size)
# mot_results = self.prepare_tracking_results(trajectories_df.copy(), query_segment_start)

print(gt_bboxes)




main_dir = "/home/kth/rva"
video_name = "lausanne_pont_bassieres"
video_path = f"{main_dir}/{video_name}.mp4"
video_chunk_path = f"{main_dir}/boggart/data/{video_name}10/video/{video_name}10_0.mp4"
output_video_path = f'{main_dir}/lausaane_output.mp4'
yolo_csv_path = f"{main_dir}/boggart/inference_results/yolov5/{video_name}/{video_name}10.csv"
traj_csv_path = f"{main_dir}/boggart/data/{video_name}10/trajectories/0_0.1_30_16_1800.csv"
boggart_result_dir = f"{main_dir}/boggart/data/{video_name}10/boggart_results/bbox"

"""
0: yolo_detection
1: trajectory
2: boggart results
"""
_type = 2
processor = ResultProcessor(video_chunk_path, output_video_path, boggart_result_dir, _type)
result_df = processor.concat_all_df()
# print(result_df)

# Store query results
query_results = []

# Extract bounding boxes from result_df for each frame and store them in query_results
for frame_index in range(len(gt_bboxes)):
    result_frame_bboxes = result_df[result_df['frame'] == frame_index][['x1', 'y1', 'x2', 'y2']].values.tolist()
    query_results.append(result_frame_bboxes)

print(query_results)
# Calculate accuracy scores for each frame
scores = []
for bbox_gt, sr in zip(gt_bboxes, query_results[0:1800]):
    if len(bbox_gt) ==0:
        continue

    scores.append(calculate_bbox_accuracy(bbox_gt, sr))

print(scores)
# Calculate and print the mean accuracy score
mean_accuracy = round(np.mean(scores), 3)
print(f"Mean Accuracy: {mean_accuracy}")