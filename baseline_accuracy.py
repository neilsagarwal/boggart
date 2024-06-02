import os
import pandas as pd
import json
from mongoengine import connect, disconnect
from ModelProcessor import ModelProcessor
from VideoData import VideoData
from utils import (calculate_bbox_accuracy, calculate_count_accuracy,
                   get_ioda_matrix, calculate_binary_accuracy, parallelize_update_dictionary)
import numpy as np
# from evaluator import Evaluator

query_class = 2
query_conf = 0.7
fps = 30

def get_gt(video_name, hour, model, query_segment_start, query_segment_size = 1800):
    video_data = VideoData(video_name, hour)
    modelProcessor = ModelProcessor(model, video_data, query_class, query_conf, fps)

    gt_bboxes, gt_counts = modelProcessor.get_ground_truth(query_segment_start, query_segment_start + query_segment_size)

    # disconnect(alias='my')
    # print(gt_bboxes)
    return gt_bboxes



def accuracy(chunk_start):

    scores = []

    gt_bboxes = get_gt("lausanne_pont_bassieres", 10, "yolov5", chunk_start)

    det_bboxes = get_gt("lausanne_crf23_pont_bassieres", 10, "yolov5", chunk_start)
    # print(det_bboxes)

    # bounding box accuracy
    for bbox_gt, sr in zip(gt_bboxes, det_bboxes):
        scores.append(calculate_bbox_accuracy(bbox_gt, sr))
        # print(scores)

    return {"scores": scores}




total_scores = []
scores_dict = parallelize_update_dictionary(accuracy, range(0, 108000, 1800), max_workers=40, total_cpus=40)
for ts, score in scores_dict.items():
    total_scores.extend(score['scores'])

print(round(np.mean(np.array(total_scores)),4))



# gt_bboxess = get_gt("auburn_first_angle", 10, "yolov5", 0)
# query_results = get_gt("auburn_first_angle", 10, "yolov5", 0)
# for model_a_dets, model_b_dets in zip(gt_bboxess, query_results):

#     # check both empty
#     if len(model_a_dets) == len(model_b_dets) == 0:
#         print(1) 
#     # check one is empty
#     if len(model_a_dets) == 0 or len(model_b_dets) == 0:
#         print(0) 

#     if len(model_a_dets) == 0:
#         model_a_dets = np.empty(shape=[0, 4], dtype=np.float32)
#     if len(model_b_dets) == 0:
#         model_b_dets = np.empty(shape=[0, 4], dtype=np.float32)
#     # print(model_b_dets)
#     # for det in model_b_dets:
#     #     print(len(det), det)
#     det_dict = {
#         'detection_boxes': np.array(model_b_dets, dtype=np.float32),
#         'detection_scores': np.array([1 for _ in range(len(model_b_dets))], dtype=np.float32),
#         'detection_classes': np.array([0 for _ in range(len(model_b_dets))], dtype=np.uint8)
#     }
#     # print(det_dict)
#     gt_dict = {
#         "groundtruth_boxes" : np.array(model_a_dets, dtype=np.float32),
#         "groundtruth_classes" : np.array([0 for _ in range(len(model_a_dets))], dtype=np.uint8)
#     }
#     # x1,y1,x2,y2 -> x1,y1,w,h
#     det_dict['detection_boxes'] = np.hstack((det_dict['detection_boxes'][:, 0:2],
#                                             (det_dict['detection_boxes'][:, 2] - det_dict['detection_boxes'][:, 0])[:,np.newaxis],
#                                             (det_dict['detection_boxes'][:, 3] - det_dict['detection_boxes'][:, 1])[:,np.newaxis]))

#     gt_dict['groundtruth_boxes'] = np.hstack((gt_dict['groundtruth_boxes'][:, 0:2],
#                                             (gt_dict['groundtruth_boxes'][:, 2] - gt_dict['groundtruth_boxes'][:, 0])[:,np.newaxis],
#                                             (gt_dict['groundtruth_boxes'][:, 3] - gt_dict['groundtruth_boxes'][:, 1])[:,np.newaxis]))

#     det_combined = np.hstack((det_dict['detection_boxes'], det_dict['detection_scores'][:, np.newaxis], det_dict['detection_classes'][:, np.newaxis]))
#     gt_combined = np.hstack((gt_dict['groundtruth_boxes'], gt_dict['groundtruth_classes'][:, np.newaxis]))

#     # print(det_combined)

#     coco_eval = Evaluator()
#     coco_eval.add(det_combined, gt_combined)
#     coco_eval.accumulate()
#     a = coco_eval.summarize()
#     if a == -1:
#         print(gt_combined)
#         print("sdf")
#         print(det_combined)
