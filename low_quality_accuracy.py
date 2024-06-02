import os
import pandas as pd
import json
from mongoengine import connect, disconnect
from ModelProcessor import ModelProcessor
from VideoData import VideoData
from utils import (calculate_bbox_accuracy, calculate_count_accuracy,
                   get_ioda_matrix, calculate_binary_accuracy, parallelize_update_dictionary)
import numpy as np

csv_directory = '/home/kth/rva/boggart/data/lausanne_pont_bassieres10/query_results/bbox'
json_directory = '/home/kth/rva/boggart/data/lausanne_pont_bassieres10/boggart_results/bbox'

# csv_directory = '/home/kth/rva/boggart/data/lausanne_pont_bassieres10/query_results/bbox'
# json_directory = '/home/kth/rva/boggart/data/lausanne_pont_bassieres10/boggart_results/bbox'

# csv_directory = '/home/kth/rva/boggart/data/experiment/crf42/0.9_0.5_car/query_results/bbox'
# json_directory = '/home/kth/rva/boggart/data/experiment/crf42/0.9_0.5_car/boggart_results/bbox'

ground_truth = "lausanne_pont_bassieres"
# vid_label = "auburn_crf32_first_angle"
hour = 10
model = "yolov5"
query_class = 2 # car
query_conf = 0.7
fps = 30

# 최고 score를 가진 파일의 min_frames 데이터를 저장할 딕셔너리
best_score_files = {} 

for csv_filename in os.listdir(csv_directory):
    if csv_filename.endswith('.csv'):
        # Construct full path to CSV file
        file_path = os.path.join(csv_directory, csv_filename)
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Extract the chunk start, score, and min_frames
        chunk_start = int(df.at[0, 'chunk_start'])
        score = df.at[0, 'score']
        min_frames = df.at[0, 'min_frames']
        
        # Store or update the dictionary with the highest score info
        if chunk_start not in best_score_files or best_score_files[chunk_start]['score'] < score:
            best_score_files[chunk_start] = {'score': score, 'min_frames': min_frames, 'filename': csv_filename}

def calculate_accuracy(_chunk_start):

    scores = []
    info = best_score_files[_chunk_start]

    # Find the JSON file corresponding to the chunk start and min_frames
    for json_filename in os.listdir(json_directory):
        parts = json_filename.split('_')
        if json_filename.split('_')[0] == str(_chunk_start) and str(info['filename'].split('_')[-2]) == parts[-2]:
            # print(json_filename)
            json_path = os.path.join(json_directory, json_filename)
            with open(json_path, 'r') as json_file:
                det_bboxes = json.load(json_file)
                
                gt_bboxes = get_gt(chunk_start)

                # bounding box accuracy
                for bbox_gt, sr in zip(gt_bboxes, det_bboxes):
                    scores.append(calculate_bbox_accuracy(bbox_gt, sr))


    return {"scores": scores}


def get_gt(query_segment_start, query_segment_size = 1800):

    # db = connect(
    #         db=ground_truth,
    #         username='root',
    #         password='root',
    #         host='mango4.kaist.ac.kr',
    #         authentication_source='admin',
    #         port=27017,
    #         maxPoolSize=10000,
    #         alias='my')

    video_data = VideoData(ground_truth, hour)
    modelProcessor = ModelProcessor(model, video_data, query_class, query_conf, fps)

    gt_bboxes, gt_counts = modelProcessor.get_ground_truth(query_segment_start, query_segment_start + query_segment_size)

    # disconnect(alias='my')
    # print(gt_bboxes)
    return gt_bboxes


    # binary_scores = []
    # count_scores = []

    # query_results_count = [int(elem > 0) for elem in query_results]
    # query_results_binary = query_results
    # for gt, sr_count, sr_binary in zip(gt_counts, query_results_count, query_results_binary):
    #     count_scores.append(calculate_count_accuracy(gt, sr_count))
    #     binary_scores.append(calculate_binary_accuracy(int(gt>0), sr_binary))

    # acc_count = round(np.mean(count_scores), 3)
    # acc_binary = round(np.mean(binary_scores), 3)


# disconnect(alias='default')
# get_gt(0)

total_scores = []
scores_dict = parallelize_update_dictionary(calculate_accuracy, range(0, 108000, 1800), max_workers=40, total_cpus=40)
for ts, score in scores_dict.items():
    total_scores.extend(score['scores'])

print(round(np.mean(np.array(total_scores)),4))

