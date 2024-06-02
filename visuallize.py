import pandas as pd
import cv2
import os
from tqdm import tqdm
from configs import BOGGART_REPO_PATH
import json

# Constants
# video_name = "auburn_crf23_first_angle"
# video_dir = "/home/kth/rva/video"
# video_path = f"{video_dir}/crf23/auburn_first_angle.mp4"
# output_video_path = f"{video_dir}/crf23/auburn_first_angle_bbox.mp4"
# csv_path = f"{BOGGART_REPO_PATH}/inference_results/yolov5/{video_name}/{video_name}10.csv"





class ResultProcessor:
    def __init__(self, video_path, output_video_path, trajectory_dir, boggart_result_dir, query_result_dir, yolo_path, _type, end_frame):
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.trajectories = trajectory_dir
        self.type = _type
        self.end_frame = end_frame
        self.boggart_result = boggart_result_dir
        self.query_result = query_result_dir
        self.yolo_gt = yolo_path
        self.df = self.read_input_file()
        self.cap, self.fps, self.width, self.height, self.total_frames = self.initialize_video_capture()
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = self.create_video_writer()

    def get_smallest_mfs_files(self):
        files = os.listdir(self.boggart_result)
        result_df = self.get_smallest_mfs_info()
        filtered_files = []
        for file in files:
            if file.endswith('.json'):
                parts = file.split('_')
                chunk_start = int(parts[0])
                mfs_approach = float(parts[-2])
                # print(chunk_start, mfs_approach)
                
                # 데이터프레임에서 chunk_start와 mfs_approach가 일치하는지 확인
                if not result_df[(result_df['chunk_start'] == int(chunk_start)) & (result_df['mfs_approach'] == int(mfs_approach))].empty:
                    filtered_files.append(os.path.join(self.boggart_result, file))
        
        return filtered_files

    def json_to_dataframe(self, file_path):
        # Load bounding boxes from JSON file
        with open(file_path, 'r') as f:
            bboxes = json.load(f)
        
        # Create a list to hold data for the DataFrame
        data = []
        index_start = int(file_path.split('/')[-1].split('_')[0])
        # print(f'frame _Start {index_start}')
        # Iterate through the bounding boxes and prepare data for DataFrame
        for frame_index, boxes in enumerate(bboxes):
            for box in boxes:
                x1, y1, x2, y2 = box
                data.append([frame_index + index_start, x1, y1, x2, y2])

        # Create DataFrame
        df = pd.DataFrame(data, columns=['frame', 'x1', 'y1', 'x2', 'y2'])
        return df
    
    def load_and_concat_json(self):
        smallest_mfs_files = self.get_smallest_mfs_files()
        all_data = []

        for file_path in smallest_mfs_files:
            df = self.json_to_dataframe(file_path)
            all_data.append(df)

        # Concatenate all dataframes into a single dataframe
        final_df = pd.concat(all_data, ignore_index=True)
        return final_df

    def get_smallest_mfs_info(self):
        full_df = self.load_and_concat_csv()

        # score > 0.9 필터링
        filtered_df = full_df[full_df['score'] > 0.9]

        # 각 chunk_start 별로 mfs_approach가 가장 큰 값을 선택
        max_mfs_df = filtered_df.loc[filtered_df.groupby('chunk_start')['mfs_approach'].idxmax()]

        # score > 0.9 조건에 만족하지 않는 chunk_start 찾기
        all_chunk_starts = set(full_df['chunk_start'])
        filtered_chunk_starts = set(filtered_df['chunk_start'])
        remaining_chunk_starts = all_chunk_starts - filtered_chunk_starts

        # 남은 chunk_start에 대해 score가 가장 높은 row 선택
        if remaining_chunk_starts:
            remaining_df = full_df[full_df['chunk_start'].isin(remaining_chunk_starts)]
            highest_score_df = remaining_df.loc[remaining_df.groupby('chunk_start')['score'].idxmax()]
            result_df = pd.concat([max_mfs_df, highest_score_df], ignore_index=True)
        
        return result_df

    def load_and_concat_csv(self):
        if self.type == 1:
            file_dir = self.trajectories
        else:
            file_dir = self.query_result

        # 모든 CSV 파일을 찾아서 읽고 하나의 DataFrame으로 합치기
        all_files = [os.path.join(file_dir, f) for f in os.listdir(file_dir) if f.endswith('.csv')]
        df_list = []
        
        for file in all_files:
            df = pd.read_csv(file)
            
            if self.type == 1:
                # column: TS -> frame
                df['frame'] = df['TS']
                df_list.append(df)
            else:
                df = df[['chunk_start', 'class', 'mfs_approach', 'score', 'min_frames']]
                df['chunk_start'] = df['chunk_start'].astype(int)
                df['mfs_approach'] = df['mfs_approach'].astype(int)
                df['score'] = df['score'].astype(float)
                df['min_frames'] = df['min_frames'].astype(int)
                df_list.append(df)
        
        # concat all dataframes
        full_df = pd.concat(df_list, ignore_index=True)
    
        return full_df

    def read_input_file(self):
        if self.type == 0:
            return pd.read_csv(self.yolo_gt)
        elif self.type == 2:
            return self.load_and_concat_json()
        else:
            return self.load_and_concat_csv()


    def initialize_video_capture(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cap, fps, width, height, total_frames

    def create_video_writer(self):
        if self.type == 0:
            idx = 'gt'
        elif self.type == 1:
            idx = 'trajectory'
        else:
            idx = 'boggart_results'
 
        self.output_video_path = self.output_video_path.split('.')[0] + '_' + idx + '_.' + self.output_video_path.split('.')[1]
        return cv2.VideoWriter(self.output_video_path, self.fourcc, self.fps, (self.width, self.height))

    def draw_bounding_boxes(self, frame, frame_data):
        for _, row in frame_data.iterrows():
            # print(row)
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            # bstate = row['bstate']
            color = (0, 255, 0) 
            # if 'OBJECT' in bstate else (0, 0, 255)  # Green for OBJECT, Red for HYPOTHESIS
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return frame

    def process_video_frames(self):
        for frame_idx in tqdm(range(self.total_frames), desc="Processing frames"):
            if frame_idx == end_frame:
                break
            ret, frame = self.cap.read()
            if not ret:
                break

            h = self.height / 2
            w = self.width / 2
            frame = cv2.resize(frame, (int(w), int(h)))

            # Get bounding boxes for the current frame
            frame_data = self.df[self.df['frame'] == frame_idx]

            # Draw bounding boxes on the frame
            if not frame_data.empty:
                frame = self.draw_bounding_boxes(frame, frame_data)
            # else:
            #     break
            frame = cv2.resize(frame, (self.width, self.height))
            # Write the frame to the output video
            self.out.write(frame)

    def release_resources(self):
        self.cap.release()
        self.out.release()

    def run_draw_bbox(self):
        self.process_video_frames()
        self.release_resources()
        print(f"Processed video saved at: {self.output_video_path}")

if __name__ == "__main__":
#     video_name = "auburn_first_angle"
# video_dir = "/home/kth/rva"
# video_path = f"{video_dir}/auburn_first_angle.mp4"
# output_video_path = f"{video_dir}/auburn_first_angle_bbox.mp4"
# csv_path = f"{BOGGART_REPO_PATH}/inference_results/yolov5/{video_name}/{video_name}10.csv"
    # main_dir = "/home/kth/rva"
    # video_name = "auburn_first_angle"
    # video_path = f"{main_dir}/{video_name}.mp4"
    # video_chunk_path = f"{main_dir}/boggart/data/{video_name}10/video/{video_name}10_0.mp4"
    # output_video_path = f'{main_dir}/auburn_output.mp4'
    # yolo_csv_path = f"{main_dir}/boggart/inference_results/yolov5/{video_name}/{video_name}10.csv"
    # traj_csv_path = f"{main_dir}/boggart/data/{video_name}10/trajectories/deprecated/0_0.1_30_16_1800.csv"
    # boggart_result_dir = f"{main_dir}/boggart/data/{video_name}10/boggart_results/bbox"
    video_name = "lausanne_pont_bassieres"
    
    main_dir = "/home/kth/rva"
    video_path = f"{main_dir}/{video_name}.mp4"
    video_chunk_path = f"{main_dir}/boggart/data/{video_name}10/video/{video_name}10_0.mp4"
    output_video_path = f'{main_dir}/{video_name}_output.mp4'
    yolo_path = f"{main_dir}/boggart/inference_results/yolov5/{video_name}/{video_name}10.csv"
    trajectory_dir = f"{main_dir}/boggart/data/{video_name}10/trajectories"
    query_result_dir = f"{main_dir}/boggart/data/{video_name}10/query_results/bbox"
    boggart_result_dir = f"{main_dir}/boggart/data/{video_name}10/boggart_results/bbox"

    """
    0: yolo_detection
    1: trajectory
    2: boggart results
    3: query_results
    """
    _type = 1
    end_frame = 10000

    processor = ResultProcessor(video_path, output_video_path, trajectory_dir, boggart_result_dir, query_result_dir, yolo_path, _type, end_frame)
    processor.run_draw_bbox()







# import cv2
# import torch
# import numpy as np
# from tqdm import tqdm, trange
# from VideoData import VideoData, NoMoreVideo
# from configs import BOGGART_REPO_PATH
# from itertools import product

# # Initialize YOLO model
# if torch.cuda.is_available():
#     device = torch.device("cuda:2")
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
#     model.to(device)
# else:
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5l')

# # Constants
# video_name = "auburn_frist_angle"
# hour = 10
# video_path = f"/home/kth/rva/auburn_first_angle.mp4"
# output_video_path = f"/home/kth/rva/tracked_output.mp4"

# def run_yolo_and_draw_bboxes(ingest_combos, vd, chunk_size, output_video_path):
#     try:
#         # Initialize video capture
#         cap = cv2.VideoCapture(video_path)

#         # Get video properties
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         # Reduce resolution
#         scale_factor = 0.5
#         new_width = int(width * scale_factor)
#         new_height = int(height * scale_factor)

#         # Define codec and create VideoWriter object
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_width, new_height))

#         for i in range(len(ingest_combos)):
#             vals = ingest_combos[i]
#             chunk_start = vals[1][0]

#             frame_generator = vd.get_frames_by_bounds(chunk_start, chunk_start + chunk_size, int(1))
#             for frame_idx in trange(chunk_start, chunk_start + chunk_size, int(1), leave=False, desc=f"{chunk_start}_{chunk_size}"):
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 # Resize frame
#                 frame = cv2.resize(frame, (new_width, new_height))

#                 results = model(frame)

#                 # Convert results to DataFrame
#                 frame_results = results.pandas().xyxy[0]
#                 frame_results = frame_results.rename(columns={
#                     "xmin": "x1", "ymin": "y1", "xmax": "x2", "ymax": "y2",
#                     "confidence": "conf", "name": "label"})
#                 frame_results = frame_results[['x1', 'y1', 'x2', 'y2', 'label', 'conf']]

#                 # Draw bounding boxes on the frame
#                 for _, row in frame_results.iterrows():
#                     x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
#                     color = (0, 255, 0)  # Green color for bounding box
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

#                 # Write the frame to the output video
#                 out.write(frame)

#         # Release everything if job is finished
#         cap.release()
#         out.release()
#         print(f"Processed video saved at: {output_video_path}")

#     except Exception as e:
#         print("FAILED AT ", e)

# if __name__ == "__main__":
#     chunk_size = 1800
#     query_seg_size = 1800

#     video_data = VideoData(db_vid=video_name, hour=hour)

#     minutes = list(range(0, 60 * 1800, 1800))

#     param_sweeps = {
#         "diff_thresh": [16],
#         "peak_thresh": [0.1],
#         "fps": [30]
#     }

#     sweep_param_keys = list(param_sweeps.keys())[::-1]
#     _combos = list(product(*[param_sweeps[k] for k in sweep_param_keys]))
#     segment_combos = []
#     for minute in minutes:
#         chunk_starts = list(range(minute, minute + 1800, chunk_size))
#         segment_combos.append(chunk_starts)
#     ingest_combos = list(product(_combos, segment_combos))

#     # Run YOLO, draw bounding boxes, and save the video
#     run_yolo_and_draw_bboxes(ingest_combos, video_data, chunk_size, output_video_path)
