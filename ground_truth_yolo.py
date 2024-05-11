from itertools import product
from tqdm import tqdm, trange
from VideoData import VideoData, NoMoreVideo
from ut_tracker import Tracker
from tqdm import tqdm, trange
import torch
import numpy as np
import pandas as pd
import os
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
main_dir = '/home/kth/rva/boggart'
csv_path = f'{main_dir}/inference_results/yolov3-coco/auburn_first_angle/auburn_first_angle10.csv'


def run_yolo(ingest_combos, vd, chunk_size):
    try:
        # Create an empty CSV file with headers only if the file does not already exist
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "label", "conf"]).to_csv(csv_path, index=False)

        for i in range(len(ingest_combos)):
            # Initialize DataFrame to store results for the current video
            results_df = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "label", "conf"])

            vals = ingest_combos[i]
            chunk_start = vals[1][0]

            # t = Tracker(chunk_start)

            frame_generator = vd.get_frames_by_bounds(chunk_start, chunk_start+chunk_size, int(1))
            for i in trange(chunk_start, chunk_start+chunk_size, int(1), leave=False, desc=f"{chunk_start}_{chunk_size}"):
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

                results = model(f)

                # Convert results to DataFrame
                frame_results = results.pandas().xyxy[0]
                frame_results = frame_results.rename(columns={
                    "xmin": "x1", "ymin": "y1", "xmax": "x2", "ymax": "y2", 
                    "confidence": "conf", "name": "label"})
                frame_results['frame'] = i
                frame_results = frame_results[['frame', 'x1', 'y1', 'x2', 'y2', 'label', 'conf']]

                # Append current frame results to main DataFrame
                results_df = pd.concat([results_df, frame_results], ignore_index=True)

            # Append results of current video to the main CSV file
            results_df.to_csv(csv_path, mode='a', header=False, index=False)   
    except Exception as e:
        print("FAILED AT ", e)



if __name__ == "__main__":
    vid = "auburn_first_angle"
    hour = 10
    chunk_size = 1800
    query_seg_size = 1800

    video_data = VideoData(db_vid = vid, hour = hour)

    minutes = list(range(0, 60 * 1800, 1800))

    param_sweeps = {
        "diff_thresh" : [16],
        "peak_thresh": [0.1],
        "fps": [30]
    }
    # additional_sweeps = None
    # if additional_sweeps is not None:
    #     for k, v in additional_sweeps.items():
    #         param_sweeps[k] = v

    sweep_param_keys = list(param_sweeps.keys())[::-1]

    _combos = list(product(*[param_sweeps[k] for k in sweep_param_keys]))

    segment_combos = []
    for minute in minutes:
        chunk_starts = list(range(minute, minute+1800, chunk_size))
        segment_combos.append(chunk_starts)
    ingest_combos = list(product(_combos, segment_combos))

    
    run_yolo(ingest_combos, video_data, chunk_size)