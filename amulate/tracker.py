import torch
import numpy as np
import pandas as pd
import os
import cv2
from configs import BOGGART_REPO_PATH
from tqdm import tqdm, trange

class Tracker:

    def __init__(self, vid, hour, model, chunk_size=1800, query_seg_size=1800):
        self.video_data = VideoData(
            db_vid = vid,
            hour = hour
        )
        self.chunk_size = chunk_size
        self.query_seg_size = query_seg_size
        self.model = torch.hub.load(f'{BOGGART_REPO_PATH}/yolov5', 'custom', f'{model}.pt', source='local')

    def run_detect(ingest_combos):
        try:
            # Create an empty CSV file with headers only if the file does not already exist
            if not os.path.exists(csv_path):
                pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "label", "conf"]).to_csv(csv_path, index=False)

            for i in range(len(ingest_combos)):
                # Initialize DataFrame to store results for the current video
                results_df = pd.DataFrame(columns=["frame", "x1", "y1", "x2", "y2", "label", "conf"])

                vals = ingest_combos[i]
                chunk_start = vals[1][0]

                frame_generator = self.video_data.get_frames_by_bounds(chunk_start, chunk_start+chunk_size, int(1))
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