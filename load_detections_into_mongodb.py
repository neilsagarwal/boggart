import os

from mongoengine import connect, DoesNotExist
import pandas as pd
from tqdm import trange

from configs import BOGGART_REPO_PATH
from db_model import DetectionResult, Frame
from utils import parallelize_update_dictionary
# from ground_truth_yolo import ml_model, video_name, hour, csv_path

video_name = "lausanne_crf37_pont_bassieres"
ml_model = "yolov5"
hour = 10
csv_path = f"{BOGGART_REPO_PATH}/inference_results/{ml_model}/{video_name}/{video_name}{hour}.csv"

def exec(frame_start, num_frames=900):
    db = connect(
            db=video_name,
            username='root',
            password='root',
            host='mango4.kaist.ac.kr',
            authentication_source='admin',
            port=27017,
            maxPoolSize=10000)

    df = pd.read_csv(csv_path, skiprows=1,names=["frame", "x1", "y1", "x2", "y2", "label", "conf"], dtype=str)
    df['frame'] = df['frame'].astype(float).astype(int)
    df['conf'] = df['conf'].astype(float)
    # print(df['frame'])
    for i in trange(frame_start, frame_start+num_frames):
        frame = None
        try:
            frame = Frame.objects.get(frame_no=i, hour=hour)
            if ml_model in frame.inferenceResults:
                continue
        except DoesNotExist:
            pass

        curr_data = df[df['frame'] == i+1]

        # save det
        det = DetectionResult()
        det.model = ml_model
        det.detection_boxes = curr_data[['x1', 'y1', 'x2', 'y2']].values.tolist()
        det.detection_classes = curr_data['label'].values.tolist()
        det.detection_scores = curr_data['conf'].round(3).values.tolist()
        det.num_detections = len(det.detection_boxes)
        det.save()

        # save frame
        if not frame:
            frame = Frame(frame_no=i, hour=hour)
        frame.inferenceResults[ml_model] = det
        frame.save()


assert os.path.exists(csv_path)
parallelize_update_dictionary(exec, range(0, 108000, 900), max_workers=40, total_cpus=40)
