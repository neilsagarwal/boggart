from configs import crops, frame_bounds
from VideoData import VideoData
import numpy as np
import concurrent.futures
from mongoengine import connect, disconnect
from db_model import Frame

class ModelProcessor:

    def __init__(self, model, video_data, query_class=None, query_conf=None, fps=None, do_crop=True, do_bound=True):
        assert type(video_data) is VideoData
        self.model = model
        self.video_data = video_data

        # unnecessary for loading into db
        if query_class is not None:
            self.query_class = query_class
            self.query_conf = query_conf
            self.do_crop = do_crop
            self.fps = fps
            self.do_bound = do_bound

            if "voc" in model:
                self.class_label = {6: "car", 14: "person"}[query_class]
            else:
                self.class_label = {2: "car", 0: "person", 1: "bicycle", 7: "truck", 8: "boat", 56 : "chair", 60 : "dining table", 41 : "cup", 39 : "bottle", 14 : "bird"}[query_class]

            self.crop_region = None
            if self.do_crop:
                assert self.class_label in crops[self.video_data.db_vid]
                self.crop_region = crops[self.video_data.db_vid][self.class_label]

            if self.do_bound:
                self.bounds = frame_bounds[self.video_data.db_vid]


    def get_ground_truth(self, start_frame, end_frame, counts_only=False, get_conf=False):

        gt_boxes = []
        gt_counts = []
        connect(self.video_data.db_vid, host="mango4.kaist.ac.kr",
            username='root',password='root',authentication_source='admin',connectTimeoutMS=10000, maxPoolSize=10000)
        for elem in Frame.objects(hour=self.video_data.hour, frame_no__in=range(start_frame, end_frame, int(30/self.fps))).order_by("+frame_no"):
            inferenceResults = elem.inferenceResults[self.model]
            curr_counts = 0
            curr_boxes = []
            for score, pred_class, det in zip(inferenceResults.detection_scores, inferenceResults.detection_classes, inferenceResults.detection_boxes):
                # type exchange
                det = [float(x) for x in det]
                
                assert score <= 1
                assert type(pred_class) is not float

                if score >= self.query_conf and pred_class == self.class_label:
                    toss = False
                    if self.crop_region is not None:
                        # ensures no intersection
                        toss = self.crop_region[0] <= float(det[0]) <= self.crop_region[2] and self.crop_region[1] <= float(det[1]) <= self.crop_region[3]
                        # toss = toss or (self.crop_region[0] <= float(det[2]) <= self.crop_region[2] and self.crop_region[1] <= float(det[3]) <= self.crop_region[3])
                        toss = toss and (self.crop_region[0] <= float(det[2]) <= self.crop_region[2] and self.crop_region[1] <= float(det[3]) <= self.crop_region[3])
                    if not toss:
                        if get_conf:
                            curr_boxes.append(det + [score])
                        else:
                            curr_boxes.append(det)
                        curr_counts += 1

            gt_boxes.append(curr_boxes)
            gt_counts.append(curr_counts)

        disconnect()

        if counts_only:
            return gt_counts

        # TODO: just do this when loading dets in db first time
        exec = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        def clean(x):
            if len(x) > 0:
                x = np.array(x)
                x[:,0:2] = np.clip(x[:, 0:2], a_min=0, a_max=None)
                x[:,2] = np.clip(x[:, 2], a_min=0, a_max=self.bounds[1])
                x[:,3] = np.clip(x[:, 3], a_min=0, a_max=self.bounds[0])
                return x.tolist()
            else:
                return x
        def f(i):
            gt_boxes[i] = clean(gt_boxes[i])
        # list(exec.map(f, range(len(gt_boxes)))) 

        return gt_boxes, gt_counts



a = ModelProcessor("yolov3-coco", VideoData("auburn_first_angle", 10), 2, 0.7, 30)
# assert type("class") is float
print(True or False)
a.get_ground_truth(0,1800)