import concurrent.futures
import os
import sys
import traceback
from functools import lru_cache, partial
from multiprocessing import Manager

import cv2
import numpy as np
from tqdm import tqdm, trange

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from object_detection.metrics import coco_evaluation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to quiet tensorflow logging

def redirect(stdout=sys.stdout, stderr=sys.stderr):
    def wrap(f):
        def newf(*args, **kwargs):
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = stdout
            sys.stderr = stderr
            try:
                return f(*args, **kwargs)
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
        return newf
    return wrap

def _fn(queue_ptr, f, *args):
    cpu_min, cpu_max = queue_ptr.get()
    os.sched_setaffinity(os.getpid(), set(range(int(cpu_min), int(cpu_max)+1)))
    try:
        ret = f(*args)
    except:
        ex_type, ex_value, ex_traceback = sys.exc_info()
        trace_back = traceback.extract_tb(ex_traceback)
        # # Format stacktrace
        stack_trace = list()
        for trace in trace_back:
            stack_trace.append(
                "File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
        print("Exception type : %s " % ex_type.__name__)
        print("Exception message : %s" % ex_value)
        print("Stack trace : %s" % stack_trace)
        ret = ex_value
    queue_ptr.put((cpu_min, cpu_max))
    return ret

def parallelize_update_dictionary(f, iterable, keys=None, max_workers=32, total_cpus=64, start_cpu=0):
    # to do... incorporate process affinity
    # to do incorporate shared dictionary
    cpus_per_worker = (total_cpus-start_cpu) / max_workers
    m = Manager()
    q = m.Queue()
    for i in range(max_workers):
        q.put((start_cpu + i*cpus_per_worker, start_cpu + (i+1)*cpus_per_worker-1))

    if keys is None:
        keys = iterable
    final_dictionary = dict()
    exceptions = dict()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # to do: differentiate between multiple args and single arg = list
        future_to_keys = {executor.submit(partial(_fn, q, f), x): k for x, k in zip(iterable, keys)}
        for future in tqdm(concurrent.futures.as_completed(future_to_keys), total=len(iterable), position=0):
            if isinstance(future.result(), Exception):
                exceptions[future_to_keys[future]] = future.result()
                print("EXCEPTION", future.result())
            else:
                final_dictionary[future_to_keys[future]] = future.result()
    if len(final_dictionary) != len(iterable):
        print("UHOH parallelization error: ", str(exceptions))
    # assert len(final_dictionary) == len(iterable), str(exceptions)
    return final_dictionary

def get_ioda_matrix(x, y):
    import numpy as np

    # get ioda matrix
    bb_gt = np.expand_dims(x, 0)
    bb_test = np.expand_dims(y, 1)
    # compute intersection area
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection_area = w * h
    # compute area of detection
    w = bb_gt[..., 2] - bb_gt[..., 0]
    h = bb_gt[..., 3] - bb_gt[..., 1]
    detection_area = w * h
    ioda_matrix = np.divide(intersection_area, detection_area)
    return ioda_matrix

def prepare_vid(vname, start):
    vid = cv2.VideoCapture(vname)
    for i in trange(start):
        vid.grab()
    return vid

@redirect(stdout=None, stderr=None)
def calculate_bbox_accuracy(model_a_dets, model_b_dets, prep_batch_only=False):
    # check both empty
    if len(model_a_dets) == len(model_b_dets) == 0:
        return 1
    # check one is empty
    if len(model_a_dets) == 0 or len(model_b_dets) == 0:
        return 0

    if len(model_a_dets) == 0:
        model_a_dets = np.empty(shape=[0, 4], dtype=np.float32)
    if len(model_b_dets) == 0:
        model_b_dets = np.empty(shape=[0, 4], dtype=np.float32)

    det_dict = {
        'detection_boxes': np.array(model_b_dets, dtype=np.float32),
        'detection_scores': np.array([1 for _ in range(len(model_b_dets))], dtype=np.float32),
        'detection_classes': np.array([0 for _ in range(len(model_b_dets))], dtype=np.uint8)
    }
    gt_dict = {
        "groundtruth_boxes" : np.array(model_a_dets, dtype=np.float32),
        "groundtruth_classes" : np.array([0 for _ in range(len(model_a_dets))], dtype=np.uint8)
    }
    
    evaluator = coco_evaluation.CocoDetectionEvaluator([{"id" : 0, "name" : ""}])
    evaluator.add_single_ground_truth_image_info(image_id="", groundtruth_dict=gt_dict)
    evaluator.add_single_detected_image_info(image_id="", detections_dict=det_dict)
    x = evaluator.evaluate()
    y = round(x['DetectionBoxes_Precision/mAP'], 3)

    return y 

@lru_cache(512)
def calculate_count_accuracy(model_a_dets, model_b_dets):
    model_a_dets = 0 if model_a_dets is None else model_a_dets
    model_b_dets = 0 if model_b_dets is None else model_b_dets
    if model_a_dets == 0 and model_b_dets == 0:
        curr_score = 1
    else:
        curr_score = 1 - float(abs(model_a_dets-model_b_dets)) / \
            float(max(model_a_dets, model_b_dets))
    return curr_score

def calculate_binary_accuracy(model_a_dets, model_b_dets):
    model_a_dets = 0 if model_a_dets is None else model_a_dets
    model_b_dets = 0 if model_b_dets is None else model_b_dets
    return int(model_a_dets == model_b_dets)