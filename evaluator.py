from collections import defaultdict
import numpy as np


class Evaluator:
    """
    COCO evaluation 기준으로 detection 성능을 평가하는 객체.
    이미지단위로 ground-truth(gt)와 detection결과(dt)를 누적하여 COCO evaluation을 수행한다.

    Attribute:
        iou_thrs (1-D numpy array):
            gt, dt 매칭시 iou threshold (default: 0.5 ~ 0.95 / 0.05 단위)
        recall_thrs (1-D numpy array):
            mAP 계산시에 precision 값을 샘플링할 recall 값 (default: 0. ~ 1. / 0.01 단위)
        max_det (list(int)):
            이미지별 detection결과에서 평가에 이용할 detection의 수 (default: 1, 10, 100)
        area_range (2-D numpy array):
            gt, dt의 bbox 넓이를 분류하는 기준 (default: [[0, 10**10], [0, 32**2], [32**2, 96**2], [96**2, 10**10]]
        area_range_name (list(str)):
            area_range의 원소들을 대표할 이름. 결과 출력시 area range값 대신 출력 (default: ['all', 'small', 'medium', 'large'])
        category (list(int) or 1-D numpy array):
            gt, dt의 카테고리 id (class id), (default: [1, 2, 3, 4, 5, 6] -> AITCS 프로젝트 기준)
        precision (5-D numpy array):
            입력된 모든 gt, dt를 종합하여 계산한 precision 배열.
            shape: (len(iou_thrs), len(recall_thrs), len(category), len(area_range), len(max_det))
        recall (4-D numpy array):
            입력된 모든 gt, dt를 종합하여 계산한 recall 배열.
            shape: (len(iou_thrs), len(category), len(area_range), len(max_det))
        eval_category (dict(key: int / val: dict(key: str, val: list(Any)))):
            category별로 dt-gt matching 결과와 bbox의 area range를 표시하는 indicator, detection의 score를 담음
    """

    def __init__(self):
        self.iou_thrs = np.linspace(0.5, 0.95, int(round((0.95 - 0.5) / 0.05) + 1))
        self.recall_thrs = np.linspace(0., 1., int(round((1. - 0.) / 0.01) + 1))
        self.max_det = np.array([1, 10, 100])
        self.area_range = np.array([[0, 10 ** 10], [0, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 10 ** 10]])
        self.area_range_name = ['all', 'small', 'medium', 'large']
        self.category = np.arange(1, 7)
        self.eval_category = defaultdict(lambda: {'match': [], 'match_valid': [], 'gt_valid': [], 'scores': []})

        self.precision = None
        self.recall = None

    def add(self, dt: np.ndarray, gt: np.ndarray):
        """
        카테고리별로 입력된 detection과 ground-truth를 매칭하여 self.eval_category에 추가

        Args:
            dt (2-D numpy array):
                이미지별 detection 결과 - x(left), y(top), w, h, score, category / shape: (m, 6)
            gt (2-D numpy array):
                이미지별 ground-truth - x(left), y(top), w, h, category / shape: (n, 5)
        """
        # category별로 수행
        for category in self.category:
            dt_cat_idx = np.where(dt[:, -1] == category)[0]
            gt_cat_idx = np.where(gt[:, -1] == category)[0]

            dt_ = dt[dt_cat_idx]
            gt_ = gt[gt_cat_idx]

            if dt_.shape[0] or gt_.shape[0]:
                dt_idx = np.argsort(-dt_[:, 4], kind='mergesort')  # score가 높은 dt부터 gt와 매칭시킴.
                dt_ = dt_[dt_idx]
                ious = self.iou_matrix(dt_[:, :4], gt_[:, :4])  # shape: (m, n)

                # area: bbox의 넓이
                # dt_valid: (m, len(self.area_range)) / bbox가 속하는 area range indicator
                dt_area = dt_[:, 2] * dt_[:, 3]
                dt_valid = np.logical_and(dt_area[:, None] >= self.area_range[:, 0],
                                          dt_area[:, None] < self.area_range[:, 1])
                gt_area = gt_[:, 2] * gt_[:, 3]
                gt_valid = np.logical_and(gt_area[:, None] >= self.area_range[:, 0],
                                          gt_area[:, None] < self.area_range[:, 1])
                if dt_valid.shape[0] == 0:
                    dt_valid = np.zeros((0, 4))

                # iou threshold별로 dt, gt를 matching. score가 높은 dt부터 시작하여 iou가 가장 큰 gt와 매칭
                # 아!: iou threshold에 따라 매칭 결과가 다름.
                match_list = []
                valid_list = []
                for iou_thr in self.iou_thrs:
                    match_iou = np.zeros((len(dt_),))
                    match_iou_gt = np.zeros((len(dt_),)).astype(np.int32)
                    if np.prod(ious.shape):
                        ious_copy = ious.copy()
                        for i, iou_row in enumerate(ious_copy):
                            max_iou = np.max(iou_row)
                            if max_iou >= iou_thr:
                                max_idx = int(np.argmax(iou_row))
                                match_iou[i] = 1.
                                match_iou_gt[i] = max_idx
                                ious_copy[:, max_idx] = 0.  # 매칭된 gt는 더 이상 매칭되지 않도록

                    # gt와 매칭된 dt는 gt의 area range를, 매칭되지 않은 dt는 dt의 area range에 속함
                    valid_iou = np.array(
                        [gt_valid[match_iou_gt[i]] if m else dt_valid[i] for i, m in enumerate(match_iou)])
                    match_list.append(match_iou)
                    valid_list.append(valid_iou)

                match = np.stack(match_list, axis=1)
                valid = np.stack(valid_list, axis=1)

                if valid.shape[0] == 0:
                    valid = np.zeros((0, 10, 4))

                scores = np.array([d[-2] for d in dt_])
                self.eval_category[category]['match'].append(match)
                self.eval_category[category]['match_valid'].append(valid)
                self.eval_category[category]['gt_valid'].append(gt_valid)
                self.eval_category[category]['scores'].append(scores)

    def accumulate(self):
        """
        add를 통해 추가된 모든 dt, gt의 매칭을 종합하여 precision, recall 계산
        """
        axis_iou = len(self.iou_thrs)
        axis_recall = len(self.recall_thrs)
        axis_cat = len(self.category)
        axis_area = len(self.area_range)
        axis_m_det = len(self.max_det)
        precision = -np.ones((axis_iou, axis_recall, axis_cat, axis_area, axis_m_det))
        recall = -np.ones((axis_iou, axis_cat, axis_area, axis_m_det))

        for cat_idx, cat in enumerate(self.category):
            if self.eval_category[cat]['gt_valid']:
                for m_idx, m_d in enumerate(self.max_det):
                    # 전체 dt를 score순으로 정렬
                    eval_cat = self.eval_category[cat]
                    scores = np.concatenate([score[:m_d] for score in eval_cat['scores']])
                    idx = np.argsort(-scores)

                    match = np.concatenate([m[:m_d] for m in eval_cat['match']], axis=0)[idx]
                    match_valid = np.concatenate([mv[:m_d] for mv in eval_cat['match_valid']], axis=0)[idx]
                    gt_valid = np.concatenate([gtv for gtv in eval_cat['gt_valid']], axis=0)

                    num_gt = np.sum(gt_valid, axis=0)
                    tps = np.logical_and(match[..., None], match_valid)  # area range에 속하지 않는 dt, gt는 계산에서 제외
                    tps_cumsum = np.cumsum(tps, axis=0)
                    fps = np.logical_and(np.logical_not(match[..., None]), match_valid)
                    fps_cumsum = np.cumsum(fps, axis=0)
                    # tps, fps: (m, iou_thrs: 10, area_range: 4)

                    for a_idx in range(len(self.area_range)):
                        if num_gt[a_idx] == 0:  # gt가 없는 경우 recall, precision을 평가하지 않음
                            continue

                        # TODO: try-except 제거?
                        try:
                            tps_, fps_ = tps_cumsum[..., a_idx], fps_cumsum[..., a_idx]
                        except IndexError:

                            tps_, fps_ = np.zeros((0, 10)), np.zeros((0, 10))

                        nd = len(tps_)
                        rc = tps_ / num_gt[a_idx]  # category, max_det, area_range별 recall
                        pr = tps_ / (fps_ + tps_ + np.spacing(1))  # precision

                        q = np.zeros((axis_iou, axis_recall))  # recall threshold마다 대응되는 precision

                        if nd:
                            recall[:, cat_idx, a_idx, m_idx] = rc[-1]
                        else:
                            recall[:, cat_idx, a_idx, m_idx] = 0.

                        pr = pr.tolist()
                        q = q.tolist()

                        # precision을 단조감소 형태로 수정
                        for i in range(nd - 1, 0, -1):
                            for j in range(axis_iou):
                                if pr[i][j] > pr[i - 1][j]:
                                    pr[i - 1][j] = pr[i][j]

                        pr = np.array(pr)
                        q = np.array(q)

                        for t_idx in range(axis_iou):
                            inds = np.searchsorted(rc[:, t_idx], self.recall_thrs, side='left')
                            try:
                                for ri, pi in enumerate(inds):
                                    q[t_idx, ri] = pr[pi, t_idx]
                            except:
                                pass
                        precision[:, :, cat_idx, a_idx, m_idx] = q
        self.precision = precision
        self.recall = recall

    def summarize(self):
        """
        accumulate를 통해 계산한 precision, recall을 요약.
        gt가 없는 경우는 -1로 표기됨.
        """

        def _summarize(result='precision', iou_thr=None, area_range_idx=0, max_det_idx=2):

            if iou_thr is None:
                iou_str = f'{self.iou_thrs[0]:0.2f}:{self.iou_thrs[-1]:0.2f}'
            else:
                iou_str = f'{iou_thr:0.2f}'
                iou_idx = np.where(iou_thr == self.iou_thrs)[0]

            if result == 'precision':
                title = 'Average Precision'
                abbr = '(AP)'
                s = self.precision
                if iou_thr is not None:
                    s = s[iou_idx]
                s = s[:, :, :, area_range_idx, max_det_idx]
            elif result == 'recall':
                title = 'Average Recall'
                abbr = '(AR)'
                s = self.recall
                if iou_thr is not None:
                    s = s[iou_idx]
                s = s[:, :, area_range_idx, max_det_idx]

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])

            area_str = self.area_range_name[area_range_idx]
            max_det = self.max_det[max_det_idx]
            # report_format = f' {title:<18} {abbr} ' \
            #                 f'@[ IoU={iou_str:<9} | ' \
            #                 f'area={area_str:>6s} | ' \
            #                 f'maxDets={max_det:>3d} ] ' \
            #                 f'= {mean_s:0.3f}'
            return mean_s

        return _summarize('precision')
        # _summarize('precision', iou_thr=.5, max_det_idx=2)
        # _summarize('precision', iou_thr=.75, max_det_idx=2)
        # _summarize('precision', area_range_idx=1, max_det_idx=2)
        # _summarize('precision', area_range_idx=2, max_det_idx=2)
        # _summarize('precision', area_range_idx=3, max_det_idx=2)

        # _summarize('recall', max_det_idx=0)
        # _summarize('recall', max_det_idx=1)
        # _summarize('recall', max_det_idx=2)
        # _summarize('recall', area_range_idx=1, max_det_idx=2)
        # _summarize('recall', area_range_idx=2, max_det_idx=2)
        # _summarize('recall', area_range_idx=3, max_det_idx=2)

    def iou_matrix(self, box_0: np.ndarray, box_1: np.ndarray) -> np.ndarray:
        """
        (m, 4), (n, 4) shape의 xywh를 받아 (m, n) shape의 iou 계산
        """
        box_0 = box_0[:, None]
        box_1 = box_1[None]

        box_0_lt, box_0_wh = np.split(box_0, 2, axis=-1)
        box_1_lt, box_1_wh = np.split(box_1, 2, axis=-1)

        max_lt = np.maximum(box_0_lt, box_1_lt)
        max_rb = np.minimum(box_0_lt + box_0_wh, box_1_lt + box_1_wh)
        inter_wh = max_rb - max_lt
        inter_wh = np.where(inter_wh > 0., inter_wh, 0.)
        intersection = np.prod(inter_wh, axis=-1)

        box_0_area = np.prod(box_0_wh, axis=-1)
        box_1_area = np.prod(box_1_wh, axis=-1)
        union = box_0_area + box_1_area - intersection

        return intersection / (union + np.spacing(1.))


# detection_sample = np.array([[0, 418, 163, 540]])
# gt_sample = np.array([[922.084228515625, 435.2584533691406, 960.0 - 922.084228515625, 514.0689697265625 - 435.2584533691406]])
# dect = np.hstack((detection_sample, np.array([[0.9, 1]])))  # Append score and category

# # coco_eval = Evaluator()
# #     cco_eval.add()

# print(gt_sample.shape)

# import numpy as np

# # 예제 데이터
# model_b_dets = [[0, 418, 163, 540-418]]  # 여기서는 단일 예시를 사용

# # Detection 데이터 준비
# detection_boxes = np.array(model_b_dets, dtype=np.float32)  # 박스 정보
# detection_scores = np.array([0.9 for _ in range(len(model_b_dets))], dtype=np.float32)  # 점수 정보 (예시로 모두 0.9)
# detection_classes = np.array([1 for _ in range(len(model_b_dets))], dtype=np.uint8)  # 클래스 정보 (예시로 모두 1)

# # 모든 정보를 하나의 배열로 결합
# det_combined = np.hstack((detection_boxes, detection_scores[:, np.newaxis], detection_classes[:, np.newaxis]))
# print("Combined Detection Data:", det_combined)
# # 예제 데이터
# model_a_dets = [[922.084228515625, 435.2584533691406, 37.915771484375, 78.81051635742188]]  # 위에서 계산한 w, h 포함

# # Ground-truth 데이터 준비
# groundtruth_boxes = np.array(model_a_dets, dtype=np.float32)
# groundtruth_classes = np.array([1 for _ in range(len(model_a_dets))], dtype=np.uint8)  # 클래스 정보 (예시로 모두 1)

# # 모든 정보를 하나의 배열로 결합
# gt_combined = np.hstack((groundtruth_boxes, groundtruth_classes[:, np.newaxis]))
# print("Combined Ground-truth Data:", gt_combined)
# coco_eval = Evaluator()
# coco_eval.add(det_combined, gt_combined)
# coco_eval.accumulate()
# print(coco_eval.summarize())


