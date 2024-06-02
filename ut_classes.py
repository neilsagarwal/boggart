import cv2
from copy import deepcopy
import numpy as np
import math
from collections import defaultdict
from typing import Set, Tuple, List, Dict, DefaultDict
from ut_util_classes import BlobRect
from ut_util_classes import log

class BlobDetector:

    def __init__(self, morph_open_kernel=(5, 5), morph_close_kernel=(10, 10)):
        # 각 blob의 Id, blob의 bbox(BlobRect)
        # Dict[int, BlobRect]
        self.mBlobIdRectMap = dict()
        # 각 pixel의 blob id 저장
        self.mLabelMask = None
        # blob의 영역
        self.mCurrentBGSMask = None
        self.morph_open_kernel = morph_open_kernel
        self.morph_close_kernel = morph_close_kernel

    def getBGSMask(self):
        return deepcopy(self.mCurrentBGSMask)

    def getLabelMask(self):
        return deepcopy(self.mLabelMask)

    def getBlobBoundingBoxMap(self):
        return deepcopy(self.mBlobIdRectMap)

    # perform connected component analysis
    def perform_cca(self, im, orig_im):
        kernel = np.ones(self.morph_open_kernel, np.uint8)
        temp = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
        kernel = np.ones(self.morph_close_kernel, np.uint8)
        temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroid = cv2.connectedComponentsWithStats(temp)

        blobs_to_remove = []
        all_blobs = []

        # get list of blob rects
        for i in range(1, num_labels):  # first cc is background
            x1, y1, w, h, area = stats[i]
            all_blobs.append([i, x1, y1, x1+w, y1+h])

        # mask with 0 or 255... 255 means pixel is part of blob
        self.mCurrentBGSMask = np.zeros(labels.shape)
        # each pixel has a blob label... no blob label = 0
        self.mLabelMask = np.zeros(labels.shape)

        blobs_to_remove.extend([-1, 0])

        desired_labels = set(np.unique(labels)) - set(blobs_to_remove)

        if len(all_blobs) != 0:
            for l in desired_labels:
                self.mLabelMask[labels == l] = l
                self.mCurrentBGSMask[labels == l] = 255

        self.mCurrentBGSMask = self.mCurrentBGSMask.astype(np.uint8)

        for b in all_blobs:
            if b[0] not in blobs_to_remove:
                subImage = labels[b[2]: b[4], b[1]: b[3]]
                binaryBlob = np.zeros(subImage.shape, dtype=np.uint8)
                binaryBlob[subImage == b[0]] = 1
                # stores blob rect, small rect of pixels that corresp to that blob
                self.mBlobIdRectMap[b[0]] = BlobRect(b[1:], binaryBlob)

    def update(self, im, orig_im):
        self.perform_cca(im, orig_im)

    def filterOutBlobsWithNoKPs(self, pointsBlob: "PointsBlob"):
        blobs_to_remove = []
        for blobId, blobRect in self.mBlobIdRectMap.items():
            if len(pointsBlob.getKpDesc(blobId)["kp"]) == 0:
                # update label mask and bgs mask
                self.mLabelMask[self.mLabelMask == blobId] = 0
                self.mCurrentBGSMask[self.mCurrentBGSMask == blobId] = 0
                blobs_to_remove.append(blobId)
                log.debug(f"Removing blob {blobId} because 0 kps detected.")
        self.mCurrentBGSMask = self.mCurrentBGSMask.astype(np.uint8)

        for blobId in blobs_to_remove:
            del self.mBlobIdRectMap[blobId]

class FeatureDetectorExtractorMatcher:

    ratio = 0.80

    @staticmethod
    def detect(im, mask=None):
        _im = im.copy()
        if len(_im.shape) == 3:
            _im = cv2.cvtColor(_im, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(_im, mask)
        return PointsBlob([i.pt for i in kp], des)

    @staticmethod
    def match(prevDes, currDes) -> List[cv2.DMatch]:
        if prevDes is None or currDes is None:
            return []
        bf = cv2.BFMatcher()
        if type(currDes) is not np.ndarray:
            currDes = np.array(currDes)
        if type(prevDes) is not np.ndarray:
            prevDes = np.array(prevDes)
        try:
            multMatches1: List[List[cv2.DMatch]] = bf.knnMatch(prevDes, currDes, k=2)
            multMatches1 = FeatureDetectorExtractorMatcher.ratio_test(multMatches1)
            multMatches2: List[List[cv2.DMatch]] = bf.knnMatch(currDes, prevDes, k=2)
            multMatches2 = FeatureDetectorExtractorMatcher.ratio_test(multMatches2)
            final_matches: List[cv2.DMatch] = FeatureDetectorExtractorMatcher.symmetry_test(multMatches1, multMatches2)
        except Exception as e:
            print("uh oh error with knn matching", e)
            return []
        return final_matches

    @staticmethod
    def ratio_test(matches: List[List[cv2.DMatch]]) -> List[List[cv2.DMatch]]:
        good_matches = []
        for m in matches:
            if len(m) <= 1:
                continue
            if m[1].distance != 0 and m[0].distance / m[1].distance > FeatureDetectorExtractorMatcher.ratio:
                continue
            good_matches.append(m)
        return good_matches

    @staticmethod
    def symmetry_test(matches1: List[List[cv2.DMatch]], matches2: List[List[cv2.DMatch]]) -> List[cv2.DMatch]:
        good_matches: List[cv2.DMatch] = []

        m1s = map(lambda x : (x[0].queryIdx, x[0].trainIdx), matches1)
        m2s = list(map(lambda x : (x[0].trainIdx, x[0].queryIdx), matches2))
        
        good_matches = [matches1[i][0] for i, m in enumerate(m1s) if m in m2s]
        return good_matches


class PointsBlob:

    def __init__(self, kp=None, des=None):
        self.kp = kp
        self.des = des
        self.mBlobId = []
        self.mBlobIdToKpDesc = dict()


    def maskOutKps(self, mask):
        mask_vals = []
        for i, kp in enumerate(self.kp):
            x, y = kp
            val = mask[int(y)][int(x)]
            if val == 0:
                val = mask[int(math.ceil(y))][int(math.ceil(x))]
            if val == 0:
                val = mask[int(math.floor(y))][int(math.floor(x))]
            if val == 0:
                val = mask[int(math.ceil(y))][int(math.floor(x))]
            if val == 0:
                val = mask[int(math.floor(y))][int(math.ceil(x))]
            mask_vals.append(int(val) != 0)

        return PointsBlob(np.array(self.kp)[mask_vals].tolist(), self.des[mask_vals])


    def calculatePointBlobAssociation(self, mask):

        for i, kp in enumerate(self.kp):
            x, y = kp
            val = mask[int(y)][int(x)]
            if val == 0:
                val = mask[int(math.ceil(y))][int(math.ceil(x))]
            if val == 0:
                val = mask[int(math.floor(y))][int(math.floor(x))]
            if val == 0:
                val = mask[int(math.ceil(y))][int(math.floor(x))]
            if val == 0:
                val = mask[int(math.floor(y))][int(math.ceil(x))]
            assert val != 0

            val = int(val)

            self.mBlobId.append(val)
            if val not in self.mBlobIdToKpDesc:
                self.mBlobIdToKpDesc[val] = {"kp": [], "des": []}
            self.mBlobIdToKpDesc[val]["kp"].append(kp)
            self.mBlobIdToKpDesc[val]["des"].append(self.des[i])

    def updateBlobId(self, oldId, newId):
        for kp in self.mBlobIdToKpDesc[oldId]["kp"]:
            self.mBlobIdToKpDesc[newId]["kp"].append(kp)
        for des in self.mBlobIdToKpDesc[oldId]["des"]:
            self.mBlobIdToKpDesc[newId]["des"].append(des)
        del self.mBlobIdToKpDesc[oldId]

        for i in range(len(self.mBlobId)):
            if self.mBlobId[i] == oldId:
                self.mBlobId[i] = newId

    def getKpDesc(self, blobId: int):
        if blobId not in self.mBlobIdToKpDesc:
            self.mBlobIdToKpDesc[blobId] = {"kp": [], "des": []}
        return self.mBlobIdToKpDesc[blobId]


class FrameAssociation:

    minimum_area_overlap = 100

    def __init__(self, oldBlobAssoc: Dict[int, BlobRect], newBlobAssoc: Dict[int, BlobRect], nbMatches: Dict[Tuple[int, int], int], min_num_kp_matches:int = 8):
        self.mNewBlobs: Set[int] = set()  # 0-1
        self.mNoMatchBlobs: Set[int] = set()  # 1-0
        self.mOneToOne: Set[Tuple[int, int]] = set()  # 1-1
        self.mOneToN: Set[Tuple[int, Tuple[int]]] = set()  # 1-N
        self.mNToOne: Set[Tuple[Tuple[int], int]] = set()  # N-1
        self.min_num_kp_matches = min_num_kp_matches

        self.updateAssociation(oldBlobAssoc, newBlobAssoc, nbMatches)

        log.debug(f"newblobs: {self.mNewBlobs}")
        log.debug(f"mNoMatchBlobs: {self.mNoMatchBlobs}")
        log.debug(f"mOneToOne: {self.mOneToOne}")
        log.debug(f"mOneToN:  {self.mOneToN}")
        log.debug(f"mNToOne: {self.mNToOne}")

    def getNewBlob(self) -> Set[int]:
        return self.mNewBlobs

    def getUnmatchedBlob(self) -> Set[int]:
        return self.mNoMatchBlobs

    def getDirectMatchBlob(self) -> Set[Tuple[int, int]]:
        return self.mOneToOne

    def getBlobSplit(self) -> Set[Tuple[int, Tuple[int]]]:
        return self.mOneToN

    def getBlobMerge(self) -> Set[Tuple[Tuple[int], int]]:
        return self.mNToOne

    def updateAssociation(self, oldBlobAssoc: Dict[int, BlobRect], newBlobAssoc: Dict[int, BlobRect], nbMatches: Dict[Tuple[int, int], int]):
        oldBlobToNewBlob: DefaultDict[int, Set[int]] = defaultdict(set)
        newBlobToOldBlob: DefaultDict[int, Set[int]] = defaultdict(set)
        matchedOldBlob: List[int] = list()
        matchedNewBlob: List[int] = list()

        log.debug(f"{nbMatches.items()}")

        for m, counter in nbMatches.items():
            if counter >= self.min_num_kp_matches:  # original was 4
                oldBlobToNewBlob[m[0]].add(m[1])
                newBlobToOldBlob[m[1]].add(m[0])
                matchedOldBlob.append(m[0])
                matchedNewBlob.append(m[1])

        oldToNewBestMatch: Dict[int, Tuple[int, int]] = dict()

        for oldBlobId, oldBlobBbox in oldBlobAssoc.items():
            if oldBlobId not in matchedOldBlob:
                matchedOld = False
                previousArea = 0
                if not matchedOld:
                    self.mNoMatchBlobs.add(oldBlobId)  # 1-0 case
                else:
                    newBlobId = oldToNewBestMatch[oldBlobId][0]
                    oldBlobToNewBlob[oldBlobId].add(newBlobId)
                    newBlobToOldBlob[newBlobId].add(oldBlobId)
                    matchedOldBlob.append(oldBlobId)
                    matchedNewBlob.append(newBlobId)

        newToOldBestMatch: Dict[int, Tuple[int, int]] = dict()
        for newBlobId, newBlobBbox in newBlobAssoc.items():
            if newBlobId not in matchedNewBlob:
                matchedOld = False

                if matchedOld:
                    oldBlobId: int = newToOldBestMatch[newBlobId][0]
                    oldBlobToNewBlob[oldBlobId].add(newBlobId)
                    newBlobToOldBlob[newBlobId].add(oldBlobId)
                    matchedOldBlob.append(oldBlobId)
                    matchedNewBlob.append(newBlobId)
                else:
                    self.mNewBlobs.add(newBlobId)  # 0-1 case

        for oldBlobId, newBlobIds in oldBlobToNewBlob.items():
            if len(newBlobIds) == 1:  # 1-1
                newBlobId = list(newBlobIds)[0]
                if len(newBlobToOldBlob[newBlobId]) == 1:
                    self.mOneToOne.add(tuple([oldBlobId, newBlobId]))
                    matchedOldBlob.append(oldBlobId)
                    matchedNewBlob.append(newBlobId)
                else:  # N -1
                    assert type(newBlobToOldBlob[newBlobId]) == set
                    self.mNToOne.add(tuple([tuple(newBlobToOldBlob[newBlobId]), newBlobId]))
            else:  # 1 - N
                self.mOneToN.add(tuple([oldBlobId, tuple(newBlobIds)]))
