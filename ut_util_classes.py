from enum import Enum
from typing import List, Tuple, DefaultDict
import math
import numpy as np
import logging as log
log.basicConfig(format='%(levelname)s:%(message)s', level=log.INFO)

class ObjectState(Enum):
    UNDEFINED = 1
    ENTERING = 2
    HYPOTHESIS = 3
    OBJECT = 4
    LOST = 5
    OBJECTGROUP = 6
    INGROUP = 7
    LEAVING = 8
    DELETED = 9
    SAVEANDELETE = 10
    ESTIMATED = 11
    STOPPED = 12
    ALL = 13


class Blob:
    def __init__(self, bb: List, objState: ObjectState):
        self.mState: ObjectState = objState
        self.mObjectBoundingBox: List = bb
        self.mProjectedBoundingBox: List = list()
        self.mCentroid: Tuple = ()
        self.mProjectedCentroid: Tuple = ()
        self.setBoundingBox(bb)

    def __repr__(self) -> str:
        return str(self.mObjectBoundingBox)

    def getBBoxArea(self) -> int:
        x1, y1, x2, y2 = self.mObjectBoundingBox
        return (x2-x1) * (y2-y1)

    def getCentroid(self) -> List:
        return self.mCentroid

    def getBoundingBox(self) -> List:
        return self.mObjectBoundingBox

    def getState(self) -> ObjectState:
        return self.mState

    def getProjectedCentroid(self) -> Tuple:
        return self.mProjectedCentroid

    def getProjectedBoundingBox(self) -> List:
        return self.mProjectedBoundingBox

    def setState(self, state: ObjectState) -> None:
        self.mState = state

    def setBoundingBox(self, bbox: List) -> None:
        self.mObjectBoundingBox = bbox
        x1, y1, x2, y2 = bbox
        self.mCentroid = (x1 + (x2-x1)/2, y1 + (y2-y1)/2)
        afterHomography: List = list()
        width = x2-x1
        height = y2-y1

        beforeHomography: List = [self.mCentroid]
        beforeHomography.append([self.mCentroid[0]-width/2, self.mCentroid[1]])
        beforeHomography.append([self.mCentroid[0]+width/2, self.mCentroid[1]])
        beforeHomography.append([self.mCentroid[0], self.mCentroid[1] - height/2])
        beforeHomography.append([self.mCentroid[0], self.mCentroid[1] + height/2])

        afterHomography = beforeHomography

        self.mProjectedCentroid = afterHomography[0]
        x1 = afterHomography[1][0]
        x2 = afterHomography[2][0]
        y1 = afterHomography[3][1]
        y2 = afterHomography[4][1]
        self.mProjectedBoundingBox = [x1, y1, x2, y2]


def getRectangleIntersection(A, B):
    b1x1, b1y1, b1x2, b1y2 = A
    b2x1, b2y1, b2x2, b2y2 = B
    xmax = min(b1x2, b2x2)
    xmin = max(b1x1, b2x1)
    ymax = min(b1y2, b2y2)
    ymin = max(b1y1, b2y1)
    return [xmin, ymin, xmax, ymax]


def getRectangleIntersectionArea(A, B):
    xmin, ymin, xmax, ymax = getRectangleIntersection(A, B)
    return (ymax-ymin) * (xmax-xmin)


class BlobRect:

    def __init__(self, mBoundingBox, mBlob):
        self.mBoundingBox = mBoundingBox
        self.mBlob = mBlob

    @staticmethod
    def getAreaIntersection(A, B) -> int:
        b1x1, b1y1, b1x2, b1y2 = A.mBoundingBox
        b2x1, b2y1, b2x2, b2y2 = B.mBoundingBox

        xmax = min(b1x2, b2x2)
        xmin = max(b1x1, b2x1)
        ymax = min(b1y2, b2y2)
        ymin = max(b1y1, b2y1)

        width = xmax - xmin
        height = ymax - ymin

        if height <= 0 or width <= 0:
            return 0

        AIntersection = A.mBlob[ymin-A.mBoundingBox[1]: ymax - A.mBoundingBox[1], xmin-A.mBoundingBox[0]: xmax-A.mBoundingBox[0]]
        BIntersection = B.mBlob[ymin-B.mBoundingBox[1]: ymax - B.mBoundingBox[1], xmin-B.mBoundingBox[0]: xmax-B.mBoundingBox[0]]

        return np.sum(np.bitwise_and(AIntersection, BIntersection))

    @staticmethod
    def onEdgeOfFrame(bbox):
        x1, y1, x2, y2 = bbox
        return x1 <= 2 or y1 <= 2 or x2 >= 1920 - 2 or y2 >= 1080-2

    @staticmethod
    def mergeBlobRect(A: "BlobRect", B: "BlobRect"):
        newBbox = mergeBBoxes(A.mBoundingBox, B.mBoundingBox)
        x1, y1, x2, y2 = newBbox
        newBlob = np.zeros([y2-y1, x2-x1], A.mBlob.dtype)

        deltaX = A.mBoundingBox[0] - newBbox[0]
        deltaY = A.mBoundingBox[1] - newBbox[1]
        h,w = A.mBlob.shape
        newBlob[deltaY:h+deltaY, deltaX:w+deltaX] = A.mBlob

        deltaX = B.mBoundingBox[0] - newBbox[0]
        deltaY = B.mBoundingBox[1] - newBbox[1]
        h,w = B.mBlob.shape
        newBlob[deltaY:h+deltaY, deltaX:w+deltaX] = np.bitwise_or(newBlob[deltaY:h+deltaY, deltaX:w+deltaX], B.mBlob) 

        return BlobRect(newBbox, newBlob)


def mergeBBoxes(bb1, bb2):
    x1 = min(bb1[0], bb2[0])
    y1 = min(bb1[1], bb2[1])
    x2 = max(bb1[2], bb2[2])
    y2 = max(bb1[3], bb2[3])
    return [x1, y1, x2, y2]


class Track:
    def __init__(self, des):
        self.des = des
        self.mPointList = dict()

    def addPoint(self, ts, kp):
        self.mPointList[ts] = kp

    def updateDescriptor(self, des):
        self.des = des.copy()

    def getDescriptor(self):
        return self.des

    def getLastTimestamp(self) -> int:
        if len(self.mPointList) == 0:
            return 0
        else:
            return max(self.mPointList.keys())

    def getFirstTimestamp(self) -> int:
        if len(self.mPointList) == 0:
            return 0
        else:
            return min(self.mPointList.keys())

    def getPointList(self):
        return self.mPointList


def dilate(rect, dilation):
    x1, y1, x2, y2 = rect
    w = x2-x1
    h = y2-y1
    xAdd = float(dilation)/2 * w
    yAdd = float(dilation)/2 * h

    posX = max(0, math.floor(x1-xAdd))
    posY = max(0, math.floor(y1-yAdd))
    return [posX, posY, posX + math.ceil(w + 2 * xAdd), posY + math.ceil(h + 2 * yAdd)]


def getArea(bbox):
    x1, y1, x2, y2 = bbox
    return (x2-x1) * (y2-y1)


def getBoxDims(bbox):
    x1, y1, x2, y2 = bbox
    return (x2-x1, y2-y1)


def doesBoxContain(bbox, pt):
    x1, y1, x2, y2 = bbox
    x, y = pt
    return x1 <= x < x2 and y1 <= y < y2


def blobListOverlap(A: List[Tuple[int, BlobRect]], B: List[Tuple[int, BlobRect]]):
    overlap = False
    i = 0
    while i < len(A) and not overlap:
        j = 0
        while j < len(B) and not overlap:
            if BlobRect.getAreaIntersection(A[i][1], B[j][1]) > 0:
                overlap = True
            j += 1
        i += 1
    return overlap


class BlobEstimation:
    def __init__(self, centroid, height, width, partialObs: bool, tempDistance):
        self.mCentroid = centroid
        self.mPartialObservation = partialObs
        self.mHeight = height
        self.mWidth = width
        self.mTemporalDistance = tempDistance

    def getTemporalTimestamp(self):
        return self.mTemporalDistance


def subtract_points(a, b):
    return [a[0]-b[0], a[1]-b[1]]


def add_points(a, b):
    return [a[0]+b[0], a[1]+b[1]]


def get_norm(pt):
    return math.sqrt(pt[0] * pt[0] + pt[1] * pt[1])
