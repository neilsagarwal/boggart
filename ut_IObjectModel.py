import cv2
from typing import Dict, Set, List, Tuple, DefaultDict
from ut_util_classes import Blob, BlobRect, mergeBBoxes, Track, ObjectState, BlobEstimation, subtract_points, add_points, getRectangleIntersection, getArea, doesBoxContain, get_norm
from ut_classes import FeatureDetectorExtractorMatcher
import numpy as np
from collections import defaultdict
from munkres import Munkres
import math
from ut_util_classes import log
from copy import deepcopy

minimumMatchBetweenBlob = 2 

class IObjectModel():
    def __str__(self):
        return f"OM: ID={id(self)}]"

    def __repr__(self):
        return f"OM: ID={id(self)}]"


class ObjectModel(IObjectModel):

    def __init__(self, obj):
        self.mBlobs: Dict[int, Blob] = dict()
        self.mNumberFeatures = 0
        self.mLinkedObject = obj
        self.mTracks: List[Track] = list()  # list of tracks

    def getTracks(self):
        return self.mTracks

    def getBlobs(self):
        return self.mBlobs

    def getNumberRemovedFeatures(self):
        return self.mNumberFeatures

    def getLinkedObject(self):
        return self.mLinkedObject

    def addMergeBlob(self, ts: int, blob: Blob):
        if ts not in self.mBlobs:
            self.mBlobs[ts] = blob
        else:
            self.mBlobs[ts].setBoundingBox(mergeBBoxes(self.mBlobs[ts].getBoundingBox(), blob.getBoundingBox()))

    def replaceBlob(self, ts: int, blob: Blob):
        self.mBlobs[ts] = blob

    def addAndMatchPoint(self, ts, kp, des):
        newPointMatchedIdx: Set[int] = set()
        if len(self.mTracks) > 0 and len(kp) > 0:
            currentTracksDesc = []
            for track in self.mTracks:
                currentTracksDesc.append(track.des)
            matches = FeatureDetectorExtractorMatcher.match(np.array(currentTracksDesc), des)

            for m in matches:
                trackIdx = m.queryIdx
                pointIdx = m.trainIdx
                newPointMatchedIdx.add(pointIdx)
                self.mTracks[trackIdx].addPoint(ts, kp[pointIdx])
                self.mTracks[trackIdx].updateDescriptor(des[pointIdx])

        for i, (pt, d) in enumerate(zip(kp, des)):
            if i not in newPointMatchedIdx:
                t = Track(d)
                t.addPoint(ts, pt)
                self.mTracks.append(t)

    def handleLeaving(self) -> List:
        from ut_IObject import Object, IObject, ObjectGroup
        newObjectList: List[IObject] = list()

        if len(self.mBlobs) > 1:
            lastGoodTimestamp = -1
            sortedMBlobs = sorted(self.mBlobs.items())[::-1]
            # deal with currentBlobIt being used later
            currentBlobIt = sortedMBlobs[0]
            currentTimestamp = sortedMBlobs[0][0]
            for k, v in sortedMBlobs[1:]:
                if v.getState() != ObjectState.LEAVING and lastGoodTimestamp == -1:
                    lastGoodTimestamp = k
            log.debug("%d is the last good timestamp" % lastGoodTimestamp)
            if lastGoodTimestamp != -1:
                compatibleTracks: Set[Track] = set()
                for t in self.mTracks:
                    if t.getLastTimestamp() == currentTimestamp and t.getFirstTimestamp() <= lastGoodTimestamp:
                        compatibleTracks.add(t)
                if len(compatibleTracks) >= minimumMatchBetweenBlob:
                    log.debug("This is the same object. We have %d compatible tracks" % len(compatibleTracks))
                else:
                    # We will try to find the moment the object changed using blob area
                    sortedMBlobs = list(filter(lambda x: x[0] >= lastGoodTimestamp, sorted(self.mBlobs.items())))
                    objAtLastGoodTS = sortedMBlobs[0][1]
                    minArea = getArea(objAtLastGoodTS.getBoundingBox())
                    minAreaTS = lastGoodTimestamp
                    for ts, blob in sortedMBlobs[1:]:
                        tmpArea = getArea(blob.getBoundingBox())
                        if tmpArea < minArea:
                            minArea = tmpArea
                            minAreaTS = ts
                    # 1 Find which coordinates are outside ROI
                    bestTimestamp = minAreaTS
                    newObject = Object()
                    newObject.setState(ObjectState.OBJECT)
                    sortedMBlobs = list(filter(lambda x: x[0] >= bestTimestamp, sorted(self.mBlobs.items())))
                    for ts, b in sortedMBlobs:
                        b.setState(ObjectState.OBJECT)
                        newObject.getIObjectModel().addMergeBlob(ts, b)

                    # Let's take all objects that start at bestTS
                    newTrackList: List[Track] = list()
                    tracksToKeep: List[Track] = list()

                    for t in self.mTracks:
                        if t.getFirstTimestamp() >= bestTimestamp:
                            newTrackList.append(t)
                        else:
                            tracksToKeep.append(t)
                    self.mTracks = tracksToKeep

                    fpList = []
                    desc = []

                    for t in newTrackList:
                        fpList.append(sorted(t.getPointList().items())[-1][1])
                        desc.append(t.getDescriptor())
                    newTrackList.clear()

                    newObject.getIObjectModel().addPoint(currentTimestamp, fpList, desc)
                    newObjectList.append(newObject)

                    self.mBlobs = dict(list(filter(lambda x: x[0] < bestTimestamp, sorted(self.mBlobs.items()))))
            else:
                pass
                self.mLinkedObject.setState(ObjectState.OBJECT)
        else:
            pass
        return newObjectList

    def simplifyModel(self, ts):

        numberOfFrameToKeep = 3
        if numberOfFrameToKeep != -1:
            goodTracks = []
            minimumPointInTrack = 2
            for t in self.mTracks:
                oldTrack = ts > (t.getLastTimestamp() + numberOfFrameToKeep)
                if oldTrack and len(t.mPointList) < minimumPointInTrack:
                    self.mNumberFeatures += len(t.mPointList)
                else:
                    goodTracks.append(t)
            self.mTracks = goodTracks

    def getLastBoundingBox(self):
        assert len(self.mBlobs) > 0
        return sorted(self.mBlobs.items())[-1][1].mObjectBoundingBox

    def getLastTimestamp(self):
        if len(self.mBlobs) == 0:
            return 0
        else:
            return max(self.mBlobs.keys())

    def getLastStableBoundingBox(self):

        areaList: List[Tuple[int, List]] = []

        for ts, blob in sorted(self.mBlobs.items())[::-1][:10]:
            bb = blob.mObjectBoundingBox
            areaList.append(tuple([blob.getBBoxArea(), bb]))

        lastStableBox = -1
        found = False

        i = 0
        while i < len(areaList) and not found:
            found = True
            deltaSz = 0.15 * float(areaList[i][0])
            j = i + 1
            while j < len(areaList) and j-i < 4 and found:
                if (areaList[j][0] - areaList[i][0]) > deltaSz:
                    found = False
                j += 1
            if found:
                lastStableBox = i
            i += 1

        if lastStableBox == -1:
            log.debug("no stable box...")
            lastStableBox = 0

        return areaList[lastStableBox][1]

    def getMatchingPointMovement(self, otherObjModel):
        deltaMovement: List[float] = []
        matches: List[cv2.DMatch] = ObjectModel.getMatches(self, otherObjModel)
        tracksA: List[Track] = self.mTracks
        tracksB: List[Track] = otherObjModel.mTracks
        for m in matches:
            ptA = tracksA[m.queryIdx].getPointList()
            p1 = sorted(ptA.items())[-1][1]
            ptB = tracksB[m.trainIdx].getPointList()
            p2 = sorted(ptB.items())[-1][1]
            deltaMovement.append(get_norm(subtract_points(p2, p1)))
        return deltaMovement

    def getMatches(mA: IObjectModel, mB: IObjectModel) -> List:
        tracksA = mA.getTracks()
        tracksB = mB.getTracks()
        if len(tracksA) > 0 and len(tracksB) > 0:
            descA = []
            descB = []
            for t in tracksA:
                descA.append(t.des)
            for t in tracksB:
                descB.append(t.des)
            return FeatureDetectorExtractorMatcher.match(np.array(descA), np.array(descB))
        return []

    def getFirstTimestamp(self):
        if len(self.mBlobs) == 0:
            return 0
        else:
            return min(self.mBlobs.keys())

    def interpolateMissingFrame(self, maxGap: int):
        interpolatedBlobList: List[Tuple[int, Blob]] = list()
        sortedMBlobs = sorted(self.mBlobs.items())
        if len(sortedMBlobs) > 1:
            for i in range(1, len(sortedMBlobs)):
                prevTS, prevBlob = sortedMBlobs[i-1]
                currTS, currBlob = sortedMBlobs[i]
                deltaTS = currTS - prevTS
                if deltaTS > 1 and deltaTS <= maxGap:
                    prevRect = prevBlob.mObjectBoundingBox
                    currRect = currBlob.mObjectBoundingBox
                    for t in range(prevTS + 1, currTS):
                        ratio = (float(t) - prevTS)/deltaTS
                        ratioStart = 1-ratio
                        ratioEnd = ratio
                        x1 = int(ratioStart*prevRect[0] + ratioEnd*currRect[0])
                        y1 = int(ratioStart*prevRect[1] + ratioEnd*currRect[1])
                        x2 = int(ratioStart*prevRect[2] + ratioEnd*currRect[2])
                        y2 = int(ratioStart*prevRect[3] + ratioEnd*currRect[3])
                        interpolatedBlobList.append((t, Blob([x1, y1, x2, y2], currBlob.getState())))
            if len(interpolatedBlobList) > 0:
                log.debug("interpolation!!")
                for elem in interpolatedBlobList:
                    assert type(elem[1]) == Blob
                    self.mBlobs[elem[0]] = elem[1]

    def correctGroupObservation(self):
        goodBlobs: Dict[int, Blob] = dict()
        for ts, blob in self.mBlobs.items():
            if blob.mState not in [ObjectState.INGROUP, ObjectState.OBJECTGROUP]:
                goodBlobs[ts] = blob
        self.mBlobs = goodBlobs

    def addTrack(self, t: Track):
        self.mTracks.append(t)

    def addPoint(self, ts, kps, des):
        for k, d in zip(kps, des):
            t = Track(d)
            t.addPoint(ts, k)
            self.addTrack(t)

    def getMatchingPointNumber(A: IObjectModel, B: IObjectModel) -> int:
        return len(ObjectModel.getMatches(A, B))

    # self is the original om, B is the om that we want to separate into
    def extractObjectModel(self, B: "ObjectModel"):
        A = self
        tracksA = A.mTracks
        tracksB = B.mTracks

        if len(tracksA) == 0 or len(tracksB) == 0:
            return

        descA = list(map(lambda t: t.des[::], tracksA))
        descB = list(map(lambda t: t.des[::], tracksB))
        matches = FeatureDetectorExtractorMatcher.match(np.array(descA), np.array(descB))
        trackToMove: Set[Track] = set()
        timestampToUpdate: Set[int] = set()

        # for all the trajectories in A that match with trajectories in B,
        # add all the points in those trajectories to the corresponding trajectory in B

        for m in matches:
            at: Track = tracksA[m.queryIdx]
            bt: Track = tracksB[m.trainIdx]
            aPointList = at.mPointList
            bPointList = bt.mPointList
            for k, v in aPointList.items():
                bPointList[k] = v
            trackToMove.add(at)
            # all the tracks in A that are also in B, we want to move!

            for k in aPointList.keys():
                timestampToUpdate.add(k)

        # all blobs when the tracks switched from A-> B, set as ingroup!
        for ts in timestampToUpdate:
            if ts in A.mBlobs:
                A.mBlobs[ts].setState(ObjectState.INGROUP)
                B.mBlobs[ts] = deepcopy(A.mBlobs[ts]) # because *itA,, pointer fix...

        newA: List[Track] = list()
        for t in tracksA:
            if t not in trackToMove:
                newA.append(t)

        self.mTracks = newA

    def addAndMatchTracks(self, tracksB: List[Track]):
        usedTracks: Set[int] = set()
        tracksA: List[Track] = self.mTracks

        if len(tracksA) > 0 and len(tracksB) > 0:

            descA = list(map(lambda t: t.des, tracksA))
            descB = list(map(lambda t: t.des, tracksB))

            matches = FeatureDetectorExtractorMatcher.match(np.array(descA), np.array(descB))

            for matchIdx in range(len(matches)):
                trackAIdx = matches[matchIdx].queryIdx
                trackBIdx = matches[matchIdx].trainIdx
                A: Track = tracksA[trackAIdx]
                B: Track = tracksB[trackBIdx]

                if A.getLastTimestamp() < B.getFirstTimestamp() or B.getLastTimestamp() < A.getFirstTimestamp():
                    pointListA: Dict = A.getPointList()
                    for k, v in B.getPointList().items():
                        pointListA[k] = v
                    usedTracks.add(trackBIdx)
                    del B

        for i in range(len(tracksB)):
            if i not in usedTracks:
                self.mTracks.append(tracksB[i])

    def addBlobs(self, blobsB: Dict[int, Blob]):

        blobsA: Dict[int, Blob] = self.mBlobs

        blobTimeStartA = 0 if len(blobsA) == 0 else min(blobsA.keys())
        blobTimeEndA = 0 if len(blobsA) == 0 else max(blobsA.keys())

        blobTimeStartB = 0 if len(blobsB) == 0 else min(blobsB.keys())
        blobTimeEndB = 0 if len(blobsB) == 0 else max(blobsB.keys())

        if blobTimeEndA < blobTimeStartB or blobTimeEndB < blobTimeStartA:
            for k, v in blobsB.items():
                blobsA[k] = v
        else:
            for k, v in blobsB.items():
                if k in blobsA:
                    blobsA[k].setBoundingBox(mergeBBoxes(blobsA[k].getBoundingBox(), v.getBoundingBox()))
                else:
                    blobsA[k] = v

    def clearObjectModel(self):
        self.mBlobs.clear()
        self.mTracks.clear()

    def moveObjectModel(self, otherModel: IObjectModel):
        otherObjectTracks: List[Track] = otherModel.getTracks()
        self.addAndMatchTracks(otherObjectTracks)
        otherObjectBlob: Dict[int, Blob] = otherModel.getBlobs()
        self.addBlobs(otherObjectBlob)
        otherModel.clearObjectModel()

    # this seems to take a while.... TODO: optimize!
    def updateInGroupBlobs(self):
        blobToUpdate: Dict[int, Blob] = dict()
        # bool indicate if less or more reliable (false are more reliable reliable)
        goodBlobs: Dict[int, Tuple[Blob, bool]] = dict()

        # log.debug(f"blobids: {sorted([id(x) for x in self.mBlobs.values()])}")

        for k, b in self.mBlobs.items():
            if b.getState() in [ObjectState.INGROUP, ObjectState.OBJECTGROUP]:
                blobToUpdate[k] = b
            else:
                partialObs: bool = b.mState in [ObjectState.LEAVING, ObjectState.ENTERING]
                goodBlobs[k] = tuple([b, partialObs])

        # Everything is sorted here since mBlobs is sorted by timestamp

        if len(blobToUpdate) == 0:
            return

        # (1) look for tracks
        interestTrack: List[Track] = list()
        startTs: int = min(blobToUpdate.keys())
        endTs: int = max(blobToUpdate.keys())
        for t in self.mTracks:
            if (t.getFirstTimestamp() <= startTs and t.getLastTimestamp() >= startTs) or (t.getLastTimestamp() >= endTs and t.getFirstTimestamp() <= endTs):
                interestTrack.append(t)

        timestampToBlobApproximation: DefaultDict[int, List[BlobEstimation]] = defaultdict(list)

        for t in interestTrack:
            pointList: Dict = t.getPointList()
            validObs = False
            lastValidTimestamp = 0
            partialObs = False
            lastValidPosRelCenter = ()
            lastValidWidth = 0
            lastValidHeight = 0
            lastObs: List[Tuple] = list()
            for ts, pt in sorted(pointList.items()):
                if blobToUpdate.get(ts) is not None and validObs:
                    projPt = list(pt).copy() 
                    estimatedProjCentroid = subtract_points(projPt, lastValidPosRelCenter) 
                    timestampToBlobApproximation[ts].append(BlobEstimation(estimatedProjCentroid, lastValidHeight, lastValidWidth, False, abs(lastValidTimestamp - ts)))
                else:
                    b = goodBlobs.get(ts)
                    if b is not None:
                        assert type(b[1]) == bool
                        validObs = True
                        lastValidTimestamp = ts
                        centroid = b[0].getProjectedCentroid()
                        point = pt
                        lastValidPosRelCenter = subtract_points(point, centroid)
                        bb = b[0].getProjectedBoundingBox()
                        lastValidWidth = bb[2] - bb[0]
                        lastValidHeight = bb[3] - bb[1]

                        if len(lastObs) > 0:
                            for timestamp, projPt in lastObs:
                                estimatedProjCentroid = add_points(projPt, lastValidPosRelCenter)
                                timestampToBlobApproximation[timestamp].append(BlobEstimation(estimatedProjCentroid, lastValidHeight, lastValidWidth, False, abs(lastValidTimestamp-timestamp)))
                            lastObs.clear()
                    elif not validObs:
                        projPt = pt  # get project point same because homography is identity matrix
                        lastObs.append((ts, projPt))


        for timestamp, estimationList in sorted(timestampToBlobApproximation.items()):
            groupBB = blobToUpdate[timestamp].getBoundingBox()
            estimationList = sorted(estimationList, key=lambda x: x.getTemporalTimestamp())

            if len(estimationList) > 0:

                xPos: List[int] = list()
                yPos: List[int] = list()
                width: List[int] = list()
                height: List[int] = list()

                nbEstimation = 0
                estim_idx = 0
                while estim_idx < len(estimationList) and nbEstimation < 10:
                    xPos.append(estimationList[estim_idx].mCentroid[0])
                    yPos.append(estimationList[estim_idx].mCentroid[1])
                    width.append(estimationList[estim_idx].mWidth)
                    height.append(estimationList[estim_idx].mHeight)
                    nbEstimation += 1
                    estim_idx += 1

                if len(estimationList) > 3:
                    midIdx = math.floor(len(xPos)/2)
                    xPos = sorted(xPos)
                    yPos = sorted(yPos)
                    width = sorted(width)
                    height = sorted(height)

                    x1 = xPos[midIdx] - width[midIdx]/2
                    y1 = yPos[midIdx] - height[midIdx]/2
                    x2 = xPos[midIdx] + width[midIdx]/2
                    y2 = yPos[midIdx] + height[midIdx]/2

                    estimatedBB = [x1, y1, x2, y2]

                    intersection: List = getRectangleIntersection(groupBB, estimatedBB)
                    x1, y1, x2, y2 = intersection
                    if x2-x1 > 0 and y2-y1 > 0:
                        log.debug(f"Setting bbox @ {timestamp} -> {intersection}")
                        blobToUpdate[timestamp].setBoundingBox(intersection)
                        blobToUpdate[timestamp].setState(ObjectState.ESTIMATED)
                        goodBlobs[timestamp] = (blobToUpdate[timestamp], False)
                        
        interpolatedBox: int = 0
        for timestamp, b in sorted(blobToUpdate.items()):
            if b is None:
                continue
            elif b.getState() in [ObjectState.INGROUP, ObjectState.OBJECTGROUP]:
                groupBB: List[float] = b.getBoundingBox()

                tempGoodBlobs: List[Tuple[int, Tuple[Blob, bool]]] = sorted(goodBlobs.items())

                if len(goodBlobs) > 0 and tempGoodBlobs[0][0] != timestamp and tempGoodBlobs[-1][0] > timestamp:
                    firstIdxOfUpper = 10000000
                    for idx, (k, v) in enumerate(tempGoodBlobs):
                        if k > timestamp and idx < firstIdxOfUpper:
                            firstIdxOfUpper = idx
                    upper = tempGoodBlobs[firstIdxOfUpper]
                    lower = tempGoodBlobs[firstIdxOfUpper-1]
                    if upper[0] > timestamp and lower[0] < timestamp:
                        before = lower[1][0].getBoundingBox()
                        after = upper[1][0].getBoundingBox()
                        timeLapse: int = upper[0] - lower[0]
                        ratio = (float(timestamp) - float(lower[0])) / timeLapse
                        ratioStart = float(1) - ratio
                        ratioEnd = ratio

                        x1: float = ratioStart * before[0] + ratioEnd * after[0]
                        y1: float = ratioStart * before[1] + ratioEnd * after[1]
                        x2: float = ratioStart * before[2] + ratioEnd * after[2]
                        y2: float = ratioStart * before[3] + ratioEnd * after[3]

                        estimatedBB = [x1, y1, x2, y2]
                        intersection: List = getRectangleIntersection(groupBB, estimatedBB)
                        x1, y1, x2, y2 = intersection
                        if x2-x1 > 0 and y2-y1 > 0:
                            b.setBoundingBox(intersection)
                            b.setState(ObjectState.ESTIMATED)
                            interpolatedBox += 1

        if interpolatedBox != 0:
            log.debug("INTERPOLATED %d boxes by updateInGroupBlobs" % interpolatedBox)

    def removeTracksNoDelete(self, tracksToRemove: Set[Track]):
        goodTracks: List = list()
        for t in self.mTracks:
            if t not in tracksToRemove:
                goodTracks.append(t)
        self.mTracks = goodTracks


class ObjectModelGroup(IObjectModel):

    def __init__(self, og):
        self.mModelGroup: ObjectModel = ObjectModel(None)
        self.mLinkedGroup = og
        self.mTrackListDirty: bool = True
        self.mObjectModelListNonOwner: List[ObjectModel] = list()
        self.mFullTrackList: List[Track] = list()
        self.mTrackToModel: Dict[Track, ObjectModel] = dict()
        self.mFullBlobList: Dict[int, Blob] = dict()

    def addMergeBlob(self, ts, b: Blob):
        self.mModelGroup.addMergeBlob(ts, b)
        for m in self.mObjectModelListNonOwner:
            m.addMergeBlob(ts, deepcopy(b))

    def addBlobs(self, blobs: Dict[int, Blob]):
        self.mModelGroup.addBlobs(blobs)

    def getLinkedObject(self):
        return self.mLinkedGroup

    def getLastTimestamp(self):
        lastTimestamp: int = self.mModelGroup.getLastTimestamp()
        for m in self.mObjectModelListNonOwner:
            if lastTimestamp < m.getLastTimestamp():
                lastTimestamp = m.getLastTimestamp
        return lastTimestamp

    def getFirstTimestamp(self):
        firstTimestamp = self.mModelGroup.getFirstTimestamp()
        for m in self.mObjectModelListNonOwner:
            if m.getFirstTimestamp() > firstTimestamp:
                firstTimestamp = m.getFirstTimestamp()
        return firstTimestamp

    def addTrack(self, t: Track):
        self.mModelGroup.addTrack(t)
        self.mTrackListDirty = True

    def updateTrackList(self):
        if self.mTrackListDirty:
            self.mTrackToModel.clear()
            for t in self.mModelGroup.getTracks():
                self.mTrackToModel[t] = self.mModelGroup
            self.mFullTrackList.clear()
            self.mFullTrackList.extend(self.mModelGroup.getTracks())
            for objModel in self.mObjectModelListNonOwner:
                self.mFullTrackList.extend(objModel.getTracks())
                for t in objModel.getTracks():
                    self.mTrackToModel[t] = objModel
            self.mTrackListDirty = False

    def getTracks(self) -> List[Track]:
        if self.mTrackListDirty:
            self.updateTrackList()
        return self.mFullTrackList

    def moveObjectModel(self, otherObject: IObjectModel):
        self.mTrackListDirty = True
        otherObjectBlob: Dict[int, Blob] = otherObject.getBlobs()
        self.addBlobs(otherObjectBlob)
        otherObject.clearObjectModel()

    def addObjectModel(self, om: ObjectModel):
        self.mObjectModelListNonOwner.append(om)

    def removeObjectModel(self, om: ObjectModel):
        self.mObjectModelListNonOwner.remove(om)

    def addUnmatchedGroupBlobToExistingObjects(self, groupObjectList: List[ObjectModel] = None):
        if groupObjectList is None:
            groupObjectList = self.mObjectModelListNonOwner
        from ut_IObject import Object, IObject, ObjectGroup
        matchedGroupTimestamp: DefaultDict[int, List[Object]] = defaultdict(list)
        for obj in groupObjectList:
            objectBlobList = obj.getBlobs()
            for timestamp, blob in sorted(objectBlobList.items()):
                if blob.getState() not in [ObjectState.INGROUP, ObjectState.OBJECTGROUP]:
                    matchedGroupTimestamp[timestamp].append(obj.getLinkedObject())

        if len(matchedGroupTimestamp) > 0:
            tracks = self.mModelGroup.getTracks()
            groupBlobs = self.mModelGroup.getBlobs()
            for timestamp, blob in sorted(groupBlobs.items()):
                missingBlobIt = matchedGroupTimestamp.get(timestamp)
                if missingBlobIt is None:
                    log.debug("Missing group blob at %d. Trying to associate it" % timestamp)

                    tempMatchedGroupTimestamp: List[Tuple[int, List[Object]]] = sorted(matchedGroupTimestamp.items())
                    if len(tempMatchedGroupTimestamp) > 1 and tempMatchedGroupTimestamp[0][0] < timestamp:
                        firstIdxOfUpper = 10000000
                        for idx, (k, v) in enumerate(tempMatchedGroupTimestamp):
                            if k >= timestamp and idx < firstIdxOfUpper:
                                firstIdxOfUpper = idx
                        if firstIdxOfUpper == 10000000:
                            firstIdxOfUpper = -1
                        firstElementGreaterOrEqual = tempMatchedGroupTimestamp[firstIdxOfUpper]
                        elementBefore = tempMatchedGroupTimestamp[firstIdxOfUpper-1]
                        bestObject = None
                        if len(elementBefore[1]) == 1:
                            bestObject = elementBefore[1][0]
                        else:
                            # we will discriminate by size
                            boundingBoxArea = getArea(blob.getBoundingBox())
                            candidateAreaDelta = -1
                            candidateBlobList = elementBefore[1]
                            for cand in candidateBlobList:
                                candBlob: Blob = cand.getIObjectModel().getBlobs().get(elementBefore[0])
                                if candBlob is not None:
                                    deltaArea = abs(getArea(candBlob.getBoundingBox()) - boundingBoxArea)
                                    if candidateAreaDelta == -1 or deltaArea < candidateAreaDelta:
                                        candidateAreaDelta = deltaArea
                                        bestObject = cand
                        if bestObject is not None:
                            b = blob
                            b.setState(ObjectState.ESTIMATED)
                            tracksToRemove: Set[Track] = set()
                            for t in tracks:
                                if t.getFirstTimestamp() == timestamp:
                                    pt = sorted(t.getPointList().items())[0][1]
                                    if doesBoxContain(b.getBoundingBox(), pt):
                                        tracksToRemove.add(t)
                                        bestObject.getObjectModel().addTrack(t)
                            log.debug("%d tracks added from group" % len(tracksToRemove))
                            if len(tracksToRemove) > 0:
                                self.mModelGroup.removeTracksNoDelete(tracksToRemove)
                            bestObject.getIObjectModel().replaceBlob(timestamp, b)
                            log.debug("Adding group blob %d at %s" % (timestamp, bestObject.getObjectId()))
                        else:
                            continue


    def handleSplit(self, modelList: List[ObjectModel], trackerObjectList: List, ts: int):
        from ut_IObject import Object, IObject, ObjectGroup

        outAssociation: List[Tuple] = list()
        groupObjectList: List[ObjectModel] = self.mObjectModelListNonOwner
        similarityMatrix = np.zeros((len(self.mObjectModelListNonOwner), len(modelList)))
        associationList: List[Tuple[ObjectModel, ObjectModel]] = list()
        lostObjects: List[ObjectModel] = list()
        newObjects: List[ObjectModel] = list()
        self.updateTrackList()

        bestMatches: Dict[ObjectModel, Tuple[ObjectModel, int]] = dict()
        currentTimestamp: int = ts

        # Step 1: We verify the object with more than 3 matches and optimize their association with the hungarian algorithm
        totalNbMatches = 0
        maxDist = float(1)

        # for objects of the model group
        for row, obj in enumerate(self.mObjectModelListNonOwner):
            for col, model in enumerate(modelList):
                matches = ObjectModel.getMatches(obj, model)
                nbMatches = len(matches)
                log.debug("%s has %d matches" % (obj.getLinkedObject().getObjectId(), nbMatches))
                nbMatches = nbMatches if nbMatches > minimumMatchBetweenBlob else 0

                if nbMatches > 0:
                    similarityMatrix[row][col] = nbMatches
                    objMatch = bestMatches.get(obj)
                    if objMatch is None:
                        bestMatches[obj] = (model, nbMatches)
                    else:
                        if nbMatches > objMatch[1]:
                            bestMatches[obj] = (model, nbMatches)
                    totalNbMatches += nbMatches
                else:
                    similarityMatrix[row][col] = 0

        log.debug(f"{similarityMatrix}")

        usedGroupObject: Set[ObjectModel] = set()
        usedNewObject: Set[ObjectModel] = set()

        if totalNbMatches > 0:
            similarityMatrix = 1-(similarityMatrix)/float(totalNbMatches)
            m = Munkres()
            if similarityMatrix.shape[0] != similarityMatrix.shape[1]:
                similarityMatrix = similarityMatrix.tolist()
            indices = m.compute(similarityMatrix)
            log.debug(f"{indices}")
            for row, col in indices:
                associationList.append((self.mObjectModelListNonOwner[row], modelList[col]))
                usedGroupObject.add(self.mObjectModelListNonOwner[row])
                usedNewObject.add(modelList[col])
        else:
            pass

        # We use the old blobs of the unmatched group and we associate them with the new one using the distance
        if len(usedGroupObject) != len(self.mObjectModelListNonOwner):
            log.debug("# Used Group Objects != # Initial Group Objects")
            unusedGroupObject: List[ObjectModel] = list()
            for obj in self.mObjectModelListNonOwner:
                if obj not in usedGroupObject:
                    best = bestMatches.get(obj)
                    if best is not None:
                        associationList.append((obj, best[0]))
                        usedGroupObject.add(obj)
                    else:
                        unusedGroupObject.append(obj)

            for obj in unusedGroupObject:
                greedyMatch = bestMatches.get(obj)
                if greedyMatch is not None:
                    associationList.append((obj, greedyMatch[0]))
                else:
                    # Before adding to the lost the objects, we try to find if there is an overlap
                    rect = obj.getLastBoundingBox()
                    lastBBArea = getArea(rect)
                    bestOverlapArea = 0
                    bestOverlapModel: ObjectModel = None

                    for model in modelList:
                        lastBB = model.getLastBoundingBox()
                        intersection = getRectangleIntersection(rect, lastBB)
                        x1, y1, x2, y2 = intersection
                        if x2-x1 > 0 and y2-y1 > 0:
                            tmpArea = getArea(intersection)
                            if tmpArea > bestOverlapArea:
                                bestOverlapArea = tmpArea
                                bestOverlapModel = model

                    overlapRatio = 0
                    if lastBBArea > 0:
                        overlapRatio = float(bestOverlapArea)/lastBBArea
                    if bestOverlapModel is not None and overlapRatio > 0.7:
                        associationList.append((obj, bestOverlapModel))
                        usedNewObject.add(bestOverlapModel)
                        log.debug("Split association with area overlap for %s with area overlap of %f" % (obj.getLinkedObject().getObjectId(), overlapRatio))
                    else:
                        lostObjects.append(obj)

        # At this point, we should have matches all the blob history with the new blob. We will now look at the new blobs
        if len(usedNewObject) != len(modelList):
            log.debug("# of Used New Objects != Num Objects to Split Into")
            for newObj in modelList:
                if newObj not in usedNewObject:
                    newObjects.append(newObj)

        # Lost objects are added to the lost object list
        for lostObj in lostObjects:
            obj: Object = lostObj.getLinkedObject()
            self.mLinkedGroup.removeObject(obj)
            obj.setState(ObjectState.LOST)
            trackerObjectList.append(obj)
            self.mObjectModelListNonOwner.remove(lostObj)

        # New objects are created

        from ut_IObject import Object

        log.debug(f"Creating New Objects: {newObjects}")
        for newObjIt in newObjects:
            newObject: Object = Object()
            assert type(newObject) == Object
            newObject.getObjectModel().moveObjectModel(newObjIt)
            newObject.setState(ObjectState.HYPOTHESIS)
            groupObjectList.append(newObject.getObjectModel())
            trackerObjectList.append(newObject)
            outAssociation.append((newObject, newObjIt))

        associationGroup: DefaultDict[ObjectModel, List[ObjectModel]] = defaultdict(list)
        for assoc in associationList:
            associationGroup[assoc[1]].append(assoc[0])

        log.debug(f"associationGroup: {associationGroup}")

        for newBlob, associatedObjects in associationGroup.items():
            if len(associatedObjects) == 1:
                obj: Object = associatedObjects[0].getLinkedObject()
                associatedObjects[0].moveObjectModel(newBlob)
                self.mLinkedGroup.removeObject(obj)
                self.mObjectModelListNonOwner.remove(associatedObjects[0])
                obj.setState(ObjectState.OBJECT)
                trackerObjectList.append(obj)
                obj.getObjectModel().updateInGroupBlobs()
                outAssociation.append((obj, newBlob))
            elif len(associatedObjects) > 1:
                og: ObjectGroup = ObjectGroup()
                for obj in associatedObjects:
                    og.addObject(obj.getLinkedObject())
                    self.mLinkedGroup.removeObject(obj.getLinkedObject())
                    self.mObjectModelListNonOwner.remove(obj)
                og.getObjectModelGroup().moveObjectModel(newBlob)
                trackerObjectList.append(og)
                outAssociation.append((og, newBlob))
            else:
                pass

        self.addUnmatchedGroupBlobToExistingObjects(groupObjectList)

        return outAssociation

    def addAndMatchPoint(self, ts, kp: List, des: List):
        self.updateTrackList()

        newPointMatchedIdx: Set[int] = set()

        if len(self.mFullTrackList) > 0 and len(kp) > 0:
            currentTracksDesc = []
            for track in self.mFullTrackList:
                currentTracksDesc.append(track.des[::])
            matches = FeatureDetectorExtractorMatcher.match(np.array(currentTracksDesc), des)

            for m in matches:
                trackIdx = m.queryIdx
                pointIdx = m.trainIdx
                newPointMatchedIdx.add(pointIdx)
                t: Track = self.mFullTrackList[trackIdx]
                t.addPoint(ts, deepcopy(kp[pointIdx]))
                t.updateDescriptor(des[pointIdx][::])

        for i in range(len(kp)):
            if i not in newPointMatchedIdx:
                t: Track = Track(des[i][::])
                t.addPoint(ts, deepcopy(kp[i]))
                self.addTrack(t)

    def getLastBoundingBox(self):
        return self.mModelGroup.getLastBoundingBox()

    def replaceBlob(self, ts: int, b: Blob):
        log.warn("replaceBlob() is not implemented")
        self.mModelGroup.replaceBlob(ts, b)

    def clearObjectModel(self):
        self.mModelGroup.clearObjectModel()
        self.mObjectModelListNonOwner.clear()
        self.mTrackListDirty = True

    def simplifyModel(self, ts):
        pass