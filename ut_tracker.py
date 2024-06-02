from typing import Set, DefaultDict, List, Tuple, Dict
from collections import defaultdict, Counter
from ut_classes import FrameAssociation, FeatureDetectorExtractorMatcher, BlobDetector, PointsBlob
from ut_IObject import Object, IObject, ObjectGroup
from ut_IObjectModel import ObjectModel, ObjectModelGroup
from ut_util_classes import ObjectState, Blob, dilate, getArea, getBoxDims, blobListOverlap, BlobRect, getRectangleIntersectionArea
import cv2
from ut_util_classes import log
import pickle
from copy import deepcopy
import numpy as np
from pathlib import Path
import time

IGNORE_MASK = False

class Tracker:

    morph_open_kernel = (5, 5)
    morph_close_kernel = (5, 5)
    # 3 according to line 38 of BlobTrackerAlgorithmParams.h...
    maxFrameHypothesis = 3
    maximumLostFrame = 10
    verifyReEnteringObject = True
    start_ts = -1
    kps_fname_template = None
    min_num_kp_matches = 4

    kps_matches_template = None
    kps_loc_template = None
    kps_raw_template = None

    def __init__(self, start_ts=0):
        self.curr_timestamp = start_ts
        self.start_ts = start_ts
        self.results: List = list()
        self.all_times = 0
        self.mIdBoundingBoxMap: List[Dict[int, BlobRect]] = [dict(), dict()]
        self.mPointBlobAssociation: List[PointsBlob] = [PointsBlob(), PointsBlob()]
        self.mBlobLabelToObject: List[Dict[int, IObject]] = [dict(), dict()]
        self.mObjectToBlobLabel: List[DefaultDict[IObject, Set[int]]] = [
            defaultdict(set), defaultdict(set)]
        self.mReplacementObject: Dict[IObject, IObject] = dict()
        self.mLastFrameIdx = 1
        self.mCurrentFrameIdx = 0
        self.mObjectList: List[IObject] = []

    def updateBlobFrameObjectMapping(self, obj: IObject, blobLabel: int):
        self.mBlobLabelToObject[self.mCurrentFrameIdx][blobLabel] = obj
        self.mObjectToBlobLabel[self.mCurrentFrameIdx][obj].add(blobLabel)

    def updateObjectReference(self, oldObject: IObject, newObject: IObject):
        for frameIdx in [0, 1]:
            for blobLabel in self.mBlobLabelToObject[frameIdx].keys():
                if self.mBlobLabelToObject[frameIdx][blobLabel] == oldObject:
                    self.mBlobLabelToObject[frameIdx][blobLabel] = newObject
            if oldObject in self.mObjectToBlobLabel[frameIdx]:
                labels: Set[int] = deepcopy(self.mObjectToBlobLabel[frameIdx][oldObject])
                del self.mObjectToBlobLabel[frameIdx][oldObject]
                self.mObjectToBlobLabel[frameIdx][newObject] = self.mObjectToBlobLabel[frameIdx][newObject].union(labels)

    def getObjectsWithState(self, objectList, state: ObjectState) -> List[IObject]:
        relevant_objs = []
        for obj in objectList:
            if obj.mState == state:
                relevant_objs.append(obj)
        return relevant_objs

    def getObjectsNotWithState(self, objectList, states: List[ObjectState]) -> List[IObject]:
        relevant_objs = []
        for obj in objectList:
            if obj.mState not in states:
                relevant_objs.append(obj)
        return relevant_objs

    def updateModelWithBlob(self, obj: IObject, newBlobId: int, ts):
        bbox = self.mIdBoundingBoxMap[self.mCurrentFrameIdx][newBlobId].mBoundingBox
        obj.getIObjectModel().addMergeBlob(ts, Blob(bbox, obj.mState))
        kp = self.mPointBlobAssociation[self.mCurrentFrameIdx].getKpDesc(newBlobId)["kp"]
        des = self.mPointBlobAssociation[self.mCurrentFrameIdx].getKpDesc(newBlobId)["des"]
        obj.getIObjectModel().addAndMatchPoint(ts, kp, des)
        self.updateBlobFrameObjectMapping(obj, newBlobId)

    def mergeObjectWithSimilarStart(self, enteringList: List[IObject], maxTimeDistance: int, newBlobLabel: int):
        mergedList: List[IObject] = []
        if len(enteringList) == 1:
            mergedList.append(enteringList[0])
        elif len(enteringList) > 1:
            for enteringObj in enteringList:
                merged: bool = False
                for mergedObj in mergedList:
                    if not merged and abs(enteringObj.getIObjectModel().getFirstTimestamp() - mergedObj.getIObjectModel().getFirstTimestamp()) < maxTimeDistance:
                        mergedObj.getIObjectModel().moveObjectModel(enteringObj.getIObjectModel())
                        self.updateObjectReference(enteringObj, mergedObj)
                        enteringObj.setState(ObjectState.DELETED)

                        merged = True
                if not merged:
                    self.updateBlobFrameObjectMapping(
                        enteringObj, newBlobLabel)
                    mergedList.append(enteringObj)
        return mergedList

    def handleNewObjects(self, association: FrameAssociation):
        newBlobs: Set[int] = association.getNewBlob()

        for newBlobId in newBlobs:
            obj = Object()
            self.updateBlobFrameObjectMapping(obj, newBlobId)
            obj.mState = ObjectState.HYPOTHESIS
            self.updateModelWithBlob(obj, newBlobId, self.curr_timestamp)
            self.mObjectList.append(obj)
            log.debug(f"New Blob: blobid={newBlobId}, state={obj.mState}")

    def handleOneToOne(self, association):
        blobsMatches: Set[Tuple[int, int]] = association.getDirectMatchBlob()
        for oldBlobId, newBlobId in blobsMatches:
            obj = self.mBlobLabelToObject[self.mLastFrameIdx].get(oldBlobId)
            if obj is None:
                log.warning("old obj is None... shouldn't happen. creating new..")
                obj = Object()
                self.updateBlobFrameObjectMapping(obj, newBlobId)
                obj.mState = ObjectState.HYPOTHESIS
                self.updateModelWithBlob(obj, newBlobId, self.curr_timestamp)
                self.mObjectList.append(obj)
                log.debug(f"New Blob: blobid={newBlobId}, state={obj.mState}")
            else:
                self.updateModelWithBlob(obj, newBlobId, self.curr_timestamp)

    def handleMerge(self, association: FrameAssociation):
        pairs: Set[Tuple[Tuple[int], int]] = association.getBlobMerge()

        for oldBlobIds, newBlobLabel in pairs:
            # objectList contains list of oldObjs
            objectList: List[IObject] = []
            for oldBlobId in oldBlobIds:
                obj = self.mBlobLabelToObject[self.mLastFrameIdx].get(oldBlobId)
                if obj is None:
                    log.warning("old obj is none... skipping")
                else:
                    objectList.append(obj)

            if len(objectList) == 0:
                log.error("object list is empty")
                continue

            enteringList = self.getObjectsWithState(objectList, ObjectState.ENTERING)
            hypothesisList = self.getObjectsWithState(objectList, ObjectState.HYPOTHESIS)
            leavingList = self.getObjectsWithState(objectList, ObjectState.LEAVING)
            otherObjectList = self.getObjectsNotWithState(objectList, [ObjectState.ENTERING, ObjectState.HYPOTHESIS, ObjectState.LEAVING, ObjectState.LOST])

            log.debug(f"enteringList: {enteringList}")
            log.debug(f"hypothesisList: {hypothesisList}")
            log.debug(f"leavingList: {leavingList}")
            log.debug(f"otherObjectList: {otherObjectList}")

            for obj in leavingList:
                newObjects = obj.getIObjectModel().handleLeaving()
                if len(newObjects) == 0:
                    obj.setState(ObjectState.OBJECT)
                    otherObjectList.append(obj)
                else:
                    obj.setState(ObjectState.SAVEANDELETE)
                    if len(newObjects) == 1:
                        objectList.remove(obj)
                        newObjects[0].setState(ObjectState.OBJECT)
                        self.updateObjectReference(obj, newObjects[0])
                        self.mObjectList.append(newObjects[0])
                        otherObjectList.append(newObjects[0])
                        objectList.append(newObjects[0])
                    else:
                        log.warning(
                            "handlLeaving does not support multiple object output")

            if len(enteringList) > 0:
                mergedObjects = self.mergeObjectWithSimilarStart(
                    enteringList, 10, newBlobLabel)
                for mergedObj in mergedObjects:
                    if mergedObj.getIObjectModel().getLastTimestamp() - mergedObj.getIObjectModel().getFirstTimestamp() + 1 >= self.maxFrameHypothesis:
                        otherObjectList.append(mergedObj)
                    else:
                        hypothesisList.append(mergedObj)
                if len(mergedObjects) > 0:
                    resultObject = mergedObjects[0]

            if len(otherObjectList) > 1:
                log.debug("Creating Group")
                groupList: List[IObject] = self.getObjectsWithState(
                    objectList, ObjectState.OBJECTGROUP)
                remainingObjectList: List[IObject] = self.getObjectsNotWithState(
                    objectList, [ObjectState.OBJECTGROUP])
                if len(groupList) == 0:
                    objectGroup = ObjectGroup()
                    self.mObjectList.append(objectGroup)
                else:
                    objectGroup = groupList[0]
                    assert type(objectGroup) == ObjectGroup
                    for g in groupList[1:]:
                        remainingObjectList.append(g)
                for obj in remainingObjectList:
                    objectGroup.addObject(obj)
                    try:
                        self.mObjectList.remove(obj)
                    except:
                        continue
                    self.updateObjectReference(obj, objectGroup)
                resultObject = objectGroup
            else:
                if len(hypothesisList) == 0:
                    continue
                mainObject: IObject
                if len(otherObjectList) > 0:
                    mainObject = otherObjectList[0]
                    temp = hypothesisList
                else:
                    mainObject = hypothesisList[0]
                    temp = hypothesisList[1:]
                self.updateBlobFrameObjectMapping(mainObject, newBlobLabel)
                for obj in temp:
                    mainObject.getIObjectModel().moveObjectModel(obj.getIObjectModel())
                    self.updateObjectReference(obj, mainObject)
                    obj.setState(ObjectState.DELETED)
                resultObject = mainObject
            self.updateModelWithBlob(
                resultObject, newBlobLabel, self.curr_timestamp)
            for val in oldBlobIds:
                if val in self.mBlobLabelToObject[self.mLastFrameIdx]:
                    self.updateBlobFrameObjectMapping(
                        self.mBlobLabelToObject[self.mLastFrameIdx][val], newBlobLabel)

    def handleSplits(self, association):
        splits: Set[Tuple[int, Tuple[int]]] = association.getBlobSplit()

        for oldBlobLabel, newBlobLabels in splits:
            oldObj = self.mBlobLabelToObject[self.mLastFrameIdx].get(oldBlobLabel)
            if oldObj is not None:
                if oldObj.getState() == ObjectState.OBJECTGROUP:

                    assert type(oldObj) == ObjectGroup, "Object is of type: %s" % type(oldObj)

                    omList: List[ObjectModel] = list()
                    modelToBlobLabel: Dict[ObjectModel, int] = dict()

                    for label in newBlobLabels:
                        om: ObjectModel = ObjectModel(None)
                        kps = self.mPointBlobAssociation[self.mCurrentFrameIdx].getKpDesc(label)["kp"]
                        des = self.mPointBlobAssociation[self.mCurrentFrameIdx].getKpDesc(label)["des"]
                        newBlobBoundingBox: List = self.mIdBoundingBoxMap[self.mCurrentFrameIdx][label].mBoundingBox
                        om.addPoint(self.curr_timestamp, kps, des)
                        om.addMergeBlob(self.curr_timestamp, Blob(newBlobBoundingBox, ObjectState.HYPOTHESIS))
                        omList.append(om)
                        modelToBlobLabel[om] = label

                    log.debug(f"omList: {omList}")
                    log.debug(f"modelToBlobLabel: {modelToBlobLabel}")

                    outputAssociation: List[Tuple[IObject, ObjectModel]] = oldObj.getObjectModelGroup().handleSplit(omList, self.mObjectList, self.curr_timestamp)
                    for assoc in outputAssociation:
                        self.updateBlobFrameObjectMapping(assoc[0], modelToBlobLabel[assoc[1]])
                    oldObj.setState(ObjectState.DELETED)

                elif oldObj.getState() == ObjectState.DELETED:
                    log.warning("Deleted object are seen in split.")
                else:
                    assert type(oldObj) == Object, "Object is of type: %s" % type(oldObj)
                    blobDivision: List[List[Tuple[int, BlobRect]]] = list()
                    for newBobLabel in newBlobLabels:
                        overlappedPair: List[Tuple[int, BlobRect]] = list()
                        existingBlobRect = self.mIdBoundingBoxMap[self.mCurrentFrameIdx].get(newBobLabel)
                        if existingBlobRect is not None:
                            bbox = existingBlobRect.mBoundingBox
                            maxSegDist = 0.4
                            dilatedbbox = dilate(bbox, maxSegDist)
                            if getArea(dilatedbbox) > 0:
                                resizedBlob = cv2.resize(
                                    existingBlobRect.mBlob, getBoxDims(dilatedbbox), 0, 0)
                                overlappedPair.append(
                                    tuple([newBobLabel, BlobRect(dilatedbbox, resizedBlob)]))
                                blobDivision.append(overlappedPair)
                            else:
                                log.error("Blob area should > 0")
                        else:
                            log.error("blob label is not present in frame")

                    log.debug(f"BlobDivision: {blobDivision}")

                    originalBlobDivisionSz = len(blobDivision)
                    blobDivisionSz = 0
                    while len(blobDivision) != blobDivisionSz and len(blobDivision) > 1:
                        blobDivisionSz = len(blobDivision)
                        i = 0
                        timeToRestart = False
                        while i < blobDivisionSz-1 and not timeToRestart:
                            j = i + 1
                            while j < blobDivisionSz and not timeToRestart:
                                if blobListOverlap(blobDivision[i], blobDivision[j]):
                                    blobDivision[i].extend(blobDivision[j])
                                    del blobDivision[j]
                                    timeToRestart = True
                                j += 1
                            i += 1

                    log.debug(f"Length of blob division is now: {len(blobDivision)}")

                    if len(blobDivision) == 0:
                        log.error(
                            "no blob division... original division size was %d" % originalBlobDivisionSz)
                    elif len(blobDivision) == 1:
                        blobsToMerge = blobDivision[0]
                        for newBlobId, blobRect in blobsToMerge:
                            self.updateModelWithBlob(
                                oldObj, newBlobId, self.curr_timestamp)
                    else:
                        log.debug("This is a real split from an object")
                        modelList: List[ObjectModel] = list()
                        indexToLabelList: DefaultDict[ObjectModel, List[int]] = defaultdict(list)
                        for splits in blobDivision:
                            om = ObjectModel(None)
                            for subsplit in splits:
                                label = subsplit[0]
                                kps = self.mPointBlobAssociation[self.mCurrentFrameIdx].getKpDesc(label)["kp"]
                                des = self.mPointBlobAssociation[self.mCurrentFrameIdx].getKpDesc(label)["des"]
                                newBlobBoundingBox = self.mIdBoundingBoxMap[self.mCurrentFrameIdx][label].mBoundingBox
                                om.addPoint(self.curr_timestamp, kps, des)
                                om.addMergeBlob(self.curr_timestamp, Blob(newBlobBoundingBox, ObjectState.HYPOTHESIS))
                                indexToLabelList[om].append(label)
                            modelList.append(om)
                            
                        keptModel, splittedModel = oldObj.split(modelList)

                        labelToUpdate: List[int] = indexToLabelList[keptModel]
                        log.debug(f"labelToUpdate: {labelToUpdate}")
                        for lb in labelToUpdate:
                            self.updateBlobFrameObjectMapping(oldObj, lb)

                        for objModel in splittedModel:
                            log.debug("In splitted model")
                            newObj: Object = Object()
                            labelToUpdate: List[int] = indexToLabelList[objModel]
                            for newBlobLabel in labelToUpdate:
                                self.updateBlobFrameObjectMapping(newObj, newBlobLabel)
                                newObj.setState(ObjectState.OBJECT)
                                newObj.getObjectModel().moveObjectModel(objModel)
                                self.updateModelWithBlob(newObj, newBlobLabel, self.curr_timestamp)
                            log.debug(f"Adding: {newObj}")
                            self.mObjectList.append(newObj)

            else:
                continue

    def updateModel(self, association):
        self.handleNewObjects(association)
        self.handleOneToOne(association)
        self.handleMerge(association)
        self.handleSplits(association)

    def saveGoodObjects(self, obj):
        if type(obj) == ObjectGroup:
            obj.getObjectModelGroup().addUnmatchedGroupBlobToExistingObjects()
            for o in obj.mObjectList:
                self.saveGoodObjects(o)
            return
        maxGap = 3
        obj.mModel.interpolateMissingFrame(maxGap)
        obj.mModel.correctGroupObservation()
        objId = obj.getObjectId()
        om = obj.getObjectModel()
        blobList: List[Tuple(int, Blob)] = sorted(om.getBlobs().items())
        currObjRows = list()
        for ts, blob in blobList:
            bbox = list(map(int, blob.getBoundingBox()))
            currObjRows.append([objId, ts, bbox[0]*2, bbox[1]*2, bbox[2]*2, bbox[3]*2, blob.mState]) # to scale back up
        self.results.extend(currObjRows)

    # go through all objects and update state
    def updateState(self):
        log.debug("updating state...")
        # objects to delete; set True if want to try to save; False if don't save
        objectsToDelete: Dict[IObject, bool] = dict()
        for obj in self.mObjectList:
            obj.mModel.simplifyModel(self.curr_timestamp)
            if obj.mState == ObjectState.HYPOTHESIS:
                # 1) The object has an association with the current frame:
                if obj.mModel.getLastTimestamp() == self.curr_timestamp:
                    # The object has been there for more than N frame
                    if obj.mModel.getLastTimestamp() - obj.mModel.getFirstTimestamp() > self.maxFrameHypothesis:
                        bb = sorted(obj.mModel.getBlobs().items())[
                            0][1].mObjectBoundingBox
                    
                        # We check if it is a previously lost item
                        lostObjs: List[IObject] = self.getObjectsWithState(
                            self.mObjectList, ObjectState.LOST)
                        matchingObjs = None
                        lastStableBoundingBox = []
                        currentNbMatches = 0

                        for lostObj in lostObjs:
                            assert type(lostObj) == Object
                            tempBB = lostObj.mModel.getLastStableBoundingBox()
                            dilatedLostObjBB = dilate(tempBB, 1.5)

                            if getRectangleIntersectionArea(dilatedLostObjBB, bb) > 0:
                                pointsDelta: List = ObjectModel.getMatchingPointMovement(
                                    obj.getIObjectModel(), lostObj.getIObjectModel())
                                if len(pointsDelta) >= 3 and len(pointsDelta) > currentNbMatches:
                                    pointsDelta = sorted(pointsDelta)
                                    medianDisplacementTmp = pointsDelta[int(
                                        len(pointsDelta)/float(2))]
                                    if medianDisplacementTmp > 3:
                                        currentNbMatches = len(pointsDelta)
                                        matchingObjs = lostObj
                                        lastStableBoundingBox = tempBB
                        if matchingObjs is not None and currentNbMatches > 3:
                            # 1 We check that the points have indeed moved. If they have moved, then we do an interpolation and we attach
                            lastTimestampMovement = matchingObjs.getIObjectModel().getLastTimestamp()
                            firstTimestampMovement = obj.getIObjectModel().getFirstTimestamp()
                            for j in range(lastTimestampMovement+1, firstTimestampMovement):
                                matchingObjs.getIObjectModel().addMergeBlob(j, Blob(lastStableBoundingBox, ObjectState.STOPPED))
                            log.debug(f"Added {firstTimestampMovement - lastTimestampMovement + 1} static entry")
                            matchingObjs.setState(ObjectState.OBJECT)
                            matchingObjs.getIObjectModel().moveObjectModel(obj.getIObjectModel())
                            self.updateObjectReference(obj, matchingObjs)
                            objectsToDelete[obj] = False
                            log.debug("merge %s and %s with %d matches" % (
                                matchingObjs.getObjectId(), obj.getObjectId(), currentNbMatches))
                        else:
                            obj.setState(ObjectState.OBJECT)
                else:
                    objectsToDelete[obj] = False

            elif obj.mState == ObjectState.OBJECT:

                if obj.getIObjectModel().getLastTimestamp() == self.curr_timestamp:
                    pass
                else:
                    if type(obj) == Object:
                        obj.setState(ObjectState.LOST)
                    elif type(obj) == ObjectGroup:
                        objectsToDelete[obj] = False
                    else:
                        log.debug("uh maybe revisit this...")

            elif obj.mState == ObjectState.LOST:

                if obj.getIObjectModel().getLastTimestamp() == self.curr_timestamp:
                    obj.setState(ObjectState.OBJECT)
                else:
                    deltaFrame = self.curr_timestamp - obj.getIObjectModel().getLastTimestamp()
                    if deltaFrame > self.maximumLostFrame:
                        objectsToDelete[obj] = True
                        obj.setState(ObjectState.DELETED) # once add to objects to delete, don't try to pair up with hypothesis

            elif obj.mState == ObjectState.OBJECTGROUP:

                assert type(obj) == ObjectGroup
                og: ObjectGroup = obj

                objectList: List[Object] = og.getObjects()
                if obj.getIObjectModel().getLastTimestamp() == self.curr_timestamp:
                    if len(objectList) == 0:
                        log.debug("objectlist.size() == 0")
                        objectsToDelete[og] = False
                    elif len(objectList) == 1:
                        log.debug("objectlist.size() == 1")
                        o: Object = objectList[0]
                        self.mObjectList.append(o)
                        self.updateObjectReference(og, o)
                        og.clearObjectList()
                        objectsToDelete[obj] = False
                else:
                    log.debug("Group leaving %s" % og.getObjectId())
                    log.debug("Change from original... setting obj as lost")
                    og.getObjectModelGroup().addUnmatchedGroupBlobToExistingObjects()
                    for o in objectList:
                        o.setState(ObjectState.LOST)
                        self.mObjectList.append(o)
                    og.clearObjectList()
                    objectsToDelete[og] = False

            elif obj.mState == ObjectState.INGROUP:
                assert False, "Object should never be in this state in there. They should be remove from the list"

            elif obj.mState == ObjectState.LEAVING:

                if obj.getIObjectModel().getLastTimestamp() == self.curr_timestamp:
                    isEntering = False
                    blobs = obj.getIObjectModel().getBlobs()
                    if len(blobs) > 1:
                        b: Blob = sorted(blobs.items())[-1][1]
                        isEntering = not BlobRect.onEdgeOfFrame(
                            b.getBoundingBox())
                    if self.verifyReEnteringObject and isEntering:
                        newObjects: List[IObject] = obj.getIObjectModel(
                        ).handleLeaving()
                        if len(newObjects) == 0:
                            obj.setState(ObjectState.OBJECT)
                        else:
                            objectsToDelete[obj] = True
                            if len(newObjects) == 1:
                                self.updateObjectReference(obj, newObjects[0])
                                self.mObjectList.append(newObjects[0])
                            else:
                                log.warning(
                                    "handleleaving does not support multiple object output")
                    elif isEntering:
                        obj.setState(ObjectState.OBJECT)
                else:
                    objectsToDelete[obj] = True

            elif obj.mState == ObjectState.DELETED:
                objectsToDelete[obj] = False

            elif obj.mState == ObjectState.SAVEANDELETE:
                objectsToDelete[obj] = True

            elif obj.mState == ObjectState.UNDEFINED:
                log.error(f"undefined state... obejct was not initialized properly.. ts={self.curr_timestamp}")

            else:
                log.error("f{obj.mState} UHOH undefined state")

        objectsToKeep: List[IObject] = list()

        for obj in self.mObjectList:
            if obj in objectsToDelete:
                if objectsToDelete[obj] is True:
                    log.debug(
                        f"==============>========>==>End of Life. Saving + Deleting Obj {obj.mObjectId}")
                    self.saveGoodObjects(obj)
                else:
                    log.debug(
                        f"============>========>====>Useless. Deleting Obj {obj.mObjectId}.")
                if obj in self.mObjectToBlobLabel[self.mCurrentFrameIdx]:
                    del self.mObjectToBlobLabel[self.mCurrentFrameIdx][obj]
                self.mBlobLabelToObject[self.mCurrentFrameIdx] = {
                    k: v for k, v in self.mBlobLabelToObject[self.mCurrentFrameIdx].items() if v != obj}
                del obj
            else:
                objectsToKeep.append(obj)
        self.mObjectList = objectsToKeep

    def process_frame(self, foreground, orig_im, save_ts):
        self.all_times = 0

        log.debug(f"== Analyzing Frame {self.curr_timestamp} ==")

        self.save_ts = save_ts

        # switch between 0 and 1 buffers
        self.mCurrentFrameIdx = not self.mCurrentFrameIdx
        self.mLastFrameIdx = not self.mLastFrameIdx

        self.mBlobLabelToObject[self.mCurrentFrameIdx] = dict()
        self.mPointBlobAssociation[self.mCurrentFrameIdx] = None
        self.mIdBoundingBoxMap[self.mCurrentFrameIdx] = dict()
        self.mObjectToBlobLabel[self.mCurrentFrameIdx] = defaultdict(set)
        self.mReplacementObject = dict()

        det_time = time.time()
        # run cca to get blobs
        self.blob_detector = BlobDetector(morph_open_kernel=self.morph_open_kernel, morph_close_kernel=self.morph_close_kernel)
        self.blob_detector.update(foreground, orig_im[::])

        kps_loc_fname = self.kps_loc_template.format(frame_no=self.save_ts)
        kps_matches_fname = self.kps_matches_template.format(frame_no=self.save_ts)

        # blob에서 sift써서 keypoint들 추출, keypoint는 blob의 feature
        pointsBlob = FeatureDetectorExtractorMatcher.detect(orig_im.copy(), self.blob_detector.getBGSMask()) # if not IGNORE_MASK else None)

        pointsBlobToDump = np.array(pointsBlob.kp).astype(np.int16) * 2 # scale back up for query-time usage
        # blob의 keypoint들 저장
        with open(kps_loc_fname, 'wb') as f:
            pickle.dump(pointsBlobToDump, f)

        self.mPointBlobAssociation[self.mCurrentFrameIdx] = pointsBlob


        # for each kp, figure out which blob it corresponds to..
        pointsBlob.calculatePointBlobAssociation(
            self.blob_detector.getLabelMask())

        self.blob_detector.filterOutBlobsWithNoKPs(pointsBlob)
        self.mIdBoundingBoxMap[self.mCurrentFrameIdx] = self.blob_detector.getBlobBoundingBoxMap()


        # compute matches between kps across frame i-1 and frame i
        matches = FeatureDetectorExtractorMatcher.match(
            self.mPointBlobAssociation[self.mLastFrameIdx].des, self.mPointBlobAssociation[self.mCurrentFrameIdx].des)

        matches_indices = [(m.queryIdx, m.trainIdx) for m in matches]
        matchesToDump = np.array(list(zip(*matches_indices)))
        # 두 frame간의 keypoint 매칭 정보 저장.
        with open(kps_matches_fname, 'wb') as f:
            pickle.dump(matchesToDump, f)
        self.all_times += time.time()-det_time

        # now using matches between kps and associations of kps to blobs, figure out associations between blobs of consec frames
        # n-1, 1-n, 1-1, 1-0, 0-1
        counts = Counter()
        for m in matches:
            p0 = self.mPointBlobAssociation[self.mLastFrameIdx].mBlobId[m.queryIdx]
            p1 = self.mPointBlobAssociation[self.mCurrentFrameIdx].mBlobId[m.trainIdx]
            pair = tuple([p0, p1])
            counts[pair] += 1
        association = FrameAssociation(
            self.mIdBoundingBoxMap[self.mLastFrameIdx], self.mIdBoundingBoxMap[self.mCurrentFrameIdx], counts, self.min_num_kp_matches)

        self.updateModel(association)

        self.updateState()

        objectToLabel: Dict[IObject, int] = dict()
        for label, obj in list(self.mBlobLabelToObject[self.mCurrentFrameIdx].items()):
            if obj in objectToLabel:
                newLabelForObject = objectToLabel[obj]
                self.mPointBlobAssociation[self.mCurrentFrameIdx].updateBlobId(
                    label, newLabelForObject)
                self.mIdBoundingBoxMap[self.mCurrentFrameIdx][newLabelForObject] = BlobRect.mergeBlobRect(
                    self.mIdBoundingBoxMap[self.mCurrentFrameIdx][label], self.mIdBoundingBoxMap[self.mCurrentFrameIdx][newLabelForObject])
                del self.mIdBoundingBoxMap[self.mCurrentFrameIdx][label]
                if newLabelForObject != label:
                    existingObj = self.mBlobLabelToObject[self.mCurrentFrameIdx].get(
                        label)
                    if existingObj is not None:
                        if existingObj in self.mObjectToBlobLabel[self.mCurrentFrameIdx]:
                            lbls = self.mObjectToBlobLabel[self.mCurrentFrameIdx].get(existingObj)
                            self.mObjectToBlobLabel[self.mCurrentFrameIdx][existingObj].remove(label)
                            if len(self.mObjectToBlobLabel[self.mCurrentFrameIdx][existingObj]) == 0:
                                del self.mObjectToBlobLabel[self.mCurrentFrameIdx][existingObj]
            else:
                objectToLabel[obj] = label

        groupList: List[IObject] = self.getObjectsWithState(
            self.mObjectList, ObjectState.OBJECTGROUP)
        for group in groupList:
            assert type(group) == ObjectGroup
            objectList: List[Object] = group.getObjects()

            for obj in objectList:
                om: ObjectModel = obj.getObjectModel()
                om.updateInGroupBlobs()

        self.curr_timestamp += 1

    def save_results(self, fname=None):
        import pandas as pd
        # add any object tracks stil at the end!!
        for obj in self.mObjectList:
            self.saveGoodObjects(obj)
        df = pd.DataFrame(self.results)

        if df.empty:
            log.info("No data to save... creating empty df")
            Path(fname).touch()
        else:
            df.to_csv(fname, index=False, header=[
                "ObjId", "TS", "x1", "y1", "x2", "y2", "bstate"])

    def get_results(self):
        import pandas as pd
        return pd.DataFrame(self.results, columns=["ObjId", "TS", "x1", "y1", "x2", "y2", "bstate"])
