from ut_util_classes import ObjectState, log
from typing import List, Dict
from ut_IObjectModel import ObjectModel, ObjectModelGroup, IObjectModel

class IObject():

    id_counter = 0

    def __init__(self):
        self.mObjectId = str(self.id_counter)
        IObject.id_counter += 1
        self.mState: ObjectState = ObjectState.UNDEFINED
        
    def getState(self) -> ObjectState:
        return self.mState

    def setState(self, state: ObjectState) -> None:
        self.mState = state

    def setObjectId(self, objId: str):
        self.mObjectId = objId

    def getObjectId(self):
        return str(self.mObjectId)

    def __str__(self):
        return f"Object[ID={self.mObjectId},STATE={self.getState()}, ID={id(self)}]"

    def __repr__(self):
        return f"Object[ID={self.mObjectId},STATE={self.getState()}, ID={id(self)}]"


class Object(IObject):

    def __init__(self):
        super().__init__()
        self.mModel = ObjectModel(self)

    def setState(self, state: ObjectState):
        self.mState = state

    def getIObjectModel(self):
        return self.mModel

    def getObjectModel(self):
        return self.mModel

    def split(self, modelList: List[ObjectModel]):
        outSplittedModel: List[ObjectModel] = list()
        maxPoints = 0
        index = 0
        for i, om in enumerate(modelList):
            nbPoints = ObjectModel.getMatchingPointNumber(self.mModel, om)
            log.debug(f"{om.getLastBoundingBox()}")
            log.debug(f"nbpoints is {nbPoints}")
            if nbPoints > maxPoints:
                maxPoints = nbPoints
                index = i

        # find which new blob is most similar to old object
        keep: ObjectModel = modelList[index]

        for i in range(len(modelList)):
            if i != index:
                self.mModel.extractObjectModel(modelList[i]) # checked
                modelList[i].updateInGroupBlobs()
                outSplittedModel.append(modelList[i])
                log.debug(f"{modelList[i].getBlobs().keys()}")

        self.mModel.moveObjectModel(keep)
        self.mModel.updateInGroupBlobs()

        return keep, outSplittedModel


class ObjectGroup(IObject):

    def __init__(self):
        super().__init__()
        self.mModel: ObjectModelGroup = ObjectModelGroup(self)
        self.mObjectList: List[Object] = list()
        self.mObjectToObjectModel: Dict[Object, IObjectModel] = dict()
        self.mObjectModelToObject: Dict[IObjectModel, Object] = dict()
        self.setState(ObjectState.OBJECTGROUP)

    def getObjectModelGroup(self) -> ObjectModelGroup:
        return self.mModel

    def getIObjectModel(self):
        return self.mModel

    def getObjects(self) -> List[Object]:
        return self.mObjectList

    def clearObjectList(self):
        self.mModel.clearObjectModel()
        self.mObjectToObjectModel.clear()
        self.mObjectModelToObject.clear()
        self.mObjectList.clear()

    def removeObject(self, o: Object):
        if o in self.mObjectList:
            self.mObjectList.remove(o)
        del self.mObjectToObjectModel[o]
        del self.mObjectModelToObject[o.getIObjectModel()]

    def addObject(self, o: IObject):
        if self == o:
            return
        if type(o) == ObjectGroup:
            objectList: List[Object] = o.getObjects()
            for oh in objectList:
                self.addObject(oh)
            objectList.clear()
        elif type(o) == Object:
            self.mObjectList.append(o)
            o.setState(ObjectState.INGROUP)
            self.mModel.addObjectModel(o.getObjectModel())
            self.mObjectToObjectModel[o] = o.getObjectModel()
            self.mObjectModelToObject[o.getObjectModel()] = o
            name = [str(m.getObjectId()) for m in self.mObjectList]
            name = ", ".join(name)
            super().setObjectId(name)
        else:
            assert False
