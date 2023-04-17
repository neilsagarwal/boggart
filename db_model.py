from mongoengine import Document, StringField, DynamicDocument, IntField, DictField, ListField


class DetectionResult(Document):
    model = StringField()
    num_detections = IntField()
    detection_scores = ListField()
    detection_classes = ListField()
    detection_boxes = ListField()


class Frame(DynamicDocument):
    frame_no = IntField(required=True)
    hour = IntField(required=True)

    meta = {
        'indexes': [{
            'fields': ('hour', 'frame_no'),
            'unique': True

        }]
    }

    inferenceResults = DictField()
