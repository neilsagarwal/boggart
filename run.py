# %%
from ClusteringPipelineEngine import ClusteringPipelineEngine
from Experiment import Experiment
from VideoData import VideoData

# vid = "lausanne_pont_bassieres"
vid = "auburn_first_angle"
hours = list(range(10,11))

### AHEAD-OF-TIME PROCESSING ###
# This is done once per video
for hr in hours:
    VideoData(vid, hr).check_vids()
    Experiment(vid, hr).run_ingest()


# QUERY-TIME PROCESSING ###
# This is done once per query

query_class = "car"
model = "yolov5"
acc_target = 0.9
query_conf = 0.7
qtype = "bbox"
pc = 0.1

# # convert query_class name to the corresponding index
qclass_label = {"car" : 2, "person": 0}[query_class] # coco

cpe = ClusteringPipelineEngine(vid, query_conf=query_conf)
results_df = cpe.execute(hours, qtype, model, qclass_label, acc_target, percent_clusters=pc, ioda=0.1, get_boggart_results=True)
print(results_df)
# Results of boggart are located in results_df

# If bounding box query,
#   columns: hour, frame_no, x1, y1, x2, y2
# If count query,
#   columns: hour, frame_no, count
# If binary query,
#   columns: hour, frame_no, found