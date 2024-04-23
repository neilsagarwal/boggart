# Boggart

## Visualization

https://user-images.githubusercontent.com/16660771/232618499-968dd37e-d9e9-4601-80d4-4096a46365e3.mp4

## Environment/Repository Setup Instructions
### Python Environment
```
cd boggart
sudo apt install python3.7
python3.7 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install opencv-python tqdm tensorflow munkres 
pip install mongoengine
```

In `configs.py`, set `BOGGART_REPO_PATH` to the location of the Boggart repository.

### Hwang Frame Extraction Repository
```
git clone https://github.com/neilsagarwal/hwang
sudo apt-get install git cmake libgoogle-glog-dev libgflags-dev yasm libx264-dev build-essential wget unzip autoconf libtool
cd hwang
bash deps.sh
export LD_LIBRARY_PATH=<PATH_TO_BOGGART_REPO>/hwang/thirdparty/install/lib:$LD_LIBRARY_PATH
mkdir build
cd build
cmake ..
make -j64
cd ..
bash build.sh
export LD_LIBRARY_PATH=<PATH_TO_BOGGART_REPO>/hwang/python/python/hwang/lib:$LD_LIBRARY_PATH
 
cd ..
```

### For mAP Evaluation
```
git clone --depth 1 https://github.com/tensorflow/models
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .
cd ..
```

## Data Setup Instructions
### Video Data Setup

Videos (and data generated during execution) will be stored at `<BOGGART_REPO_PATH>/data/`. Make sure to set this path in `configs.py` (see the `main_dir` property).

Videos are expected to be split up by hour and then stored in ten-minute chunks. For example, the first ten minute chunk of hour 10 of the `auburn_first_angle` video dataset would be located at `<BOGGART_REPO_PATH>/data/auburn_first_angle/video/auburn_first_angle10_0.mp4`.

Example file structure for `data/`:
- boggart/
    - data/
        - auburn_first_angle10/
            - video/
                - auburn_first_angle10_0.mp4
                - auburn_first_angle10_1.mp4
                - auburn_first_angle10_2.mp4
                - auburn_first_angle10_3.mp4
                - auburn_first_angle10_4.mp4
                - auburn_first_angle10_5.mp4
        - auburn_first_angle11/
            - ...

### Model Inference Data Setup

Boggart's current implementation requires that model results are already generated and saved into MongoDB. The repository contains a helper script to load model inference results into MongoDB. `load_inference_results_into_mongodb.py` requires that inference results are stored in per-hour chunks. For example, the inference results for running YOLOv3 (trained on the COCO dataset) for hour 10 of the `auburn_first_angle` video dataset should be located at `<BOGGART_REPO_PATH>/inference_results/yolo3-coco/auburn_first_angle/auburn_first_angle10.csv` .


To set up MongoDB, run:
```
sudo apt install -y mongodb
```
Add 'directoryperdb=True' to `/etc/mongodb.conf`.
```
sudo mongod --config /etc/mongodb.conf
```

Then, update `ml_model`, `video_name` and `hour` in  `load_detections_into_mongodb.py`. Running this script will then load that hour's worth of inference results into the database.

## Run Boggart
Instructions to execute Boggart's ahead-of-time and query-time processing can be found in `run.py`.