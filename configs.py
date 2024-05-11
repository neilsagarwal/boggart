import os

# location of boggart repository
BOGGART_REPO_PATH = "/home/kth/rva/boggart"
main_dir = f"{BOGGART_REPO_PATH}/data/"

assert os.path.exists(BOGGART_REPO_PATH), "Update Boggart Repository Path in configs.py"
assert main_dir is not None, "Set main_dir to <REPO PATH>/boggart/data/"

video_directory = f"{main_dir}{{vid_label}}{{hour}}/"

frames_dir = f"{{video_dir}}frames/"

trajectories_dir = f"{{video_dir}}trajectories/"
background_dir = f"{{video_dir}}backgrounds/"

kps_loc_dir = f"{{video_dir}}kps_locs/{{traj_info}}/"
kps_matches_dir = f"{{video_dir}}kps_matches/{{traj_info}}/"
kps_raw_dir = f"{{video_dir}}raw_kps/"

obj_dets_dir = f"{{video_dir}}object_detection_results/{{model}}/"
obj_dets_csv = f"{{obj_dets_dir}}{{name}}{{hour}}.csv"

query_results_dir = f"{{video_dir}}query_results/{{query_type}}/"
boggart_results_dir = f"{{video_dir}}boggart_results/{{query_type}}/"

pipeline_results_dir = f"{main_dir}pipeline_results/"

video_files_dir = f"{{video_dir}}video/"

crops = {
    # this is what you DON'T WANT!
    # vname : {class_id : [x1, y1, x2, y2]} where x1, y1 top left
    "auburn_first_angle" : {
        "person" : [0, 0, 1920, 400],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },

    "jackson_hole_wy" : {
        "person" : [0, 0, 1920, 500],
        "car" : [0, 0, 1920, 500],
        "bicycle" : [0, 0, 1920, 500],
        "truck" : [0, 0, 1920, 500]
    },
}

frame_bounds = {
    "auburn_first_angle": [1080, 1920],

    "jackson_hole_wy": [1080, 1920],
}

class BackgroundConfig:
    def __init__(self, peak_thresh):
        self.peak_thresh : int = peak_thresh
        self.sample_rate : int = 30
        self.box_length : int = 2
        self.quant : int = 16
        self.bg_dur : int = 1800

    def get_bg_dir(self, video_dir):
        return background_dir.format(video_dir=video_dir)

    def get_bg_start(self, chunk_start):
        return chunk_start//self.bg_dur * self.bg_dur

    def get_base_bg_fname(self, video_dir, bg_start):
        return f"{self.get_bg_dir(video_dir)}bg_{bg_start}.pkl"

    def get_proper_bg_fname(self, video_dir, bg_start):
        return f"{self.get_bg_dir(video_dir)}bg_{bg_start}_{self.peak_thresh}.pkl"

class TrajectoryConfig:
    def __init__(self, diff_thresh, chunk_size, fps=30):
        self.fps : int = fps
        self.diff_thresh : int = diff_thresh
        self.blur_amt = 15
        self.chunk_size = chunk_size

        assert 0 < self.chunk_size <= 1800
        assert 0 < self.fps <= 30

        self.kps_loc_template = None
        self.kps_matches_template = None
        # self.kps_raw_template = None

    def get_traj_dir(self, video_dir):
        return trajectories_dir.format(video_dir=video_dir)

    def get_traj_fname(self, video_dir, chunk_start, bg_config):
        return f"{trajectories_dir.format(video_dir=video_dir)}{chunk_start}_{bg_config.peak_thresh}_{self.fps}_{self.diff_thresh}_{self.chunk_size}.csv"

    def get_kps_loc_template(self, video_dir, bg_peak_thresh):
        if self.kps_loc_template is None:
            local_dir = kps_loc_dir.format(video_dir=video_dir, traj_info=f"{bg_peak_thresh}_{self.fps}_{self.diff_thresh}_{self.chunk_size}")
            os.makedirs(local_dir, exist_ok=True)
            self.kps_loc_template = f"{local_dir}{{frame_no}}.pkl"
        return self.kps_loc_template

    def get_kps_matches_template(self, video_dir, bg_peak_thresh):
        if self.kps_matches_template is None:
            local_dir = kps_matches_dir.format(video_dir=video_dir, traj_info=f"{bg_peak_thresh}_{self.fps}_{self.diff_thresh}_{self.chunk_size}")
            os.makedirs(local_dir, exist_ok=True)
            self.kps_matches_template = f"{local_dir}{{frame_no}}.pkl"
        return self.kps_matches_template
