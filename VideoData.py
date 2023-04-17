from configs import video_directory, frame_bounds, video_files_dir
import os, cv2
from utils import prepare_vid, parallelize_update_dictionary
from tqdm import trange

class NoMoreVideo(Exception):
    pass

class VideoData:

    source_video_ts = "TODO: <set cloud folder of videos in ts format>"
    source_video_mp4 = "TODO: <set cloud folder of videos in mp4 format>"
    stored_dur = 18000

    def __init__(self, db_vid, hour, vid_label=None):
        self.db_vid    : str = db_vid
        self.hour      : int = hour
        self.vid_label : str = vid_label if vid_label is not None else db_vid
        self.video_dir = video_directory.format(vid_label=self.vid_label, hour=self.hour)
        self.video_files_dir = video_files_dir.format(video_dir=self.video_dir)
        self.vname_ts = f"{self.video_files_dir}{self.vid_label}{hour}.ts"
        os.makedirs(self.video_files_dir, exist_ok=True)

        self.decoder = None

        self.nchunks = 6

    def vname_chunked(self, idx):
        return f"{self.video_files_dir}{self.vid_label}{self.hour}_{idx}.mp4"

    def get_frame_bounds(self):
        return frame_bounds[self.vid_label] # [height, width]

    def get_frames_by_bounds(self, start, stop, skip=1):

        start_chunk_idx = start//self.stored_dur
        stop_chunk_idx = (stop-1)//self.stored_dur

        # print(start, stop, start_chunk_idx, stop_chunk_idx)

        assert start_chunk_idx == stop_chunk_idx, "Not handling cross chunk frame extraction..."
        # dealing with only a single chunk
        if start_chunk_idx == stop_chunk_idx:
            start -= (start_chunk_idx * self.stored_dur)
            stop -= (start_chunk_idx * self.stored_dur)
            if self.decoder is None or self.decoder[0] != start_chunk_idx:
                import hwang
                if not os.path.exists(self.vname_chunked(start_chunk_idx)):
                    raise NoMoreVideo
                self.decoder = (start_chunk_idx, hwang.Decoder(self.vname_chunked(start_chunk_idx)))
            num_frames = self.decoder[1].video_index.frames()

            if start >= num_frames:
                raise NoMoreVideo

            if stop > num_frames:
                stop_at = num_frames
            else:
                stop_at = stop

            for frame in self.decoder[1].retrieve_generator(range(start, stop_at, skip)):
                yield frame

            for i in range(int((stop-num_frames)/skip)):
                yield None

    def _reencode_helper(self, chunk_idx):

        chunked_vname = self.vname_chunked(chunk_idx)
        if os.path.exists(chunked_vname) and cv2.VideoCapture(chunked_vname).get(cv2.CAP_PROP_FRAME_COUNT) > .90 * self.stored_dur:
            return

        import subprocess

        start = self.stored_dur * chunk_idx

        temp_vid_filename = f"temp_{self.db_vid}{self.hour}_{start}.mp4"

        vid = prepare_vid(self.vname_ts, start)

        out = None
        for i in trange(start, start+self.stored_dur):
            ret, im = vid.read()
            if not ret:
                assert i != start
                print(f"Breaking Rencoding of {chunked_vname} at {i}")
                break
            if out is None:
                h, w, _ = im.shape
                out = cv2.VideoWriter(temp_vid_filename, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w,h))
            out.write(im)
        out.release()

        # encode with h264 (add audio as hack to get hwang working...)
        cmd = f"ffmpeg -hide_banner -loglevel error -y -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 -i {temp_vid_filename} -c:v libx264 -c:a aac -shortest {chunked_vname}"
        print(cmd)
        subprocess.check_output(cmd, shell=True)

        if cv2.VideoCapture(chunked_vname).get(cv2.CAP_PROP_FRAME_COUNT) > .90 * self.stored_dur:
            print(f"{chunked_vname} is SHORT.")

        # delete temp vid
        os.remove(temp_vid_filename)

        # copy back chunked re-encoded mp4
        cmd = f"rclone -v copyto {chunked_vname} {self.source_video_mp4}/{self.vid_label}/{self.hour}/{self.vid_label}{self.hour}_{chunk_idx}.mp4"
        print(cmd)
        subprocess.check_output(cmd, shell=True)

        return

    def download_vids(self):

        # check that all ten min chunks exist
        all_good = all([os.path.exists(self.vname_chunked(i)) for i in range(self.nchunks)])
        if all_good:
            return

        # check if in gdrive and can be downloaded
        import subprocess
        cmd = f"rclone -v copy {self.source_video_mp4}/{self.vid_label}/{self.hour} {self.video_files_dir}"
        print(cmd)
        try:
            subprocess.check_output(cmd, shell=True)
            for i in range(self.nchunks):
                assert cv2.VideoCapture(self.vname_chunked(i)).get(cv2.CAP_PROP_FRAME_COUNT) > 17000
            return
        except (subprocess.CalledProcessError, AssertionError):
            print(f"Chunked mp4 videos not found. Downloading + converting .TS")

        # 1. download TS
        # 2. for each chunk: iterate through frame + save as mp4 + then reencode to h264 + send back to gdrive

        if not os.path.exists(self.vname_ts) or cv2.VideoCapture(self.vname_ts).get(cv2.CAP_PROP_FRAME_COUNT) < 108000 * .75:
            cmd = f"rclone -v copyto {self.source_video_ts}/{self.vid_label}/{self.vid_label}{self.hour}.ts {self.vname_ts}"
            print(cmd)
            subprocess.check_output(cmd, shell=True)
            assert cv2.VideoCapture(self.vname_ts).get(cv2.CAP_PROP_FRAME_COUNT) > 108000 * .75

        parallelize_update_dictionary(self._reencode_helper, range(self.nchunks), max_workers=self.nchunks, total_cpus=self.nchunks * 10)

        os.remove(self.vname_ts)
