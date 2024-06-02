import os

from configs import video_directory, frame_bounds, video_files_dir

class NoMoreVideo(Exception):
    pass

class VideoData:

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
            # print(start, stop)
            if self.decoder is None or self.decoder[0] != start_chunk_idx:
                import hwang
                if not os.path.exists(self.vname_chunked(start_chunk_idx)):
                    raise NoMoreVideo
                self.decoder = (start_chunk_idx, hwang.Decoder(self.vname_chunked(start_chunk_idx)))
            num_frames = self.decoder[1].video_index.frames()
            # print(f"num frames {num_frames}")
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

    def check_vids(self):

        # check that all ten min chunks exist
        all_good = all([os.path.exists(self.vname_chunked(i)) for i in range(self.nchunks)])
        if all_good:
            return
        else:
            for i in range(self.nchunks):
                if not os.path.exists(self.vname_chunked(i)):
                    print(f"File Not Found: {self.vname_chunked(i)}")
            return