
class FramesData:
    def __init__(self):
        self.prev = None
        self.curr = None
        self.focal = None
        self.pp = None  # principle_point
        self.curr_frame_id = None
        self.prev_frame_id = None

    def init_focal_pp(self, focal, pp):
        self.focal = focal
        self.pp = pp

    def update(self, curr, curr_id):
        self.prev = self.curr
        self.prev_frame_id = self.curr_frame_id
        self.curr = curr
        self.curr_frame_id = curr_id