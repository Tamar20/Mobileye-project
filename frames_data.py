
class FramesData:
    def __init__(self):
        self.prev = None
        self.curr = None
        self.focal = None
        self.pp = None

    def init_f_pp(self, focal, pp):
        self.focal = focal
        self.pp = pp

    def update(self, curr):
        self.prev = self.curr
        self.curr = curr
