import numpy as np

from SFM_standAlone import FrameContainer
from file_utils import read_txt, read_pickle


class ProcessData:
    def __init__(self, pls_file):
        self.pls_file = pls_file

    def process_data(self, frames):
        lines = read_txt(self.pls_file)
        pkl_path = lines[0]
        data = read_pickle(pkl_path)

        frames.init_f_pp(data['flx'], data['principle_point'])

        for img in lines[1:]:
            curr_frame_id = int(img[31:33])
            curr_img_path = img
            curr_container = FrameContainer(curr_img_path)
            curr_container.traffic_light = np.array(data['points_' + str(curr_frame_id)][0])
            EM = np.eye(4)
            for i in range(curr_frame_id - 1, curr_frame_id):
                EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
            curr_container.EM = EM

            frames.update(curr_container)
            yield

    # def count_lines(self):
    #     lines = read_txt(self.pls_file)
    #     return len(lines) - 1
    #
    # def first_id(self):
    #     lines = read_txt(self.pls_file)
    #     return int(lines[1][31:33])
