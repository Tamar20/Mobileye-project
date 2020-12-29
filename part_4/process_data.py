import numpy as np

from part_3.SFM_standAlone import FrameContainer
from part_4.file_utils import read_txt, read_pkl


class ProcessData:
    def __init__(self, pls_file):
        self.pls_file = pls_file

    def process_data(self, frame_data):
        lines = read_txt(self.pls_file)
        pkl_path = lines[0]
        data = read_pkl(pkl_path)
        frame_data.init_focal_pp(data['flx'], data['principle_point'])

        for curr_img_path in lines[1:]:
            curr_frame_id = int(curr_img_path[-18:-16])
            curr_frame_container = FrameContainer(curr_img_path)
            curr_frame_container.traffic_light = np.array(data['points_' + str(curr_frame_id)][0])

            EM = np.eye(4)

            for i in range(curr_frame_id - 1, curr_frame_id):
                EM = np.dot(data['egomotion_' + str(i) + '-' + str(i + 1)], EM)

            curr_frame_container.EM = EM

            frame_data.update(curr_frame_container, curr_frame_id)
            yield frame_data
