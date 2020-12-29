import numpy as np
from tensorflow.keras.models import load_model

from part_1.run_attention import find_tfl_lights
from part_3.SFM import calc_TFL_dist
from part_3.SFM_standAlone import visualize
from part_4.adapter_1to2 import Adapter_1to2
from part_4.adapter_2to3 import Adapter_2to3


def padding_img_3D(src_img):
    height, width, d = src_img.shape
    padded_img_3D = np.zeros((height + 81, width + 81, d), dtype='uint8')
    padded_img_3D[40:height + 40, 40:width + 40, :] = src_img
    return padded_img_3D


def crop_img_by_indx(src_img, indx):
    padded_img_3D = padding_img_3D(src_img)
    return padded_img_3D[indx[0]:indx[0] + 81, indx[1]:indx[1] + 81, :]


def crop_img_by_indx_list(img, indx_list):
    cropped = [crop_img_by_indx(img, indx) for indx in indx_list]
    return [x for x in cropped if x.shape == (81, 81, 3)]


class TFLManager:
    def run(self, data):
        adapter = Adapter_1to2()
        image = np.array(data.curr.img)

        candidates, auxiliary = adapter.adapt(find_tfl_lights(image))
        croped_imgs = crop_img_by_indx_list(image, candidates)

        loaded_model = load_model("model.h5")
        cropped_imgs_predicts = loaded_model.predict(np.array(croped_imgs))

        # update data according
        adapter = Adapter_2to3()
        candidates_list = adapter.adapt(cropped_imgs_predicts, candidates)
        data.curr.traffic_light = np.array(candidates_list)

        # if data.prev:
        #     # curr = SFM.calc_TFL_dist(data.prev, data.curr, data.focal, data.pp)
        #     curr_container = calc_TFL_dist(data.prev, data.curr, data.focal, data.pp)
        #     visualize(data.prev, data.prev_frame_id, curr_container, data.curr_frame_id, data.focal, data.pp)

        if data.prev:  # and percents.shape[0] <= len(cropped_imgs):
            data.curr = calc_TFL_dist(data.prev, data.curr, data.focal, data.pp)
            # print(type(data.curr.traffic_lights_3d_location[0]))
            # data.curr.traffic_lights_3d_location = \
            #     np.array(
            #         [np.array(c) for c in data.curr.traffic_lights_3d_location if c[0] >= 0 and c[1] >= 0 and c[2] >= 0]
            #     )
            # np.array(filter(lambda c: c[0] >= 0 and c[1] >= 0 and c[2] >= 0, data.curr.traffic_lights_3d_location))
            visualize(data.prev, data.prev_frame_id, data.curr, data.curr_frame_id, data.focal, data.pp
                           )
