from part_4.frames_data import FramesData
from part_4.process_data import ProcessData
from part_4.tfl_manager import TFLManager


class Controller:

    def process(self, pls_path):
        data = FramesData()
        proc_data = ProcessData(pls_path)

        manager = TFLManager()

        for curr_img_data in proc_data.process_data(data):
            manager.run(curr_img_data)
