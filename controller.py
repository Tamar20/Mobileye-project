from frames_data import FramesData
from process_data import ProcessData
from tfl_manager import TFLManager


class Controller:

    def process(self, pls_path):
        print("in Controller::process:")
        data = FramesData()
        p = ProcessData(pls_path)

        manager = TFLManager()

        for process_data in p.process_data(data):
            manager.run(data)


def main():
    print("in main: ")
    my_controller = Controller()
    my_controller.process('./pls_files/pls.txt')


if __name__== "__main__":
    main()