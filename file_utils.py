import pickle

def read_txt(file):
    with open(file, "r") as f:
        lines = [line.rstrip() for line in f]
    return lines

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as pklfile:
        data = pickle.load(pklfile, encoding='latin1')
    return data

# def read_txt_file(path):
#     with open(path) as f:
#         lines = f.readlines()
#     return lines
#
#
# def read_pickle_file(path):
#     with open(path, 'rb') as pklfile:
#         data = pickle.load(pklfile, encoding='latin1')
#     return data
