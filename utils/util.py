import os


def check_directories(path):
    if not os.path.exists(path):
        print("makedirs", path)
        os.makedirs(path)