import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Pitcher Training")
    parser.add_argument('-c', "--camera", dest="camera", type=int, default=0, help='Index of the camera to use. Default 0, usually this is the camera on the laptop display')
    parser.add_argument('-d', "--debug", dest="debugging", type=bool, default=False, help='Print all windows. This option is for debugging')
    parser.add_argument('-tf', "--test-folder", dest="test_folder", type=int, default=0, help='Select a test folder from video folder')
    parser.add_argument('-ti', "--test-frame", dest="test_frame", type=int, default=0, help='Select a test frame from test folder')

    return parser.parse_args()

args = parse_args()
