import argparse
import os

def expand_path(p):
    return os.path.expanduser(p)

def get_args():
    parser = argparse.ArgumentParser(description='Train an LSTM for the prediction of causal precedence.')
    parser.add_argument(
        '-c', '--config',
        dest='config_file',
        required=True,
        help='a .yml config file'
    )
    return parser.parse_args()
