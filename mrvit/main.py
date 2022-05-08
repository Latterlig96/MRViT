import argparse
from mrvit.config import Config
from mrvit.train import TrainTriggerer

parser = argparse.ArgumentParser(description="MRViT default runner")


parser.add_argument(
    "-c", "--config",
    type=str,
    dest="config",
    default="config.json",
    help="Path to json file that stores necessary configuration to run given stage"
)

def main():
    args = parser.parse_args()
    config = Config.parse_file(args.config)
    triggerer = TrainTriggerer(config)
    triggerer.trigger()
