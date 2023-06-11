import _init_paths
import argparse
from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist


def parse_args():
    """
    args for evaluation.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='ostrack', help='training script name')
    parser.add_argument('--config', type=str, default='base_ori', help='yaml configure file name')
    parser.add_argument('--dataset', type=str, default='lasot', help='dataset name')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    trackers = []
    trackers.extend(trackerlist(args.script, args.config, "None", None, args.config))

    dataset = get_dataset(args.dataset)

    print_results(trackers, dataset, args.dataset, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))