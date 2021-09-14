"""Argument parser functions."""

import os.path as osp
import argparse
import sys
import os

from config.defaults import get_cfg


def parse_args():
    """
    Parse the following arguments for a default parser.
    Args:
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Training and testing pipeline."
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:6006",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        type=str,
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Path to output directory",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See /config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `cfg_file`, `output_dir` and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    if args.output_dir is not None:
        cfg.OUTPUT_DIR = args.output_dir

    # Create output directory if it doesn't exits
    if not osp.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    return cfg