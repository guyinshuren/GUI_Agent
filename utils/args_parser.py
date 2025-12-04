# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"

import argparse


def parse_cli_args_from_init() -> dict:
    parser = argparse.ArgumentParser(description="run GUI Agent test framework")
    parser.add_argument(
        "--file-setting-path", 
        default="configs/file_path_config.json", 
        type=str, 
        help="file setting path"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO", 
        type=str, 
        help="logger record level"
    )
    parser.add_argument(
        "--provider", 
        default="sample", 
        type=str, 
        help="which agent will be used"
    )
    parser.add_argument(
        "--hdc-command", 
        default="hdc/hdc.exe", 
        type=str, 
        help="hdc cli"
    )
    parser.add_argument(
        "--max-retries", 
        default=5, 
        type=int, 
        help="each step will be tried for maximum time"
    )
    parser.add_argument(
        "--factor", 
        default=0.5, 
        type=float, 
        help="resize img proportion"
    )
    args = vars(parser.parse_args())
    return args
