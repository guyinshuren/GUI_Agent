# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

__author__ = "Zhipeng Hou"

import base64
import os
import time
import json
from typing import (
    Optional,
    Union
)

from loguru import logger
from colorama import Style


def read_json(file_path: str) -> Union[dict, list]:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def write_json(file_path: str,
               input_data: Union[dict, list],
               json_type: str = "dict",
               mode: str = 'w') -> None:
    if mode not in ('w', 'a'):
        raise ValueError("[mode] must be 'w' or 'a'")

    if json_type not in ('dict', 'list'):
        raise ValueError("[json_type] must be 'dict' or 'list'")
    
    if json_type == "dict":
        # input_data must be dict
        if not isinstance(input_data, dict):
            raise ValueError("For dict json_type, input_data must be a dict")
        
        # overwrite
        if mode == "w":
            data = input_data
        # append
        else:
            try:
                data = read_json(file_path)
                if not isinstance(data, dict):
                    raise ValueError("Existing JSON is not a dict, by default a initial dict")
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                data = {}

            dup_keys = set(data.keys()) & set(input_data.keys())
            if dup_keys:
                raise KeyError(f"append failed：the field is existed -> {dup_keys}")
            data.update(input_data)

    else:
        if mode == "w":
            if not isinstance(input_data, list):
                data = [input_data]
            else:
                data = input_data
        else:
            try:
                data = read_json(file_path)
                if not isinstance(data, list):
                    raise ValueError("Existing JSON is not a list, by default a initial list")
            except (FileNotFoundError, json.JSONDecodeError, ValueError):
                data = []
            if isinstance(input_data, list):
                data.extend(input_data)
            else:
                data.append(input_data)
    
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def encode_image(image_path: Optional[str] = None,
                 byte_stream: Optional[bytes] = None) -> str:
    if image_path is None and byte_stream is None:
        raise ValueError("args [image_path] and [byte_stream] should not empty for all.")
    
    if image_path is not None and byte_stream is not None:
        raise ValueError("args [image_path] and [byte_stream] should have values for one.")
    
    if image_path:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    else:
        return base64.b64encode(byte_stream).decode('utf-8')


def setup_logging(log_level: str = "INFO") -> None:
    log_file = os.path.join(os.environ['DATA_DIR'], f'appagent_test.log')

    logger.remove()
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level=log_level
    )


def load_config(root_dir: str,
                results_dir: str,
                temp_dir: str) -> None:
    os.environ['ROOT_DIR'] = root_dir
    os.environ['TEMP_DIR'] = os.path.join(root_dir, temp_dir)
    os.environ['RESULTS_DIR'] = os.path.join(root_dir, results_dir)


def track_usage(res_json: dict) -> dict:
    usage = res_json['usage']
    prompt_tokens, completion_tokens, total_tokens = usage['prompt_tokens'], usage['completion_tokens'], usage['total_tokens']

    if "gpt-4o" in res_json['model']:
        prompt_token_price = (2.5 / 1000000) * prompt_tokens
        completion_token_price = (10 / 1000000) * completion_tokens
        return {
            "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "id": res_json.get('id', 'unknown'),
            "model": res_json.get('model', 'unknown'),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_token_price": prompt_token_price,
            "completion_token_price": completion_token_price,
            "total_price": prompt_token_price + completion_token_price
        }
    else:    
        return {
            "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "id": res_json.get('id', 'unknown'),
            "model": res_json.get('model', 'unknown'),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }


# apikey out of quota
class OutOfQuotaException(Exception):
    "Raised when the key exceeded the current quota"
    def __init__(self, 
                 key: str, 
                 cause: Optional[str] = None) -> None:
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()


# api key with no permission
class AccessTerminatedException(Exception):
    "Raised when the key has been terminated"
    def __init__(self, 
                 key: str, 
                 cause: Optional[str] = None) -> None:
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()


def print_out(mesg: str,
              logout: bool = True,
              stdout: bool = False,
              stdout_color: str = "",
              log_level: str = "info") -> None:
    if logout:
        if log_level.lower() == "info":
            logger.info(mesg)
        elif log_level.lower() == "error":
            logger.error(mesg)
        elif log_level.lower() == "debug":
            logger.debug(mesg)
        elif log_level.lower() == "warning":
            logger.warning(mesg)
        
    if stdout:
        if stdout_color:
            print(stdout_color + mesg + Style.RESET_ALL)
        else:
            print(mesg)
