# import json
# import os
# import time
# from typing import Any, Dict
#
# from adb_utils import setup_device
# from agent_wrapper import MiniCPMWrapper
# import numpy as np
# from PIL import Image
# import logging
#
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s %(levelname)s %(name)s: %(message)s")
#
# HISTORY_EXAMPLE_PATH = os.path.join(
#     os.path.dirname(__file__), "history_example.json")
#
#
# def get_break_point() -> list[int]:
#     """Placeholder for break point retrieval."""
#     # raise NotImplementedError("get_break_point is not implemented yet")
#     return [1, 2]
#
#
# def load_history_points(file_path: str) -> list[dict]:
#     """Loads cached user/assistant message pairs for history replay."""
#     if not os.path.exists(file_path):
#         logger.warning("History example file not found: %s", file_path)
#         return []
#
#     with open(file_path, "r", encoding="utf-8") as f:
#         raw_history = json.load(f)
#
#     if len(raw_history) % 2 != 0:
#         logger.warning(
#             "History example entries should be in user/assistant pairs.")
#
#     points: list[dict] = []
#     for i in range(0, len(raw_history) - 1, 2):
#         user_msg = raw_history[i]
#         assistant_msg = raw_history[i + 1]
#         if user_msg.get("role") != "user" or assistant_msg.get("role") != "assistant":
#             logger.warning("Unexpected role ordering at history index %s", i)
#             continue
#         points.append({"user": user_msg, "assistant": assistant_msg})
#
#     return points
#
#
# def _ensure_action_dict(action_raw: Any) -> Dict[str, Any]:
#     """Normalize the raw action payload into a dict for device.step."""
#     if isinstance(action_raw, dict):
#         return action_raw
#     if isinstance(action_raw, str):
#         try:
#             parsed = json.loads(action_raw)
#         except json.JSONDecodeError as exc:
#             raise ValueError(
#                 f"Action string is not valid JSON: {action_raw}") from exc
#         if isinstance(parsed, dict):
#             return parsed
#         raise ValueError(f"Action JSON did not produce a dict: {parsed}")
#     raise TypeError(f"Unsupported action type: {type(action_raw).__name__}")
#
#
# def build_user_content(query: str, screenshot_np: np.ndarray) -> list[dict]:
#     """Create a user message payload with current query and screenshot."""
#     return [
#         {
#             "type": "text",
#             "text": f"<Question>{query}</Question>\n当前屏幕截图：(<image>./</image>)",
#         },
#         {
#             "type": "image_url",
#             "image_url": {
#                 "url": f"data:image/jpeg;base64,{MiniCPMWrapper.encode_image(screenshot_np)}"
#             },
#         },
#     ]
#
#
# def run_task(query):
#     device = setup_device()
#     minicpm = MiniCPMWrapper(model_name='AgentCPM-GUI',
#                              temperature=1, use_history=True, history_size=2)
#     try:
#         break_point = get_break_point()
#     except NotImplementedError:
#         logger.warning(
#             "get_break_point not implemented, defaulting to empty break points.")
#         break_point = []
#     history_points = load_history_points(HISTORY_EXAMPLE_PATH)
#     is_finish = False
#     step_counter = 0
#
#     while not is_finish:
#         text_prompt = query
#         screenshot = device.screenshot(1120)
#         screenshot_np = np.array(screenshot)
#
#         if step_counter < len(history_points) and step_counter not in break_point:
#             current_point = history_points[step_counter]
#             action = _ensure_action_dict(
#                 current_point["assistant"].get("content"))
#
#             # Replay cached history so future calls still have the expected context
#             minicpm._push_history(
#                 "user", build_user_content(text_prompt, screenshot_np))
#             minicpm._push_history(
#                 "assistant", current_point["assistant"].get("content"))
#             logger.info("Reusing cached step %s", step_counter)
#         else:
#             response = minicpm.predict_mm(text_prompt, [screenshot_np])
#             action = _ensure_action_dict(response[3])
#
#         print(action)
#         is_finish = device.step(action)
#         time.sleep(2.5)
#         step_counter += 1
#
#     minicpm.save()
#     return is_finish
#
#
# if __name__ == "__main__":
#     run_task("去哔哩哔哩看李子柒的最新视频，并且点赞。")


import time
from hdc_utils import setup_device
import logging
import os
from agent_wrapper import MiniCPMWrapper
from utils import read_json
import numpy as np
from PIL import Image
from args_parser import parse_cli_args_from_init

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
from utils import (
    load_config,
    read_json,
    parse_cli_args_from_init,
)
from utils import Operate


def run_task(query):
    args = parse_cli_args_from_init()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    file_path_config = read_json(os.path.join(root_dir, args.get("file_setting_path")))
    load_config(
        root_dir,
        file_path_config["results_dir"],
        file_path_config["temp_dir"]
    )
    hdc_dir = os.path.join(root_dir, "hdc")
    os.environ["PATH"] = hdc_dir + os.pathsep + os.environ.get("PATH", "")
    operator1 = Operate(None, args.get("hdc_command"), args.get("factor"))
    os.environ['DATA_DIR'] = os.environ['RESULTS_DIR']
    device = setup_device()
    minicpm = MiniCPMWrapper(model_name='AgentCPM-GUI', temperature=1, use_history=True, history_size=10)

    step = 1
    is_finish = False
    while not is_finish:
        text_prompt = query
        encoded_image, _ = operator1.get_screenshot_data()
        operator1.dump_ui_tree(step)
        response = minicpm.predict_mm(text_prompt, encoded_image)
        action = response[3]
        print(action)
        is_finish = device.step(action)
        time.sleep(1.5)
        input()
        step += 1

    minicpm.save_full_history_to_json()
    return is_finish


if __name__ == "__main__":
    # run_task("去哔哩哔哩看李子柒的最新视频，并且点赞。")
    run_task("播放华为音乐中最近播放的第二首歌")
