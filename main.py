from typing import Any, Dict
from sim.user_to_index import call_cloud_for_candidates, find_best_match
from predict_dif import get_base_query_by_index, load_base_workflow_by_index, predict_dif
from adb_utils import setup_device
from agent_wrapper import MiniCPMWrapper
from run_agent import load_history_points, _ensure_action_dict, build_user_content
import numpy as np
import logging
import json
import os
import time

GTE_MODEL_NAME = "gte"
APIKEY = "sk-bd4a7ab5beb44be7badc85bf93451360"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main():
    new_query = input("请输入自然语言指令：").strip()
    if not new_query:
        print("未检测到输入，退出。")
        return

    candidate_sentences = call_cloud_for_candidates(new_query)

    sel = input("\n选择序号：")
    chosen_instruction = candidate_sentences[int(sel) - 1]


    index = find_best_match(chosen_instruction, GTE_MODEL_NAME)

    break_point = []

    if index == 0:
        print("\n没有匹配到可复用任务")
    else:
        print(f"\n匹配到历史任务，全局序号：{index}")
        base_query = get_base_query_by_index(index)
        base_workflow = load_base_workflow_by_index(index)
        print("选中的 base_query:", base_query)
        print(base_workflow)
        break_point = predict_dif(base_workflow, base_query, new_query, APIKEY)
        print("差异步骤:", break_point)

    device = setup_device()
    minicpm = MiniCPMWrapper(model_name='AgentCPM-GUI',
                             temperature=1, use_history=True, history_size=2)

    history_points = load_history_points(f"Database/refined/{index}.json")
    is_finish = False
    step_counter = 1

    start_time = time.time()


    while not is_finish:
        text_prompt = chosen_instruction
        screenshot = device.screenshot(1120)
        screenshot_np = np.array(screenshot)

        if not index == 0 and step_counter <= len(history_points) and step_counter not in break_point:
            current_point = history_points[step_counter - 1]
            action = _ensure_action_dict(
                current_point["assistant"].get("content"))

            # Replay cached history so future calls still have the expected context
            minicpm._push_history(
                "user", build_user_content(text_prompt, screenshot_np))
            minicpm._push_history(
                "assistant", current_point["assistant"].get("content"))
            logger.info("Reusing cached step %s", step_counter)
        else:
            response = minicpm.predict_mm(text_prompt, [screenshot_np])
            action = _ensure_action_dict(response[3])

        print(action)
        is_finish = device.step(action)
        time.sleep(2.5)
        step_counter += 15

    end_time = time.time()
    print("总用时:%s", end_time - start_time)
    minicpm.save_full_history_to_json()
    return is_finish


if __name__ == "__main__":
    main()
