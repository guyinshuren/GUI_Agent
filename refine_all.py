import json
from openai import OpenAI

# 你的模型 client
client = OpenAI(
    api_key="sk-bd4a7ab5beb44be7badc85bf93451360",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

SYSTEM_PROMPT = """
你是一个“动作增强型意图压缩器”（Action-Aware Thought Compressor）。

你的任务是处理一个完整的 GUI-Agent 工作流 JSON（多个 step）。每个 step 中：

- 用户消息（role=user）不处理
- 助手消息（role=assistant）包含一个 JSON 字符串，其中包括：
    - thought 字段（自然语言推理）
    - 可能存在的动作字段，例如 POINT / TYPE / PRESS / CLEAR / to / STATUS

你必须对每个 assistant step 的 thought，结合动作字段，进行“动作增强型意图压缩”，并返回压缩后的 short-thought。

【压缩规则】

1. 行为驱动（Action-aware）
   - 依据动作字段判断真实意图：
     POINT → 点击  
     TYPE → 输入  
     PRESS → 按键  
     CLEAR → 清空  
     to（滑动）→ 滑动页面  
     STATUS=finish → 最终确认类意图

2. 限制输出
   - 仅输出“意图短语”，不包含动作、不包含坐标、不包含解释。
   - 删除所有冗余语言：
       “我需要…”，“界面显示…”，“首先…”，“当前看到…”
   - 不得引用屏幕细节、视觉描述、位置描述。

3. 输出格式
   - 输出一个数组，每个元素对应原 workflow 中的 assistant step
   - 每个元素为：
       {"thought": "<凝练后的短意图>"}

最终只输出数组，不包含任何解释或其他自然语言。
"""


def compress_workflow(input_path, output_path):
    # 加载原 workflow
    with open(input_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # 收集 assistant steps 的 JSON
    assistant_steps = []
    for message in workflow:
        if message["role"] == "assistant":
            try:
                step_obj = json.loads(message["content"])
                assistant_steps.append(step_obj)
            except:
                assistant_steps.append({"thought": message["content"]})

    # 调用模型批处理压缩
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(assistant_steps, ensure_ascii=False)}
        ]
    )

    compressed_list = json.loads(response.choices[0].message.content)

    # 将精炼 thought 写回原 workflow
    idx = 0
    for message in workflow:
        if message["role"] == "assistant":
            step_json = json.loads(message["content"])
            step_json["thought"] = compressed_list[idx]["thought"]
            message["content"] = json.dumps(step_json, ensure_ascii=False)
            idx += 1

    # 保存新 workflow
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, ensure_ascii=False, indent=2)

    print(f"Saved compressed workflow → {output_path}")


if __name__ == "__main__":
    compress_workflow("full_history.json", "workflow_compressed.json")
