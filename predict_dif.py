from http import HTTPStatus
from dashscope import Generation
import time
import json
import os


def predict_dif(base_workflow, base_query, new_query, apikey):
    """
    比较工作流差异，返回不同步骤编号列表。

    参数：
        base_workflow: list[str]，基准工作流
        base_query: str，基准指令
        new_query: str，新指令
        apikey: str，API Key

    返回：
        list[int]，new_workflow 与 base_workflow 的不同步骤编号
    """

    # System Prompt
    messages = [
        {
            "role": "system",
            "content": """你是一个用于分析工作流差异的助手。

给定：
- 一个基准工作流（base_workflow），包含按顺序执行的步骤；
- 一个基准指令（base_query）；
- 一个新的指令（new_query）；

你的任务是：
1. 根据新的指令，推理出新的工作流（new_workflow）；
2. 比较 new_workflow 与 base_workflow；
3. 输出它们不同的步骤编号（从 1 开始计数）；
4. 仅输出一个 Python 列表，例如：[6,7]；
5. 不要输出多余解释、描述或文字。

示例：
输入：
base_workflow = [
    "打开美团", "点击搜索框", "输入星巴克",
    "寻找离目标位置最近的星巴克", "点击星巴克商家",
    "搜索冰美式", "下单"
]
base_query = "帮我点一杯星巴克的冰美式"
new_query = "帮我点一杯星巴克的拿铁"

输出：
[6,7]"""
        }
    ]

    # 用户输入
    user_input = f"""
base_workflow = {base_workflow}
base_query = "{base_query}"
new_query = "{new_query}"
"""
    messages.append({"role": "user", "content": user_input})

    # 调用模型
    start_time = time.time()
    response = Generation.call(
        model="qwen-max",
        messages=messages,
        result_format="message",
        api_key=apikey
    )
    elapsed = time.time() - start_time
    print(f"请求耗时：{elapsed:.2f} 秒")

    if response.status_code == HTTPStatus.OK:
        output_text = response.output.choices[0].message.content.strip()
        # 尝试直接 eval 成 Python list
        try:
            return eval(output_text)
        except Exception:
            # 如果解析失败，返回原始文本
            return output_text
    else:
        raise RuntimeError(f"请求出错: {response}")


def get_base_query_by_index(index, file_path="Database/query.json"):
    """
    从 Database/query.json 中读取所有句子，根据序号提取 base_query
    """
    # 读取 JSON 文件
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = data.get("sentences", [])

    if not (1 <= index <= len(sentences)):
        raise ValueError(f"序号 {index} 超出范围 1~{len(sentences)}")

    # 提取指定句子
    return sentences[index - 1]


def load_base_workflow_by_index(index, folder_path="Database/refined"):
    """
    根据序号读取 Database/refined/{index}.json 文件，
    提取每个 step 中 assistant 的 thought，生成 base_workflow 列表
    """
    file_path = os.path.join(folder_path, f"{index}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} 不存在")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_workflow = []

    for step in data:
        if step.get("role") == "assistant":
            content = step.get("content")
            if isinstance(content, str):
                try:
                    # 尝试解析为字典
                    thought_data = json.loads(content)
                    thought_text = thought_data.get("thought")
                    if thought_text:
                        base_workflow.append(thought_text)
                except json.JSONDecodeError:
                    # 如果解析失败，直接当作文本
                    base_workflow.append(content)

    return base_workflow


# ====== 使用示例 ======
if __name__ == "__main__":
    index = int(input("请输入句子序号（1-10）："))
    base_query = get_base_query_by_index(index)
    base_workflow = load_base_workflow_by_index(index)
    print("选中的 base_query:", base_query)
    print(base_workflow)
    apikey = "sk-bd4a7ab5beb44be7badc85bf93451360"
    new_query = "用微信给周喆宇发消息 今天晚上一起讨论吧"
    diff_steps = predict_dif(base_workflow, base_query, new_query, apikey)
    print("差异步骤:", diff_steps)
 