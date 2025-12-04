import json
import numpy as np
from typing import List, Tuple
import torch
from http import HTTPStatus
from openai import OpenAI
import time
import re

# ==== GTE ====
from sentence_transformers import SentenceTransformer

# ==== 云端 LLM（语义归一化，照 demo2.py） ====
from dashscope import Generation

# ==== 本地 Qwen3-8B（辅助智能体，参考 run_sim.py） ====
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ================== 配置区 ==================
# 旧指令文本 & 向量
OLD_QUERY_JSON = "C:/Users/86159/Desktop/gist 2.0/sim/old_query.json"  # { "sentences": [ ... ] }
OLD_QUERY_NPY = "C:/Users/86159/Desktop/gist 2.0/sim//old_query.npy"  # shape = (N, D)
# 阈值 & TOP-K
SIM_THRESHOLD = 0.7
TOP_K = 5

# 云端 LLM
DASHSCOPE_API_KEY = "sk-bd4a7ab5beb44be7badc85bf93451360"

# 本地 Qwen3-8B 模型目录（按你的 run_sim.py 改）
QWEN_MODEL_DIR = "/root/autodl-tmp/Qwen3-8B"


# ================== 工具函数 ==================

def load_old_tasks(json_path: str, npy_path: str) -> Tuple[List[str], np.ndarray]:
    """加载历史指令（自然语言 + 向量）"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sentences = data["sentences"]
    embeddings = np.load(npy_path)
    if embeddings.shape[0] != len(sentences):
        raise ValueError("old_query.json 中的句子数量与 old_query.npy 的向量数量不一致！")
    return sentences, embeddings


def call_cloud_for_candidates(user_input: str) -> List[str]:
    """
    使用云端大模型（qwen-max 等）对用户指令做语义归一化，
    返回候选指令列表（>=3 条），system prompt 完全照 demo2.py。
    """
    apikey = DASHSCOPE_API_KEY

    # ====== 初始化 system prompt（照 demo2.py 原文） ======
    messages = [
        {
            'role': 'system',
            'content': '''你是一个面向 GUI agent 的语义归一化助手。你的任务是：
你是一个面向 GUI agent 的语义归一化助手。你的任务是：

理解用户输入的自然语言指令。

在内心判断该指令最适合归类到以下五种指令类别中的哪一种（只在思考中判断，不要把类别名称输出给用户）。

选定一个最合适的类别后，严格按照该类别对应的语义模板，生成至少 3 条语义等价但表述略有差异的候选指令。

所有候选指令必须是自然语言句子，用户可以直接理解。
你只能输出候选指令本身，每条指令单独一行，不允许输出任何解释、类别名称、步骤说明、JSON、代码或列表结构。

以下是五类指令类别和模板，你需要在内部使用它们进行分类和生成。

类别 1：控制类或配置类指令
适用范围：调节系统或应用的状态，如音量、亮度、网络、开关、模式等。
语义模板（概念框架）：
[动作] + [对象] + [属性/参数/程度/类型/量化]
动作示例：调高、调低、打开、关闭、设置、切换、启用、禁用
对象示例：音量、屏幕亮度、蓝牙、WiFi、飞行模式、省电模式
参数示例：一级、两格、50%、夜间模式、静音模式
示例（不要原样输出）：
调高音量一级
设置屏幕亮度为五十百分比
打开飞行模式

类别 2：路径类或多层级导航指令
适用范围：逐层进入多个界面或菜单，例如“设置 → 隐私 → 定位”。
语义模板（概念框架）：
[动作] + [路径节点1] + [路径节点2] + [路径节点3或最终操作]
动作示例：进入、打开、前往、切换到
路径示例：设置、隐私、定位服务、相册、最近、通知设置
示例（不要原样输出）：
进入设置 进入隐私 关闭定位服务
打开相册 进入最近文件夹
进入微信 进入支付页面

类别 3：可视控件类或界面控件操作指令
适用范围：按钮、弹窗、标签页、图标等控件操作。
语义模板（概念框架）：
[动作] + [控件的定位信息] + [可选的结果]
动作示例：点击、轻点、长按、双击、切换、关闭、展开
控件定位信息可描述为：按钮文本、屏幕位置、控件类型、顺序位置等
示例（不要原样输出）：
点击右上角菜单按钮 打开更多选项
点击允许按钮 同意权限申请
关闭提示窗口
切换到底部第二个标签页

类别 4：手势类指令
适用范围：滑动、拖动、缩放、长按等触摸手势。
语义模板（概念框架）：
[手势动作] + [方向/目标区域/对象] + [可选结果]
手势动作示例：上滑、下滑、左滑、右滑、长按、双击、拖动、缩放
目标示例：解锁界面、通知栏、页面底部、当前卡片、图片
示例（不要原样输出）：
上滑解锁屏幕
下滑打开通知栏
左滑返回上一页
长按应用图标 打开快捷菜单

类别 5：任务类或内容操作类指令
适用范围：发送消息、创建记录、搜索内容等具有内容目标的操作。
语义模板（概念框架）：
[动作] + [对象/载体] + [任务内容或文本内容]
动作示例：发送、创建、新建、记录、搜索、查找、添加
对象示例：消息、短信、备忘录、日程、联系人、搜索栏
示例（不要原样输出）：
给小明发送消息 我到了
在备忘录创建新笔记
搜索手机壳优惠信息

你的工作流程：
内部判断用户输入属于五类中的哪一类，不要输出类别。
只使用该类对应的语义模板，生成三条语义等价的候选自然语言指令。
指令必须符合模板中的槽位结构，但不输出方括号，占位符需要替换为自然语言。
只输出候选指令，不输出任何说明。
每条指令必须独立一行，不能有序号或前缀符号。


请对用户接下来的每一句输入，生成三条符合上述要求的候选指令,最后不要有句号。
 
'''

        }
    ]

    # ====== 用户输入 ======
    messages.append({'role': 'user', 'content': user_input})

    # ====== 调用模型 ======
    start_time = time.time()
    response = Generation.call(
        model="qwen-max",
        messages=messages,
        result_format='message',
        api_key=apikey
    )
    end_time = time.time()
    elapsed = end_time - start_time

    # ====== 输出解析 ======
    if response.status_code == HTTPStatus.OK:
        output_text = response.output.choices[0].message.content.strip()

        # 解析每行候选句
        candidate_sentences = [line.strip()
                               for line in output_text.splitlines()
                               if line.strip()]

        if len(candidate_sentences) < 3:
            print("⚠ 云端返回候选不足 3 条，当前为：", len(candidate_sentences))
        print("\n候选句列表:")
        for i, sent in enumerate(candidate_sentences, start=1):
            print(f"{i}. {sent}")
        # print(f"\n语义归一化用时：{elapsed:.2f} 秒")

        return candidate_sentences
    else:
        print("❌ 请求错误:", response)
        raise RuntimeError("云端 LLM 调用失败")


def cosine_sim_with_all(new_vec: np.ndarray, old_vecs: np.ndarray) -> np.ndarray:
    """
    计算 new_vec 与 old_vecs 中每一行的余弦相似度
    """
    new_norm = new_vec / (np.linalg.norm(new_vec) + 1e-12)
    old_norm = old_vecs / (np.linalg.norm(old_vecs, axis=1, keepdims=True) + 1e-12)
    sims = old_norm @ new_norm  # shape = (N,)
    return sims


# ================== Qwen3-8B 辅助智能体 ==================

class QwenSelector:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def select_best(self, user_instruction: str, candidates: List[str]) -> int:
        """
        使用云端 Qwen API 从 candidates 中选出语义最相似的一条
        返回候选的局部序号（0-based）
        """

        numbered = "".join(
            [f"{i + 1}.{sent}" for i, sent in enumerate(candidates)]
        )

        system_prompt = (
                "你是一个语义相似度判别工具，现在有这几条老指令：" +
                numbered +
                "，给你一条新指令，从老指令中找出语义上最相似的一条，只输出该指令的编号，不要有任何解释和复述。"
        )

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"新指令：{user_instruction}"}
        ]

        response = Generation.call(
            model="qwen-max",
            messages=messages,
            result_format='message',
            api_key=self.api_key
        )

        if response.status_code != HTTPStatus.OK:
            raise RuntimeError(f"云端调用失败：{response}")

        resp = response.output.choices[0].message.content.strip()

        # 提取数字
        m = re.search(r"\d+", resp)
        if not m:
            raise RuntimeError(f"Qwen 输出中未找到编号: {resp}")

        idx = int(m.group()) - 1
        if not (0 <= idx < len(candidates)):
            raise RuntimeError(f"Qwen 输出编号越界: {resp} (len={len(candidates)})")

        return idx


# ================== 主流程 ==================

def find_best_match(chosen_instruction: str, gte_model_name: str) -> int:
    """
    输入：用户最终选择的一条规范化指令（字符串）
    输出：
        - 匹配到的全局序号（1-based）
        - 如果没有可复用任务，则返回 0
    """

    print("\n[步骤2] 加载 GTE 模型并编码所选指令 ...")
    embedder = SentenceTransformer(gte_model_name)
    new_vec = embedder.encode(chosen_instruction, convert_to_numpy=True)

    print("\n[步骤3] 加载历史指令及其向量 ...")
    old_sentences, old_embeddings = load_old_tasks(OLD_QUERY_JSON, OLD_QUERY_NPY)
    print(f"历史指令数量：{len(old_sentences)}")

    print("\n[步骤4] 计算与所有历史指令的余弦相似度 ...")
    sims = cosine_sim_with_all(new_vec, old_embeddings)

    # Top5
    sorted_idx = np.argsort(-sims)
    top5_idx = sorted_idx[:TOP_K]
    top5_scores = sims[top5_idx]

    # 阈值判断
    if top5_scores[0] <= SIM_THRESHOLD:
        print("\n无可复用任务（最高相似度 <= 阈值）")
        return 0

    # 筛选超过阈值的
    candidate_global_indices = [
        idx for idx, score in zip(top5_idx, top5_scores) if score > SIM_THRESHOLD
    ]
    candidate_texts = [old_sentences[i] for i in candidate_global_indices]

    if not candidate_global_indices:
        print("\n无可复用任务（无候选超过阈值）")
        return 0

    print("\n[步骤6] 调用 Qwen3-8B 在这些候选中做最终判别 ...")
    qwen = QwenSelector(DASHSCOPE_API_KEY)
    local_idx = qwen.select_best(chosen_instruction, candidate_texts)

    global_idx = candidate_global_indices[local_idx]

    return global_idx + 1  # 返回1-based序号
