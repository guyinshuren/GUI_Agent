# test_pipeline.py
import json
from pathlib import Path
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# ========= 配置区 =========

# 已格式化的新指令
NEW_QUERY_JSON = "new_query_format.json"   # { "sentences": [ ... ] }

# 老指令（自然语言 + 向量）
OLD_QUERY_JSON = "old_query.json"         # { "sentences": [ ... ] }
OLD_QUERY_NPY  = "old_query.npy"          # shape = (N, D)

# 相似度阈值 & topk
SIM_THRESHOLD = 0.7
TOP_K = 5

# GTE 模型名称
GTE_MODEL_NAME = "/root/autodl-tmp/gte"

# Qwen3-8B 本地模型目录
QWEN_MODEL_DIR = "/root/autodl-tmp/Qwen3-8B"


# ========= 工具函数 =========

def load_sentences(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sents = data.get("sentences", [])
    if not sents:
        raise ValueError(f"{path} 中未找到 'sentences' 或列表为空")
    return sents


def load_old_data():
    """加载老指令的自然语言和 npy 向量"""
    sentences = load_sentences(OLD_QUERY_JSON)
    embeddings = np.load(OLD_QUERY_NPY)
    if embeddings.shape[0] != len(sentences):
        raise ValueError("old_query.json 中句子数与 old_query.npy 行数不一致")
    return sentences, embeddings


def cosine_sim(new_vec: np.ndarray, old_vecs: np.ndarray) -> np.ndarray:
    """new_vec: (D,), old_vecs: (N, D)"""
    new_norm = new_vec / (np.linalg.norm(new_vec) + 1e-12)
    old_norm = old_vecs / (np.linalg.norm(old_vecs, axis=1, keepdims=True) + 1e-12)
    return old_norm @ new_norm   # (N,)


class QwenSelector:
    """用 Qwen3-8B 在候选中选出最相似的一条"""

    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            torch_dtype="auto",
            use_safetensors=True,
        )
        self.model.eval()

    def select_best(self, new_instruction: str, candidates: list[str]) -> int:
        """
        返回候选列表中的 index（0-based）
        """
        numbered = "".join([f"{i+1}.{c}" for i, c in enumerate(candidates)])
        system_content = (
            "你是一个语义相似度判别工具，现在有这几条老指令：" +
            numbered +
            "，给你一条新指令，从老指令中找出语义最相似的一条，"
            "只输出该指令的编号（阿拉伯数字），不要有任何解释和复述。"
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": f"新指令：{new_instruction}"}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            gen_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=16,
            )
        gen_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, gen_ids)
        ]
        resp = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

        m = re.search(r"\d+", resp)
        if not m:
            raise RuntimeError(f"Qwen 输出中未找到编号: {resp}")
        idx = int(m.group()) - 1
        if not (0 <= idx < len(candidates)):
            raise RuntimeError(f"Qwen 输出编号越界: {resp} (len={len(candidates)})")
        return idx


# ========= 主测试流程 =========

def main():
    # 1. 检查文件
    if not Path(NEW_QUERY_JSON).exists():
        raise FileNotFoundError(f"找不到 {NEW_QUERY_JSON}")
    if not Path(OLD_QUERY_JSON).exists():
        raise FileNotFoundError(f"找不到 {OLD_QUERY_JSON}")
    if not Path(OLD_QUERY_NPY).exists():
        raise FileNotFoundError(f"找不到 {OLD_QUERY_NPY}")

    # 2. 加载数据
    new_sents = load_sentences(NEW_QUERY_JSON)
    old_sents, old_embs = load_old_data()

    print(f"新指令数量: {len(new_sents)}")
    print(f"老指令数量: {len(old_sents)}")

    # 3. 加载模型
    print("\n[加载 GTE 模型] ...")
    embedder = SentenceTransformer(GTE_MODEL_NAME)

    print("[加载 Qwen3-8B] ...")
    qwen = QwenSelector(QWEN_MODEL_DIR)

    total = len(new_sents)
    success = 0
    mismatches = []

    # 4. 逐条测试
    for i, new_ins in enumerate(new_sents):
        expected_index = i + 1  # new_query_format 中的序号（从1开始）
        print("\n" + "=" * 60)
        print(f"测试样本 #{expected_index}")
        print("新指令：", new_ins)

        # 4.1 编码新指令
        new_vec = embedder.encode(new_ins, convert_to_numpy=True)

        # 4.2 计算与老指令的相似度
        sims = cosine_sim(new_vec, old_embs)
        sorted_idx = np.argsort(-sims)
        top_idx = sorted_idx[:TOP_K]
        top_scores = sims[top_idx]

        print("Top5 相似度：")
        for rank, (idx0, sc) in enumerate(zip(top_idx, top_scores), 1):
            print(f"  排名{rank}: 全局序号{idx0+1}, 相似度={sc:.4f}, 指令={old_sents[idx0]}")

        # 4.3 阈值判断
        if top_scores[0] <= SIM_THRESHOLD:
            print("  → 最高相似度<=阈值，判定为无可复用任务")
            mismatches.append({
                "case_id": expected_index,
                "query": new_ins,
                "expected": expected_index,
                "got": None,
                "reason": "no_match",
            })
            continue

        # 4.4 从 topK 中选出 > 阈值 的候选
        cand_global_idx = [idx0 for idx0, sc in zip(top_idx, top_scores) if sc > SIM_THRESHOLD]
        cand_texts = [old_sents[idx0] for idx0 in cand_global_idx]

        print("\n超过阈值的候选：")
        for j, (gidx, txt) in enumerate(zip(cand_global_idx, cand_texts), 1):
            print(f"  局部{j} -> 全局{gidx+1}, 指令：{txt}")

        # 理论上 cand_global_idx 至少有一个（因为 top1>阈值），但保险起见再判断一下：
        if not cand_global_idx:
            print("  → 出现异常：top1>阈值但候选为空")
            mismatches.append({
                "case_id": expected_index,
                "query": new_ins,
                "expected": expected_index,
                "got": None,
                "reason": "empty_candidates",
            })
            continue

        # 4.5 用 Qwen 在候选中做最终判断
        local_idx = qwen.select_best(new_ins, cand_texts)
        global_idx0 = cand_global_idx[local_idx]  # 0-based
        predicted_index = global_idx0 + 1         # 对外从1开始

        if predicted_index == expected_index:
            print(f"\n✅ 匹配成功：期望序号 = {expected_index}, 实际序号 = {predicted_index}")
            success += 1
        else:
            print(f"\n❌ 匹配失败：期望序号 = {expected_index}, 实际序号 = {predicted_index}")
            mismatches.append({
                "case_id": expected_index,
                "query": new_ins,
                "expected": expected_index,
                "got": predicted_index,
                "reason": "wrong_index",
            })

    # 5. 汇总
    print("\n" + "=" * 60)
    print(f"测试完成：通过 {success}/{total} 条")

    if mismatches:
        print("\n以下样本为异常（包括无匹配、匹配错序号等）：")
        for m in mismatches:
            print("-" * 40)
            print(f"样本序号（new_query_format.json 中，从1开始）：{m['case_id']}")
            print(f"  指令：{m['query']}")
            print(f"  期望老指令序号：{m['expected']}")
            print(f"  实际老指令序号：{m['got']}")
            reason = m["reason"]
            if reason == "no_match":
                print("  异常原因：最高相似度<=阈值，被判定为无可复用任务")
            elif reason == "wrong_index":
                print("  异常原因：选中的老指令序号与期望不一致")
            elif reason == "empty_candidates":
                print("  异常原因：逻辑异常（top1>阈值但候选列表为空）")
            else:
                print(f"  异常原因：{reason}")
    else:
        print("🎉 所有样本都匹配到了正确序号！")


if __name__ == "__main__":
    main()
