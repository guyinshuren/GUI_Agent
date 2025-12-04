import json
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 读取 JSON 文件
json_path = "old_query.json"  # 你提供的文件路径
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

queries = data["sentences"]  # 保持原有顺序

# 2. 加载 GTE 模型
model = SentenceTransformer("../gte")

# 3. 编码所有 query
embeddings = model.encode(queries, convert_to_numpy=True, batch_size=32)

# 4. 保存为 .npy 文件（按顺序）
save_path = "old_query.npy"
np.save(save_path, embeddings)

print("已保存:", save_path)
print("向量形状:", embeddings.shape)
