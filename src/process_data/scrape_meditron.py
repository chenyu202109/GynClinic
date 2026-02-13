from datasets import load_dataset
import json

path = "epfl-llm/guidelines"
output_file_path = "./Data/meditron.jsonl"

dataset = load_dataset(path, split="train")

# 下载下来就是json文件 无需进行处理
# with open(output_file_path, "w") as f:
#     for item in dataset:
#         json.dump(item, f)
#         f.write("\n")
