import os

input_path = r"D:\BruniResearch\Emu3-Mutimodal-NextToken\data\processed\input_ids_100.jsonl" # 改为你实际设置的路径

if not os.path.exists(input_path):
    print(f"文件不存在：{input_path}")
else:
    print(f"找到文件：{input_path}")
