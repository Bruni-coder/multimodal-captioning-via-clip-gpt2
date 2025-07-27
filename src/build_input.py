import json
from transformers import GPT2Tokenizer
from tqdm import tqdm

def generate_input_ids(
    input_path="data/processed/first_100_pairs.jsonl",
    output_path="data/processed/input_ids_100.jsonl",
    num_image_tokens=32,
    image_token_start=50000,
    max_length=None
):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    image_token_ids = list(range(image_token_start, image_token_start + num_image_tokens))

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    with open(output_path, "w", encoding="utf-8") as fout:
        for item in tqdm(lines, desc="Encoding text"):
            caption = item.get("caption", "")
            text_token_ids = tokenizer.encode(caption, add_special_tokens=False)

            input_ids = image_token_ids + text_token_ids

            if max_length:
                input_ids = input_ids[:max_length]

            # 允许保留原始 image_id / image_path
            out = {
                "input_ids": input_ids,
                "image_id": item.get("image_id"),
                "caption": caption
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f" 成功生成 {len(lines)} 条图文拼接 token 序列到：{output_path}")


if __name__ == "__main__":
    generate_input_ids(
        input_path="data/processed/coco_pairs_full.jsonl",
        output_path="data/processed/input_ids_coco.jsonl",
        num_image_tokens=32,
        image_token_start=50000,
        max_length=512
    )
