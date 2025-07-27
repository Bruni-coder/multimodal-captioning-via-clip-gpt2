import json
import os
from tqdm import tqdm

def extract_image_caption_pairs(caption_file, image_dir, output_path, max_samples=None):
    """
    Extract unique image-caption pairs from COCO-style JSON annotations, where image files exist.
    """
    with open(caption_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}

    pairs = []
    seen_image_ids = set()

    for ann in tqdm(coco["annotations"], desc="Selecting image-caption pairs"):
        image_id = ann["image_id"]

        if image_id in seen_image_ids:
            continue

        filename = id_to_filename.get(image_id)
        if not filename:
            continue

        image_path = os.path.join(image_dir, filename)
        if not os.path.exists(image_path):
            continue

        caption = ann.get("caption", "").strip()
        if not caption:
            continue

        pairs.append({
            "image_id": image_id,
            "image_path": image_path,
            "caption": caption
        })
        seen_image_ids.add(image_id)

        if max_samples and len(pairs) >= max_samples:
            break

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_file:
        for item in pairs:
            out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f" Saved {len(pairs)} image-caption pairs to: {output_path}")



if __name__ == "__main__":
    extract_image_caption_pairs(
        caption_file="data/raw/captions_train2017.json",
        image_dir="data/images/train2017",
        output_path="data/processed/coco_pairs_full.jsonl",  # 支持 full 或部分数据
        max_samples=None  # 设置为整数如 100 只提取部分
    )
