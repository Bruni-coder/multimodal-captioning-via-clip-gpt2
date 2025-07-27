import os
import json
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

def extract_and_save(
    image_folder,
    caption_file,
    output_path,
    max_image_tokens=1024,
    max_samples=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)

    # 加载所有 caption
    with open(caption_file, "r", encoding="utf-8") as f:
        caption_data = json.load(f)["annotations"]

    image_caption_dict = {}
    for item in caption_data:
        sid = f"{item['image_id']:012d}.jpg"
        image_caption_dict.setdefault(sid, []).append(item["caption"])

    image_items = list(image_caption_dict.items())
    if max_samples:
        image_items = image_items[:max_samples]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for image_filename, captions in tqdm(image_items, desc="Process images"):
            image_path = os.path.join(image_folder, image_filename)
            if not os.path.exists(image_path):
                tqdm.write(f" Missing image: {image_filename}")
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                tqdm.write(f"️ Unable to read {image_filename}: {e}")
                continue

            for caption in captions:
                try:
                    inputs = processor(text=caption, images=image, return_tensors="pt", padding="max_length", truncation=True)
                    pixel_values = inputs["pixel_values"].to(device)
                    with torch.no_grad():
                        vision_outputs = model.vision_model(pixel_values=pixel_values)
                    img_embeds = vision_outputs.pooler_output.squeeze(0).cpu().tolist()
                except Exception as e:
                    tqdm.write(f"️Error processing caption/dialog for {image_filename}: {e}")
                    continue

                img_tokens = img_embeds[:max_image_tokens]

                text_tokens = inputs["input_ids"].squeeze(0).tolist()

                record = {
                    "image_id": image_filename,
                    "caption": caption,
                    "image_tokens": img_tokens,
                    "text_tokens": text_tokens
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f" Finished writing to {output_path}")

if __name__ == "__main__":
    extract_and_save(
        image_folder="data/images/train2017",
        caption_file="data/raw/captions_train2017.json",
        output_path="data/processed/aligned_tokens_full.jsonl",
        max_image_tokens=128,
        max_samples=None
    )
