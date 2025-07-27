import os
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_scheduler
from torch.optim import AdamW
from dataset import MultiModalDataset
from tqdm import tqdm

# ==== 配置参数 ====
json_path = "data/processed/coco_pairs_full.jsonl"
image_root = "data/images/train2017"
aligned_token_path = "data/processed/aligned_tokens_full.jsonl"

batch_size = 8
epochs = 8
lr = 5e-5
save_path = "models/full_coco"
resume_path = os.path.join(save_path, "latest.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 加载 tokenizer 和模型 ====
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.model = model
model.resize_token_embeddings(len(tokenizer))
model = model.to(device)

# ==== 数据集与 DataLoader ====
dataset = MultiModalDataset(
    json_path=json_path,
    image_root=image_root,
    aligned_token_path=aligned_token_path,
    tokenizer=tokenizer,
)

def collate_fn(batch):
    inputs = [b["inputs_embeds"] for b in batch]
    masks = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    masks = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return dict(inputs_embeds=inputs, attention_mask=masks, labels=labels)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# ==== 优化器与调度器 ====
optimizer = AdamW(model.parameters(), lr=lr)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=len(dataloader) * epochs
)

start_epoch = 0

# ==== 支持断点恢复 ====
if os.path.exists(resume_path):
    print(f" Resuming from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    start_epoch = checkpoint["epoch"] + 1

# ==== 正式训练 ====
model.train()
for epoch in range(start_epoch, epochs):
    print(f"\n Epoch {epoch + 1} / {epochs}")
    total_loss = 0.0

    loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=False)
    for step, batch in enumerate(loop):
        inputs_embeds = batch["inputs_embeds"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f" Epoch {epoch + 1} Finished. Avg Loss: {avg_loss:.4f}")

    # 保存最新模型
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch
    }, os.path.join(save_path, "latest.pt"))

    # 保存 checkpoint（按 epoch 编号）
    model.save_pretrained(os.path.join(save_path, f"gpt2_epoch{epoch + 1}"))
    tokenizer.save_pretrained(os.path.join(save_path, f"gpt2_epoch{epoch + 1}"))
