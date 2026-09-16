from model import RuriClapProjectionModel
from ._datasets import TranslateDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from tqdm import tqdm
from safetensors.torch import save_file
import os

load_dotenv()


def train_main(
    device: torch.device | str,
    temperature=0.07,
    batch_size=512,
    dataloader_workers=4,
    epochs=20,
):
    device = torch.device(device)

    dataset = TranslateDataset()
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers
    )

    model = RuriClapProjectionModel()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    for epoch in tqdm(range(epochs), desc="Epochs"):  # Example: train for 10 epochs
        for batch in tqdm(dataloader, desc="Batches"):
            src_ja_anc, src_ja_pos, src_en_pos, dst_en_pos, dst_en_neg = batch
            src_ja_anc = src_ja_anc.to(device, dtype=torch.float32)  # (B, 256)
            src_ja_pos = src_ja_pos.to(device, dtype=torch.float32)  # (B, 256)
            src_en_pos = src_en_pos.to(device, dtype=torch.float32)  # (B, 256)
            dst_en_pos = dst_en_pos.to(device, dtype=torch.float32)  # (B, 512)
            dst_en_neg = dst_en_neg.to(device, dtype=torch.float32)  # (B, 512)

            dst_candidates = torch.cat([dst_en_pos, dst_en_neg], dim=0)  # (2B, 512)
            # 正解インデックスは 0 ~ B-1
            labels = torch.arange(src_ja_anc.size(0), device=device)

            proj_ja_anc = model(src_ja_anc)  # (B, 512)
            proj_ja_pos = model(src_ja_pos)  # (B, 512)
            proj_en_pos = model(src_en_pos)  # (B, 512)

            proj_ja_anc = nn.functional.normalize(proj_ja_anc, dim=-1)
            proj_ja_pos = nn.functional.normalize(proj_ja_pos, dim=-1)
            proj_en_pos = nn.functional.normalize(proj_en_pos, dim=-1)

            def single_view_loss(proj_embeds: torch.Tensor):
                logits = torch.matmul(proj_embeds, dst_candidates.T) / temperature
                return nn.functional.cross_entropy(logits, labels)

            loss_ja_anc = single_view_loss(proj_ja_anc)
            loss_ja_pos = single_view_loss(proj_ja_pos)
            loss_en_pos = single_view_loss(proj_en_pos)

            loss = (loss_ja_anc + loss_ja_pos + loss_en_pos) / 3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            tqdm.write(f"Epoch [{epoch+1}/10] Loss: {loss.item():.4f}")
            tqdm.write(
                f"Epoch [{epoch+1}/10] loss_ja_anc: {loss_ja_anc.item():.4f}, loss_ja_pos: {loss_ja_pos.item():.4f}, loss_en_pos: {loss_en_pos.item():.4f}"
            )

    output_dir = ".var/output"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    save_file(model.state_dict(), os.path.join(output_dir, "model.safetensors"))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-t", "--temperature", type=float, default=0.07)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-b", "--batch-size", type=int, default=512)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    args = parser.parse_args()

    train_main(**vars(args))
