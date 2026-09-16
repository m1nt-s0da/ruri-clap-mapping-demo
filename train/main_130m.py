from model import Ruri130mClapProjectionModel
from ._datasets import Translate130mDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from tqdm import tqdm
from safetensors.torch import save_file
import os

load_dotenv()


def train_main_130m(
    device: torch.device | str,
    temperature=0.07,
    batch_size=512,
    dataloader_workers=4,
    epochs=20,
    mse_weight=0.1,
):
    device = torch.device(device)

    dataset = Translate130mDataset()
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers
    )

    model = Ruri130mClapProjectionModel()
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(dataloader), eta_min=1e-6
    )

    for epoch in tqdm(range(epochs), desc="Epochs"):  # Example: train for 10 epochs
        for batch in tqdm(dataloader, desc="Batches"):
            src_ja_anc, src_ja_pos, src_en_pos, dst_en_pos, dst_en_neg = batch
            src_ja_anc = src_ja_anc.to(device, dtype=torch.float32)  # (B, 512)
            src_ja_pos = src_ja_pos.to(device, dtype=torch.float32)  # (B, 512)
            src_en_pos = src_en_pos.to(device, dtype=torch.float32)  # (B, 512)
            dst_en_pos = dst_en_pos.to(device, dtype=torch.float32)  # (B, 512)
            dst_en_neg = dst_en_neg.to(device, dtype=torch.float32)  # (B, 512)

            dst_en_pos = nn.functional.normalize(dst_en_pos, dim=-1)
            dst_en_neg = nn.functional.normalize(dst_en_neg, dim=-1)

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
                with torch.no_grad():
                    sim_matrix = torch.matmul(proj_embeds, proj_embeds.T)
                    mask = ~torch.eye(
                        sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device
                    )
                    mean_self_sim = sim_matrix[mask].mean().item()

                    mean_correct_label_cosine_sim = (
                        torch.sum(proj_embeds * dst_en_pos, dim=-1).mean().item()
                    )

                    _, S, _ = torch.svd(proj_embeds - proj_embeds.mean(dim=0))
                    prob = S / S.sum()
                    effective_dim = torch.exp(
                        -torch.sum(prob * torch.log(prob + 1e-8))
                    ).item()

                mse_loss = torch.nn.functional.mse_loss(proj_embeds, dst_en_pos)

                logits = torch.matmul(proj_embeds, dst_candidates.T) / temperature
                return (
                    nn.functional.cross_entropy(logits, labels),
                    mse_loss,
                    mean_self_sim,
                    effective_dim,
                    mean_correct_label_cosine_sim,
                )

            (
                loss_ja_anc,
                mse_loss_ja_anc,
                mean_self_sim_ja_anc,
                effective_dim_ja_anc,
                mean_correct_label_cosine_sim_ja_anc,
            ) = single_view_loss(proj_ja_anc)
            (
                loss_ja_pos,
                mse_loss_ja_pos,
                mean_self_sim_ja_pos,
                effective_dim_ja_pos,
                mean_correct_label_cosine_sim_ja_pos,
            ) = single_view_loss(proj_ja_pos)
            (
                loss_en_pos,
                mse_loss_en_pos,
                mean_self_sim_en_pos,
                effective_dim_en_pos,
                mean_correct_label_cosine_sim_en_pos,
            ) = single_view_loss(proj_en_pos)

            loss = (loss_ja_anc * 20 + loss_ja_pos + loss_en_pos) / 22 + mse_weight * (
                mse_loss_ja_anc * 20 + mse_loss_ja_pos + mse_loss_en_pos
            ) / 22

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            tqdm.write(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")
            tqdm.write(
                f"Epoch [{epoch+1}/{epochs}] loss_ja_anc: {loss_ja_anc.item():.4f}, loss_ja_pos: {loss_ja_pos.item():.4f}, loss_en_pos: {loss_en_pos.item():.4f}"
            )
            tqdm.write(
                f"Epoch [{epoch+1}/{epochs}] mse_loss_ja_anc: {mse_loss_ja_anc:.4f}, mse_loss_ja_pos: {mse_loss_ja_pos:.4f}, mse_loss_en_pos: {mse_loss_en_pos:.4f}"
            )
            tqdm.write(
                f"Epoch [{epoch+1}/{epochs}] mean_self_sim_ja_anc: {mean_self_sim_ja_anc:.4f}, mean_self_sim_ja_pos: {mean_self_sim_ja_pos:.4f}, mean_self_sim_en_pos: {mean_self_sim_en_pos:.4f}"
            )
            tqdm.write(
                f"Epoch [{epoch+1}/{epochs}] effective_dim_ja_anc: {effective_dim_ja_anc:.4f}, effective_dim_ja_pos: {effective_dim_ja_pos:.4f}, effective_dim_en_pos: {effective_dim_en_pos:.4f}"
            )
            tqdm.write(
                f"Epoch [{epoch+1}/{epochs}] mean_correct_label_cosine_sim_ja_anc: {mean_correct_label_cosine_sim_ja_anc:.4f}, mean_correct_label_cosine_sim_ja_pos: {mean_correct_label_cosine_sim_ja_pos:.4f}, mean_correct_label_cosine_sim_en_pos: {mean_correct_label_cosine_sim_en_pos:.4f}"
            )

    output_dir = ".var/output"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model_130m.pth"))
    save_file(model.state_dict(), os.path.join(output_dir, "model_130m.safetensors"))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-t", "--temperature", type=float, default=0.07)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("-b", "--batch-size", type=int, default=512)
    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("--dataloader-workers", type=int, default=4)
    args = parser.parse_args()

    train_main_130m(**vars(args))
