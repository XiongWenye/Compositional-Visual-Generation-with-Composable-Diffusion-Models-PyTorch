import argparse
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)

RELATION_MAP = {
    0: "left",
    1: "right",
    2: "front",
    3: "behind",
    4: "below",
    5: "above",
}


def get_args():
    options = model_and_diffusion_defaults()
    
    options["dataset"] = "clevr_rel"
    options["use_fp16"] = th.cuda.is_available()
    options["timestep_respacing"] = "100"
    options["num_classes"] = "4,3,9,3,3,7"

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, options)

    parser.add_argument("--ckpt_path", default="./logs_clevr_rel_128/ema_0.9999_740000.pt")
    parser.add_argument("--data_path", default="./dataset/clevr_generation_1_relations.npz")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_t_samples", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=7.5)

    return parser.parse_args()


class InverseCLEVRRelDataset(Dataset):
    def __init__(self, data_path):
        print(f"Loading dataset from {data_path}...")
        data = np.load(data_path)
        self.labels = data["labels"]
        self.ims = data["ims"]
        print(f"ims shape: {self.ims.shape}, labels shape: {self.labels.shape}")

    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, index):
        img = self.ims[index].astype(np.float32) / 127.5 - 1.0
        img = np.transpose(img, [2, 0, 1])  # [C,H,W]
        label = self.labels[index]
        if label.ndim == 2:
            label = label[0]
        return th.from_numpy(img), th.from_numpy(label).long()


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def evaluate(model, diffusion, dataloader, device, num_t_samples: int, guidance_scale: float):
    """
    Inverse inference using the same model_fn pattern as the official sampling script.
    
    For each image, we test all 6 relations and pick the one with lowest denoising loss.
    The key insight: with classifier-free guidance, the model predicts:
        eps_guided = eps_uncond + scale * (eps_cond - eps_uncond)
    
    If the condition matches the image, eps_cond should be closer to the true noise,
    making the guided prediction better.
    """
    cm = np.zeros((6, 6), dtype=np.int64)
    model.eval().to(device)

    # Timesteps to evaluate - use middle range where signal is informative
    t_grid = np.linspace(
        int(0.2 * diffusion.num_timesteps),
        int(0.8 * diffusion.num_timesteps) - 1,
        num_t_samples,
        dtype=int,
    )
    print(f"Evaluating on spaced timesteps: {t_grid}")
    print(f"Using guidance scale: {guidance_scale}")

    weights = th.tensor([guidance_scale]).reshape(-1, 1, 1, 1).to(device)

    pbar = tqdm(dataloader, desc="Classifying")
    for imgs, gt_labels in pbar:
        imgs = imgs.to(device)       # [B, 3, H, W]
        gt_labels = gt_labels.to(device)  # [B, 11]
        B = imgs.shape[0]

        # For each relation candidate, compute the guided denoising loss
        total_loss = th.zeros((B, 6), device=device)

        for rel_idx in range(6):
            # Create labels for this relation candidate
            # Format: conditional label + unconditional (zeros with mask=False)
            cond_labels = gt_labels.clone()
            cond_labels[:, -1] = rel_idx  # Set relation
            
            # Stack conditional + unconditional for each image in batch
            # Shape: [B, 2, 11] -> we need [B*2, 11] for batch processing
            uncond_labels = th.zeros_like(cond_labels)
            
            for t_val in t_grid:
                t = th.full((B,), int(t_val), device=device, dtype=th.long)
                
                # Same noise for fair comparison across relations
                noise = th.randn_like(imgs)
                
                # Create noisy images
                x_t = diffusion.q_sample(imgs, t, noise=noise)
                
                # Process each image with conditional + unconditional
                # Following the official model_fn pattern
                with th.no_grad():
                    # Replicate x_t for conditional and unconditional
                    x_t_double = th.cat([x_t, x_t], dim=0)  # [2B, 3, H, W]
                    t_double = th.cat([t, t], dim=0)  # [2B]
                    
                    # Labels: first B are conditional, next B are unconditional
                    labels_double = th.cat([cond_labels, uncond_labels], dim=0)  # [2B, 11]
                    masks_double = th.cat([
                        th.ones(B, dtype=th.bool, device=device),
                        th.zeros(B, dtype=th.bool, device=device)
                    ], dim=0)  # [2B]
                    
                    # Get model output
                    model_out = model(x_t_double, t_double, y=labels_double, masks=masks_double)
                    eps_out, _ = model_out[:, :3], model_out[:, 3:]
                    
                    # Split into conditional and unconditional
                    eps_cond = eps_out[:B]      # [B, 3, H, W]
                    eps_uncond = eps_out[B:]    # [B, 3, H, W]
                    
                    # Apply classifier-free guidance
                    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                    
                    # Compute loss: how well does guided prediction match true noise?
                    loss = mean_flat((eps_guided - noise) ** 2)
                    total_loss[:, rel_idx] += loss

        # Pick relation with lowest loss (best denoising)
        preds = total_loss.argmin(dim=1)
        gt_rels = gt_labels[:, -1]

        for i in range(B):
            gt = int(gt_rels[i].item())
            pred = int(preds[i].item())
            if 0 <= gt < 6:
                cm[gt, pred] += 1

    return cm


def print_metrics(cm):
    total = cm.sum()
    acc = (np.trace(cm) / total) if total > 0 else 0.0
    print(f"\nFinal Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    for idx, name in RELATION_MAP.items():
        row_sum = cm[idx, :].sum()
        per_acc = (cm[idx, idx] / row_sum) if row_sum > 0 else 0.0
        print(f"  {name:<6}: {cm[idx, idx]}/{row_sum} ({per_acc:.4f})")


def safe_load_state_dict(path: str):
    try:
        return th.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return th.load(path, map_location="cpu")


if __name__ == "__main__":
    args = get_args()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    options = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**options)

    model.eval()
    if options["use_fp16"]:
        model.convert_to_fp16()
    model.to(device)

    print(f"Loading checkpoint from {args.ckpt_path}...")
    checkpoint = safe_load_state_dict(args.ckpt_path)
    model.load_state_dict(checkpoint, strict=True)

    print("total parameters", sum(p.numel() for p in model.parameters()))
    print(f"Spaced diffusion num_timesteps: {diffusion.num_timesteps}")
    print(f"Original num_timesteps: {diffusion.original_num_steps}")

    dataset = InverseCLEVRRelDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    cm = evaluate(model, diffusion, dataloader, device, args.num_t_samples, args.guidance_scale)
    print_metrics(cm)