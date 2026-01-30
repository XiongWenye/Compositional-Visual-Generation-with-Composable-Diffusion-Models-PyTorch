"""
Sanity check: Generate images with the same objects but different relations
to verify the model actually learned to distinguish relations.
"""
import numpy as np
import torch as th
from PIL import Image
from torchvision.utils import make_grid, save_image

from composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
import argparse

def main():
    options = model_and_diffusion_defaults()
    options["dataset"] = "clevr_rel"
    options["use_fp16"] = th.cuda.is_available()
    options["timestep_respacing"] = "100"
    options["num_classes"] = "4,3,9,3,3,7"

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, options)
    parser.add_argument("--ckpt_path", default="./logs_clevr_rel_128/ema_0.9999_740000.pt")
    args = parser.parse_args()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    options = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if options["use_fp16"]:
        model.convert_to_fp16()
    model.to(device)
    
    checkpoint = th.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.ckpt_path}")

    # Fixed object attributes (matching the format from the official script):
    # Label format: [shape1, size1, color1, material1, pos1, shape2, size2, color2, material2, pos2, relation]
    # Example from official: [2, 0, 5, 1, 0, 2, 0, 2, 0, 1, 5]
    # Let's use: cylinder(2), small(0), purple(5), metal(1), pos(0), cylinder(2), small(0), blue(2), rubber(0), pos(1), relation
    
    RELATIONS = ["left", "right", "front", "behind", "below", "above"]
    guidance_scale = 7.5
    
    all_samples = []
    
    for rel_idx in range(6):
        print(f"\nGenerating with relation: {RELATIONS[rel_idx]} (idx={rel_idx})")
        
        # Create label tensor - shape [1, 1, 11] then process like official script
        # Using same objects as official example but varying relation
        label = th.tensor([[[2, 0, 5, 1, 0, 2, 0, 2, 0, 1, rel_idx]]]).long()
        
        # Process labels like the official script does
        labels = [x.squeeze(dim=1) for x in th.chunk(label, label.shape[1], dim=1)]  # List of [1, 11] tensors
        full_batch_size = 1 * (len(labels) + 1)  # batch_size * (num_conditions + 1 unconditional)
        masks = [True] * len(labels) + [False]
        labels = th.cat((labels + [th.zeros_like(labels[0])]), dim=0)  # [2, 11] - conditional + unconditional
        
        weights = th.tensor([guidance_scale]).reshape(-1, 1, 1, 1).to(device)
        
        model_kwargs = dict(
            y=labels.clone().detach().to(device),
            masks=th.tensor(masks, dtype=th.bool, device=device)
        )
        
        # Model function matching official implementation exactly
        def model_fn(x_t, ts, **kwargs):
            half = x_t[:1]
            combined = th.cat([half] * kwargs['y'].size(0), dim=0)
            model_out = model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = eps[:-1], eps[-1:]
            # Classifier-free guidance
            half_eps = uncond_eps + (weights * (cond_eps - uncond_eps)).sum(dim=0, keepdim=True)
            eps = th.cat([half_eps] * x_t.size(0), dim=0)
            return th.cat([eps, rest], dim=1)
        
        # Generate
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:1]  # Take only first sample
        
        all_samples.append(samples)
        print(f"Generated sample for {RELATIONS[rel_idx]}")
    
    # Save grid
    all_samples = th.cat(all_samples, dim=0)
    all_samples = ((all_samples + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu() / 255.
    grid = make_grid(all_samples, nrow=3, padding=2)
    save_image(grid, "sanity_check_relations.png")
    print("\nSaved to sanity_check_relations.png")
    print("The 6 images should show the same two objects with different spatial relations:")
    print("Row 1: left, right, front")
    print("Row 2: behind, below, above")

if __name__ == "__main__":
    main()