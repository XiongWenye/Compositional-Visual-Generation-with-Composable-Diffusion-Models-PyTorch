import numpy as np

import argparse
import torch as th

from composable_diffusion.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict
)

from PIL import Image
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.utils import save_image

has_cuda = th.cuda.is_available()
device = th.device('cpu' if not has_cuda else 'cuda')

options = model_and_diffusion_defaults()
options['dataset'] = 'clevr_rel'
options['use_fp16'] = has_cuda
options['timestep_respacing'] = '100'  # use 100 diffusion steps for fast sampling
options['num_classes'] = '4,3,9,3,3,7'

parser = argparse.ArgumentParser()
add_dict_to_argparser(parser, options)

parser.add_argument('--ckpt_path', required=True)
parser.add_argument('--weights', type=float, nargs="+", default=7.5)
parser.add_argument('--output_dir', type=str, default='outputs/clevr_rel')
parser.add_argument('--num_labels', type=int, default=1000)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--num_samples', type=int, default=4)

args = parser.parse_args()
ckpt_path = args.ckpt_path
del args.ckpt_path

options = args_to_dict(args, model_and_diffusion_defaults().keys())
model, diffusion = create_model_and_diffusion(**options)

model.eval()
if options['use_fp16']:
    model.convert_to_fp16()
model.to(device)

print(f'loading from {ckpt_path}')

checkpoint = th.load(ckpt_path, map_location='cpu')
model.load_state_dict(checkpoint)

print('total base parameters', sum(x.numel() for x in model.parameters()))

def show_images(batch: th.Tensor, file_name: str = 'result.png'):
    """ Display a batch of images inline. """
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    Image.fromarray(reshaped.numpy()).save(file_name)


class CLEVRRelDataset(Dataset):
    def __init__(
        self,
        resolution,
        random_crop=False,
        random_flip=False,
    ):
        self.resolution = resolution
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.data_path = './dataset/clevr_generation_1_relations.npz'

        data = np.load(self.data_path)
        self.labels = data['labels']

        self.description = {
            "left": ["to the left of"],
            "right": ["to the right of"],
            "behind": ["behind"],
            "front": ["in front of"],
            "above": ["above"],
            "below": ["below"]
        }

        self.shapes_to_idx = {"cube": 0, "sphere": 1, "cylinder": 2, 'none': 3}
        self.colors_to_idx = {"gray": 0, "red": 1, "blue": 2, "green": 3, "brown": 4, "purple": 5, "cyan": 6,
                              "yellow": 7, 'none': 8}
        self.materials_to_idx = {"rubber": 0, "metal": 1, 'none': 2}
        self.sizes_to_idx = {"small": 0, "large": 1, 'none': 2}
        self.relations_to_idx = {"left": 0, "right": 1, "front": 2, "behind": 3, 'below': 4, 'above': 5, 'none': 6}

        self.idx_to_colors = list(self.colors_to_idx.keys())
        self.idx_to_shapes = list(self.shapes_to_idx.keys())
        self.idx_to_materials = list(self.materials_to_idx.keys())
        self.idx_to_sizes = list(self.sizes_to_idx.keys())
        self.idx_to_relations = list(self.relations_to_idx.keys())

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        label = self.labels[index]
        return label, self.convert_caption(label)

    def convert_caption(self, label):
        paragraphs = []
        for j in range(label.shape[0]):
            text_label = []
            for k in range(2):
                shape, size, color, material, pos = label[j, k * 5:k * 5 + 5]
                obj = ' '.join([self.idx_to_sizes[size], self.idx_to_colors[color],
                                self.idx_to_materials[material], self.idx_to_shapes[shape]])
                text_label.append(obj.strip())

            relation = self.idx_to_relations[label[j, -1]]
            # single object
            if relation == 'none':
                paragraphs.append(text_label[0])
            else:
                paragraphs.append(f'{text_label[0]} {self.description[relation][0]} {text_label[1]}')
        return ' and '.join(paragraphs)


dataset = CLEVRRelDataset(128, random_crop=False, random_flip=False)

batch_size = 1

# Tune this parameter to control the sharpness of 256x256 images.
# A value of 1.0 is sharper, but sometimes results in grainy artifacts.
# upsample_temp = 0.997
upsample_temp = 0.980

weights = args.weights
weights = th.tensor(weights).reshape(-1, 1, 1, 1).to(device)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

if args.num_labels < 0:
    end_index = len(dataset)
else:
    end_index = min(args.start_index + args.num_labels, len(dataset))

def model_fn(x_t, ts, **kwargs):
    half = x_t[:1]
    combined = th.cat([half] * kwargs['y'].size(0), dim=0)
    model_out = model(combined, ts, **kwargs)
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = eps[:-1], eps[-1:]
    # assume weights are equal to guidance scale
    half_eps = uncond_eps + (weights * (cond_eps - uncond_eps)).sum(dim=0, keepdim=True)
    eps = th.cat([half_eps] * x_t.size(0), dim=0)
    return th.cat([eps, rest], dim=1)

for index in range(args.start_index, end_index):
    label, caption = dataset[index]  # <-- label comes from clevr_generation_1_relations.npz
    label_tensor = th.from_numpy(label).long().unsqueeze(0)

    labels = [x.squeeze(dim=1) for x in th.chunk(label_tensor, label_tensor.shape[1], dim=1)]
    full_batch_size = batch_size * (len(labels) + 1)
    masks = [True] * len(labels) + [False]
    labels = th.cat((labels + [th.zeros_like(labels[0])]), dim=0)

    model_kwargs = dict(
        y=labels.clone().detach().to(device),
        masks=th.tensor(masks, dtype=th.bool, device=device)
    )

    for sample_idx in range(args.num_samples):
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]

        image = (samples[0] + 1) / 2
        image = image.clamp(0, 1)

        image_path = output_dir / f'sample_{index:06d}_{sample_idx:02d}.png'
        metadata_path = output_dir / f'sample_{index:06d}_{sample_idx:02d}.txt'

        save_image(image, image_path)

        with metadata_path.open('w', encoding='utf-8') as f:
            f.write(f'index: {index}\n')
            f.write(f'caption: {caption}\n')
            f.write(f'label: {label.tolist()}\n')