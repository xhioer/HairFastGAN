from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as tvF
from PIL import Image, ImageFilter
from tqdm.auto import tqdm

from hair_swap import HairFast, get_parser

IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def parse_ref(text):
    parts = text.split(':')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('ref must be name:shape_path:color_path, e.g. h1:input/7.png:input/8.png')
    name, shape, color = parts
    if not name:
        raise argparse.ArgumentTypeError('ref name cannot be empty')
    return name, Path(shape), Path(color)


def resolve_ref(path: Path, input_dir: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    cand = input_dir / path
    return cand


def iter_prcc_train(train_dir: Path):
    for pid_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        for img_path in sorted(pid_dir.iterdir()):
            if img_path.suffix.lower() in IMG_EXTS:
                yield pid_dir.name, img_path


def mask_path_for(parse_dir: Path, pid: str, img_path: Path) -> Path:
    return parse_dir / pid / (img_path.stem + '.png')


def square_bbox_from_mask(mask: np.ndarray, pad_ratio: float, image_size):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    w_img, h_img = image_size
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x2 - x1, y2 - y1
    pad = int(max(bw, bh) * pad_ratio)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = int(max(bw, bh) + 2 * pad)

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = x1 + side
    y2 = y1 + side

    # Shift the square back inside the image where possible.
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w_img:
        x1 -= x2 - w_img
        x2 = w_img
    if y2 > h_img:
        y1 -= y2 - h_img
        y2 = h_img
    x1, y1 = max(0, x1), max(0, y1)

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    image = image.detach().cpu().clamp(0, 1)
    return tvF.to_pil_image(image)


def composite_hair(raw: Image.Image, generated_crop_1024: Image.Image, hair_mask: Image.Image, bbox, feather: float):
    x1, y1, x2, y2 = bbox
    crop_w, crop_h = x2 - x1, y2 - y1
    generated_crop = generated_crop_1024.resize((crop_w, crop_h), Image.BICUBIC)

    mask_crop = hair_mask.crop(bbox).resize((crop_w, crop_h), Image.NEAREST)
    if feather > 0:
        mask_crop = mask_crop.filter(ImageFilter.GaussianBlur(radius=feather))

    raw_crop = raw.crop(bbox)
    pasted_crop = Image.composite(generated_crop, raw_crop, mask_crop)
    out = raw.copy()
    out.paste(pasted_crop, bbox[:2])
    return out


def process_one(hair_fast, img_path: Path, parse_path: Path, refs, output_dir: Path, input_dir: Path,
                hair_label: int, face_label: int, pad_ratio: float, feather: float, overwrite: bool):
    raw = Image.open(img_path).convert('RGB')
    parse = Image.open(parse_path)
    parse_np = np.array(parse)
    if parse_np.shape[:2] != (raw.height, raw.width):
        parse = parse.resize(raw.size, Image.NEAREST)
        parse_np = np.array(parse)

    hair_np = parse_np == hair_label
    head_np = hair_np | (parse_np == face_label)
    bbox = square_bbox_from_mask(head_np, pad_ratio=pad_ratio, image_size=raw.size)
    if bbox is None:
        return 'skip_no_head'

    hair_mask = Image.fromarray((hair_np.astype(np.uint8) * 255), mode='L')
    face_crop_1024 = raw.crop(bbox).resize((1024, 1024), Image.BICUBIC)

    output_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for name, shape_path, color_path in refs:
        out_path = output_dir / f'{name}_{img_path.name}'
        if out_path.exists() and not overwrite:
            continue
        shape_path = resolve_ref(shape_path, input_dir)
        color_path = resolve_ref(color_path, input_dir)
        result = hair_fast.swap(face_crop_1024, shape_path, color_path)
        generated = tensor_to_pil(result)
        out = composite_hair(raw, generated, hair_mask, bbox, feather=feather)
        out.save(out_path, quality=95)
        made += 1
    return f'ok_{made}'


def main():
    parser = argparse.ArgumentParser(description='Generate PRCC hair-augmented images with HairFastGAN and SCHP masks.')
    parser.add_argument('--rgb-root', type=Path, default=Path('/home/datasets/PRCC/prcc/rgb'))
    parser.add_argument('--train-dir', type=Path, default=None)
    parser.add_argument('--parse-dir', type=Path, default=None)
    parser.add_argument('--hair-dir', type=Path, default=None)
    parser.add_argument('--input-dir', type=Path, default=Path('input'))
    parser.add_argument('--ref', action='append', type=parse_ref,
                        help='Reference triplet name:shape_path:color_path. Can be repeated.')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--pid', type=str, default=None)
    parser.add_argument('--image', type=Path, default=None, help='Process one specific PRCC train image path.')
    parser.add_argument('--hair-label', type=int, default=2)
    parser.add_argument('--face-label', type=int, default=13)
    parser.add_argument('--pad-ratio', type=float, default=0.65)
    parser.add_argument('--feather', type=float, default=1.5)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    train_dir = args.train_dir or args.rgb_root / 'train'
    parse_dir = args.parse_dir or args.rgb_root / 'processed'
    hair_dir = args.hair_dir or args.rgb_root / 'hair'
    refs = args.ref or [parse_ref('h1:7.png:8.png')]

    model_args = get_parser().parse_args(['--device', args.device])
    hair_fast = HairFast(model_args)

    if args.image is not None:
        img_path = args.image
        samples = [(img_path.parent.name, img_path)]
    else:
        samples = list(iter_prcc_train(train_dir))
    if args.pid is not None:
        samples = [(pid, path) for pid, path in samples if pid == args.pid]
    if args.start:
        samples = samples[args.start:]
    if args.limit:
        samples = samples[:args.limit]

    stats = {}
    for pid, img_path in tqdm(samples, desc='PRCC hair'):
        parse_path = mask_path_for(parse_dir, pid, img_path)
        if not parse_path.exists():
            key = 'skip_no_parse'
        else:
            try:
                key = process_one(
                    hair_fast=hair_fast,
                    img_path=img_path,
                    parse_path=parse_path,
                    refs=refs,
                    output_dir=hair_dir / pid,
                    input_dir=args.input_dir,
                    hair_label=args.hair_label,
                    face_label=args.face_label,
                    pad_ratio=args.pad_ratio,
                    feather=args.feather,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                key = 'error'
                print(f'ERROR {img_path}: {type(exc).__name__}: {exc}', flush=True)
        stats[key] = stats.get(key, 0) + 1

    print('Done:', stats)


if __name__ == '__main__':
    main()
