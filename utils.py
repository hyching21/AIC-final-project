import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import staintools
import albumentations as A
from albumentations import ToTensorV2
from medpy.metric.binary import hd, assd, hd95

IMAGE_SIZE = 256

def resize_img(img, target_size=(IMAGE_SIZE, IMAGE_SIZE), is_mask=False):
    # shorter edge resize to 256
    h, w = img.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    img = cv2.resize(img, (new_w, new_h), interpolation=interp)

    # padding
    delta_w = target_size[1] - new_w
    delta_h = target_size[0] - new_h
    top, bottom = delta_h // 2, delta_h - delta_h // 2
    left, right = delta_w // 2, delta_w - delta_w // 2

    if img.ndim == 3:
        color = [0, 0, 0]
    else:
        color = 0
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img


class GlandDataset(Dataset):
    def __init__(self, img_paths, mask_paths, args, target_img_path=None, aug=True):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.macenko = (target_img_path is not None)
        self.args = args
        self.aug = aug

        # stain normalizer
        if self.macenko:
            self.normalizer = staintools.StainNormalizer(method='macenko')

            target = cv2.imread(target_img_path)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            target = np.clip(target, 0, 255).astype(np.uint8)
            target = staintools.LuminosityStandardizer.standardize(target)
            self.normalizer.fit(target)

        # data augmentation pipeline
        if args.rgb:
            normalize_val = (0.5, 0.5, 0.5)
        else:
            normalize_val = (0.5,)

        if self.aug:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                # A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ElasticTransform(p=0.2),
                A.Normalize(mean=normalize_val, std=normalize_val),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=normalize_val, std=normalize_val),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # read img & mask
        img = cv2.imread(self.img_paths[idx])
        img_trans = cv2.imread(self.img_paths[idx])     # only do augmentation, preserve original color
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if self.args.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.macenko:
                img = staintools.LuminosityStandardizer.standardize(img)
                img = self.normalizer.transform(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=2)   # (H, W) to (H, W, C)

        # resize
        if self.args.resize:
            img = resize_img(img, is_mask=False)
            img_trans = resize_img(img_trans, is_mask=False)
            mask = resize_img(mask, is_mask=True)
        else:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img_trans = cv2.resize(img_trans, (IMAGE_SIZE, IMAGE_SIZE))
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))

        # data augmentation
        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = (augmented['mask'] > 0).float().unsqueeze(0)
        return img, mask, img_trans


def dice_score(pred, target, threshold=0.5):
    # equivalent to F1 score
    pred = (pred > threshold).float()
    smooth = 1e-6
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    smooth = 1e-6
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def hd_score(pred, target, voxelspacing=None, threshold=0.5):
    # voxelspacing: spacing of voxels (tuple or None)
    pred = (pred > threshold).float()
    
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Ensure binary masks
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    return hd(pred, target, voxelspacing=voxelspacing), hd95(pred, target, voxelspacing=voxelspacing)

def assd_score(pred, target, voxelspacing=None, threshold=0.5):
    pred = (pred > threshold).float()
    
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Ensure binary masks
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    return assd(pred, target, voxelspacing=voxelspacing)