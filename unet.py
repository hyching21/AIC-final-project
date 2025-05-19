import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import staintools
import argparse

IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "dataset"

class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        if args.rgb:
            self.enc1 = CBR(3, 64)      # rbg img
        else:
            self.enc1 = CBR(1, 64)      # gray img
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = CBR(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)

        self.conv_out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return torch.sigmoid(self.conv_out(d1))

def reisze_img(img, target_size=(IMAGE_SIZE, IMAGE_SIZE), is_mask=False):
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
    def __init__(self, img_paths, mask_paths, args, target_img_path=None, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.macenko = (target_img_path is not None)
        self.args = args

        # stain normalizer
        if self.macenko:
            # self.transform = transform
            self.normalizer = staintools.StainNormalizer(method='macenko')

            target = cv2.imread(target_img_path)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            target = np.clip(target, 0, 255).astype(np.uint8)
            target = staintools.LuminosityStandardizer.standardize(target)
            self.normalizer.fit(target)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # read img as rgb
        if self.macenko:
            img = cv2.imread(self.img_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = staintools.LuminosityStandardizer.standardize(img)
            img = self.normalizer.transform(img)
            if self.args.resize:
                img = reisze_img(img, is_mask=False)
            else:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = transforms.ToTensor()(img)

            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if self.args.resize:
                mask = reisze_img(mask, is_mask=True)
            else:
                mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
            mask = (mask > 0).astype(np.float32)
            mask = torch.tensor(mask).unsqueeze(0)  # (1, H, W) to fit (channel, H, W)

        # read img as gray scale
        else:
            img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
            if self.args.resize:
                img = reisze_img(img, is_mask=False)
            else:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = transforms.ToTensor()(img)
            
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
            if self.args.resize:
                mask = reisze_img(mask, is_mask=True)
            else:
                mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
            mask = (mask > 0).astype(np.float32)
            mask = torch.tensor(mask).unsqueeze(0)  # (1, H, W) to fit (channel, H, W)

        return img, mask

def dice_score(pred, target, threshold=0.5):
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

def load_data(args):
    target_img_path = "dataset/images/train/train_6.bmp"

    train_imgs = sorted(glob(os.path.join(DATA_DIR, "images/train/*.bmp")))
    train_masks = [p.replace("images", "masks").replace(".bmp", "_anno.bmp") for p in train_imgs]
    if args.rgb:
        train_set = GlandDataset(train_imgs, train_masks, args, target_img_path)
    else:
        train_set = GlandDataset(train_imgs, train_masks, args)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    test_imgs = sorted(glob(os.path.join(DATA_DIR, "images/test/*.bmp")))
    test_masks = [p.replace("images", "masks").replace(".bmp", "_anno.bmp") for p in test_imgs]
    if args.rgb:
        test_set = GlandDataset(test_imgs, test_masks, args, target_img_path)
    else:
        test_set = GlandDataset(test_imgs, test_masks, args)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, test_loader, test_imgs

def train(model, train_loader, args):
    criterion = nn.BCELoss()    # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    with open("result.txt", 'w', encoding='utf-8') as f:
        for epoch in tqdm(range(EPOCHS)):
            model.train()
            epoch_loss = 0.0
            # for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            for images, masks in train_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}")
            f.write(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}\n")

    torch.save(model.state_dict(), "model/UNet.pth")
    return model

def test(test_loader, test_imgs, model, args):
    if args.test != "none":
        model.load_state_dict(torch.load(f"model/{args.test}", map_location=DEVICE))
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)

            total_dice += dice.item()
            total_iou += iou.item()

            # visualize
            save_dir = "results"
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.basename(test_imgs[i])
            ori_img = cv2.imread(test_imgs[i])  # BGR
            ori_img = cv2.resize(ori_img, (IMAGE_SIZE, IMAGE_SIZE))
            pred = outputs[0].cpu().numpy()[0]
            pred_mask = (pred > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay = ori_img.copy()
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(save_dir, filename.replace(".bmp", f"_{args.output_name}.bmp")), overlay)

    print(f"\nTest Set Dice Score: {total_dice / len(test_loader):.4f}")
    print(f"Test Set IoU Score: {total_iou / len(test_loader):.4f}")
    with open("result.txt", 'a', encoding='utf-8') as f:
        f.write(f"\nTest Set Dice Score: {total_dice / len(test_loader):.4f}")
        f.write(f"\nTest Set IoU Score: {total_iou / len(test_loader):.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="UNet!")
    parser.add_argument('--ep', type=int, default=30)
    parser.add_argument('--output_name', type=str, default='pred')
    parser.add_argument('--rgb', action='store_true')       # use rgb img to train
    parser.add_argument('--resize', action='store_true')    # resize + padding
    parser.add_argument('--test', type=str, default="none")      # test only
    return parser.parse_args()

def main():
    args = parse_args()
    global EPOCHS
    EPOCHS = args.ep

    model = UNet(args).to(DEVICE)
    train_loader, test_loader, test_imgs = load_data(args)

    if args.test == "none":
        model = train(model, train_loader, args)

    test(test_loader, test_imgs, model, args)

if __name__ == "__main__":
    main()
