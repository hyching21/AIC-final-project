import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from utils import dice_score, iou_score, hd_score, assd_score
from utils import GlandDataset, resize_img

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


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        smooth = 1e-6
        intersection = (pred * target).sum()
        dice = (2.*intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

def train(model, train_loader, args):
    bce_loss = nn.BCELoss()    # Binary Cross Entropy Loss
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    with open(f"result_{args.output_name}.txt", 'w', encoding='utf-8') as f:
        for epoch in tqdm(range(EPOCHS)):
            model.train()
            epoch_loss = 0.0
            # for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            for images, masks in train_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)
                bce = bce_loss(outputs, masks)
                dice = dice_loss(outputs, masks)
                loss = 0.7 * dice + 0.3 * bce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            f.write(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}\n")

    torch.save(model.state_dict(), f"model/UNet_{args.output_name}.pth")
    return model

'''
TTA: Test-Time Augmentation
在testing 的時候產生一些augmented image
把這些圖片都拿去做預測
最終的預測 = 這些augmented image 的預測的平均
'''
def tta(model, image):
    flips = [lambda x: x,                               # ori img
             lambda x: torch.flip(x, dims=[-1]),        # flip W
             lambda x: torch.flip(x, dims=[-2]),        # flip H
             lambda x: torch.flip(x, dims=[-1, -2])]    # flip H & W
    preds = []
    for flip in flips:
        img = flip(image)
        out = model(img)
        out = flip(out)
        preds.append(out)
    return torch.stack(preds).mean(dim=0)   # pred = avg of all flips

def test(test_loader, test_imgs, model, args):
    if args.test != "none":
        model.load_state_dict(torch.load(f"model/{args.test}", map_location=DEVICE))
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    total_hd = 0.0
    total_hd95 = 0.0
    total_assd = 0.0
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            if args.tta:
                outputs = tta(model, images)        # use TTA
            else:
                outputs = model(images)
            dice = dice_score(outputs, masks)
            iou = iou_score(outputs, masks)
            hd, hd95 = hd_score(outputs, masks)
            assd = assd_score(outputs, masks)

            total_dice += dice.item()
            total_iou += iou.item()
            total_hd += hd
            total_hd95 += hd95
            total_assd += assd

            # visualize
            save_dir = "results"
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.basename(test_imgs[i])
            ori_img = cv2.imread(test_imgs[i])  # BGR
            if args.resize:
                ori_img = resize_img(ori_img, is_mask=False)
            else:
                ori_img = cv2.resize(ori_img, (IMAGE_SIZE, IMAGE_SIZE))
            pred = outputs[0].cpu().numpy()[0]
            pred_mask = (pred > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay = ori_img.copy()
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
            cv2.imwrite(os.path.join(save_dir, filename.replace(".bmp", f"_{args.output_name}.bmp")), overlay)

    print(f"\nTest Set Dice Score: {total_dice / len(test_loader):.4f}")
    print(f"Test Set IoU Score: {total_iou / len(test_loader):.4f}")
    print(f"Test Set HD Score: {total_hd / len(test_loader):.4f}")
    print(f"Test Set HD95 Score: {total_hd95 / len(test_loader):.4f}")
    print(f"Test Set ASSD Score: {total_assd / len(test_loader):.4f}")
    with open(f"result_{args.output_name}.txt", 'a', encoding='utf-8') as f:
        f.write(f"\n\n-----------------------------------------\n")
        f.write(f"\nTest Set Dice Score: {total_dice / len(test_loader):.4f}")
        f.write(f"\nTest Set IoU Score: {total_iou / len(test_loader):.4f}")
        f.write(f"\nTest Set HD Score: {total_hd / len(test_loader):.4f}")
        f.write(f"\nTest Set HD95 Score: {total_hd95 / len(test_loader):.4f}")
        f.write(f"\nTest Set ASSD Score: {total_assd / len(test_loader):.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="UNet!")
    parser.add_argument('--ep', type=int, default=30)
    parser.add_argument('--output_name', type=str, default='pred')
    parser.add_argument('--rgb', action='store_true')       # use rgb img to train
    parser.add_argument('--resize', action='store_true')    # resize + padding
    parser.add_argument('--tta', action='store_true')       # use TTA when testing
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
