
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
import os

from utils import dice_score, iou_score, hd_score, assd_score
from utils import GlandDataset, resize_img
from unet import load_data


IMAGE_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "dataset"


class DCAN(nn.Module):
    def __init__(self, args):
        super(DCAN, self).__init__()
        
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        in_ch = 3 if args.rgb else 1
        # Encoder
        self.enc1 = CBR(in_ch, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = CBR(512, 1024)

        # Decoder for object mask
        self.upconv4_obj = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4_obj = CBR(1024, 512)
        self.upconv3_obj = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3_obj = CBR(512, 256)
        self.upconv2_obj = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2_obj = CBR(256, 128)
        self.upconv1_obj = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1_obj = CBR(128, 64)
        self.conv_out_obj = nn.Conv2d(64, 1, kernel_size=1)

        # Decoder for contour
        self.upconv4_contour = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4_contour = CBR(1024, 512)
        self.upconv3_contour = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3_contour = CBR(512, 256)
        self.upconv2_contour = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2_contour = CBR(256, 128)
        self.upconv1_contour = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1_contour = CBR(128, 64)
        self.conv_out_contour = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # Decoder for object mask
        d4_obj = self.dec4_obj(torch.cat([self.upconv4_obj(b), e4], dim=1))
        d3_obj = self.dec3_obj(torch.cat([self.upconv3_obj(d4_obj), e3], dim=1))
        d2_obj = self.dec2_obj(torch.cat([self.upconv2_obj(d3_obj), e2], dim=1))
        d1_obj = self.dec1_obj(torch.cat([self.upconv1_obj(d2_obj), e1], dim=1))
        out_obj = torch.sigmoid(self.conv_out_obj(d1_obj))

        # Decoder for contour
        d4_contour = self.dec4_contour(torch.cat([self.upconv4_contour(b), e4], dim=1))
        d3_contour = self.dec3_contour(torch.cat([self.upconv3_contour(d4_contour), e3], dim=1))
        d2_contour = self.dec2_contour(torch.cat([self.upconv2_contour(d3_contour), e2], dim=1))
        d1_contour = self.dec1_contour(torch.cat([self.upconv1_contour(d2_contour), e1], dim=1))
        out_contour = torch.sigmoid(self.conv_out_contour(d1_contour))

        return out_obj, out_contour


def create_contour_mask(mask):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask = mask.astype(bool)

    mask = (mask * 255).astype(np.uint8)    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, contours, -1, color=255, thickness=1)
    
    contour_mask = (contour_mask > 0).astype(np.uint8)
    contour_mask = torch.from_numpy(contour_mask).unsqueeze(0).float()
    
    return contour_mask


def train(model, train_loader, args):
    criterion = nn.BCELoss()    # Binary Cross Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    with open("result_dcan.txt", 'w', encoding='utf-8') as f:
        for epoch in tqdm(range(EPOCHS)):
            model.train()
            epoch_loss = 0.0

            for images, masks, _ in train_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                contour_masks = []
                for i in range(masks.shape[0]):
                    contour = create_contour_mask(masks[i, 0])
                    contour_masks.append(contour)
                
                contour_masks = torch.stack(contour_masks).to(DEVICE).float()  # batch tensor (B, 1, H, W)

                outputs_obj, outputs_contour = model(images)

                loss_obj = criterion(outputs_obj, masks)
                loss_contour = criterion(outputs_contour, contour_masks)
                
                lambda_contour = 0.5
                loss = loss_obj + lambda_contour * loss_contour

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            f.write(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}\n")

    torch.save(model.state_dict(), "model/DCAN.pth")
    return model

def tta(model, image):
    flips = [lambda x: x,                               # ori img
             lambda x: torch.flip(x, dims=[-1]),        # flip W
             lambda x: torch.flip(x, dims=[-2]),        # flip H
             lambda x: torch.flip(x, dims=[-1, -2])]    # flip H & W
    preds = []
    for flip in flips:
        img = flip(image)
        out, _ = model(img)
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
        for i, (images, masks, draw_img) in enumerate(test_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            draw_img = draw_img.to(DEVICE)

            if args.tta:
                outputs = tta(model, images)    # use TTA
            else:
                outputs, _ = model(images)
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
            if args.visualize:
                save_dir = "results"
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.basename(test_imgs[i])
                ori_img = draw_img[0].cpu().numpy()

                pred = outputs[0].cpu().numpy()[0]
                pred_mask = (pred > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # GT contour
                if args.drawGT:
                    gt = masks[0].cpu().numpy()[0]
                    gt_mask = (gt > 0.5).astype(np.uint8)
                    contours_gt, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                overlay = ori_img.copy()
                if args.drawGT:
                    cv2.drawContours(overlay, contours_gt, -1, (0, 0, 255), 1)      # Red for GT contour
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
                cv2.imwrite(os.path.join(save_dir, filename.replace(".bmp", f"_{args.output_name}.bmp")), overlay)

    print(f"\nTest Set Dice Score: {total_dice / len(test_loader):.4f}")
    print(f"Test Set IoU Score: {total_iou / len(test_loader):.4f}")
    print(f"Test Set HD Score: {total_hd / len(test_loader):.4f}")
    print(f"Test Set HD95 Score: {total_hd95 / len(test_loader):.4f}")
    print(f"Test Set ASSD Score: {total_assd / len(test_loader):.4f}")
    with open("result_dcan.txt", 'a', encoding='utf-8') as f:
        f.write(f"\n\n-----------------------------------------\n")
        f.write(f"\nTest Set Dice Score: {total_dice / len(test_loader):.4f}")
        f.write(f"\nTest Set IoU Score: {total_iou / len(test_loader):.4f}")
        f.write(f"\nTest Set HD Score: {total_hd / len(test_loader):.4f}")
        f.write(f"\nTest Set HD95 Score: {total_hd95 / len(test_loader):.4f}")
        f.write(f"\nTest Set ASSD Score: {total_assd / len(test_loader):.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="DCAN!")
    parser.add_argument('--ep', type=int, default=30)
    parser.add_argument('--output_name', type=str, default='pred')
    parser.add_argument('--rgb', action='store_true')       # use rgb img to train
    parser.add_argument('--resize', action='store_true')    # resize + padding
    parser.add_argument('--tta', action='store_true')       # use TTA when testing
    parser.add_argument('--visualize', action='store_true')      # draw image
    parser.add_argument('--test', type=str, default="none")      # test only
    parser.add_argument('--drawGT', action='store_true')      # draw GT contour
    return parser.parse_args()


def main():
    args = parse_args()
    global EPOCHS
    EPOCHS = args.ep

    model = DCAN(args).to(DEVICE)
    train_loader, test_loader, test_imgs = load_data(args)

    if args.test == "none":
        model = train(model, train_loader, args)

    test(test_loader, test_imgs, model, args)


if __name__ == "__main__":
    main()
