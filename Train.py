import random

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.backends import cudnn
from tqdm import tqdm
from utils.metrics import calculate_metrics

from model.Net import PFCT as model
from improve_image.dataloader import get_loader
from loss_function import losses


def train(model, device, batch_size, Dice_loss, BCE_loss):
    lr = 0.0001

    num_epochs = 40
    size = 352

    image_path = "./dataset/TrainDataset/image/"
    mask_path = "./dataset/TrainDataset/mask/"

    # 各个验证集的路径
    val_paths = []
    val_data_list = ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Kvasir']
    for i in val_data_list:
        val_paths.append((f"./dataset/TestDataset/{i}/images/", f"./dataset/TestDataset/{i}/masks/"))

    # Load training data
    train_loader = get_loader(image_path, gt_root=mask_path, batchsize=batch_size, trainsize=size, shuffle=True)

    # Model initialization
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_logs = []
    val_logs = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        val_epoch_loss = 0.0
        # Adjust learning rate based on epoch
        if epoch >= 20:
            lr = 0.00005

        # Update optimizer with the new learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for images, masks in tqdm(train_loader):
            optimizer.zero_grad()

            images, masks = images.to(device), masks.to(device)
            output = model(images)

            loss = Dice_loss(output, masks) + BCE_loss(torch.sigmoid(output), masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_logs.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.10f}, lr: {lr}")

        best_val_loss = float('inf')
        # Validation phase
        model.eval()
        all_metrics_per_dataset = []  # 存储每个验证集的所有指标
        for val_image_path, val_mask_path in val_paths:
            dataset_name = val_image_path.split('/')[-3]  # 提取文件夹名
            val_loader = get_loader(val_image_path, gt_root=val_mask_path, batchsize=batch_size, trainsize=size,
                                    shuffle=False)
            val_loss = 0.0
            all_metrics = {
                'iou': 0,
                'dice_coefficient': 0,
                'precision': 0,
                'recall': 0,
            }
            with torch.no_grad():
                num_val_images = 0
                for val_images, val_masks in val_loader:

                    val_images, val_masks = val_images.to(device), val_masks.to(device)
                    val_output = model(val_images)
                    val_loss_batch = Dice_loss(val_output, val_masks) + BCE_loss(torch.sigmoid(val_output), val_masks)
                    val_loss += val_loss_batch.item()

                    preds = (val_output > 0.5).float().squeeze(1)  # 转换为(batch, H, W)
                    val_masks = val_masks.squeeze(1)  # 转换为(batch, H, W)

                    for i in range(val_images.size(0)):
                        metrics = calculate_metrics(preds[i], val_masks[i])
                        for key in all_metrics:
                            all_metrics[key] += metrics[key]
                        num_val_images += 1

            val_epoch_loss = val_loss / len(val_loader)
            val_logs.append(val_epoch_loss)

            # Compute average metrics
            for key in all_metrics:
                all_metrics[key] /= num_val_images

            all_metrics_per_dataset.append(all_metrics)

            print(f"\n{dataset_name}: Dice: {all_metrics['dice_coefficient']:.4f}, IoU: {all_metrics['iou']:.4f}, "
                  f" Accuracy: {all_metrics['accuracy']:.4f}, "
                  f" Precision: {all_metrics['precision']:.4f}, Recall: {all_metrics['recall']:.4f}, ")

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_model_path = ""
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved with val loss: {val_epoch_loss:.10f}")
            #
            # if (epoch + 1) % 1 == 0:
            #     save_path = ""
            #     torch.save(model.state_dict(), save_path)
            #     print("Model checkpoint saved")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model(3, 1).to(device)
    Dice_loss = losses.SoftDiceLoss()
    BCE_loss = nn.BCELoss()
    batch_size = 8
    train(model, device, batch_size, Dice_loss, BCE_loss)
