import torch


def calculate_metrics(predict_image, gt_image):
    tp = torch.sum((predict_image == 1) & (gt_image == 1)).item()
    tn = torch.sum((predict_image == 0) & (gt_image == 0)).item()
    fp = torch.sum((predict_image == 1) & (gt_image == 0)).item()
    fn = torch.sum((predict_image == 0) & (gt_image == 1)).item()

    iou = tp / (tp + fn + fp + 1e-7)
    dice_coefficient = 2 * tp / (2 * tp + fn + fp + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)

    return {
        'iou': iou,
        'dice_coefficient': dice_coefficient,
        'precision': precision,
        'recall': recall,
    }


def evaluate_folder(model, dataloader, device):
    all_metrics = {
        'iou': 0,
        'dice_coefficient': 0,
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'sensitivity': 0,
        'f1': 0,
        'specificity': 0
    }

    num_images = 0

    with torch.no_grad():
        for images, gt_images in dataloader:
            images = images.to(device)
            gt_images = gt_images.to(device).squeeze(1)  # 转换为(batch, H, W)

            outputs = model(images)

            preds = (outputs > 0.5).float().squeeze(1)  # 转换为(batch, H, W)

            for i in range(images.size(0)):
                metrics = calculate_metrics(preds[i], gt_images[i])

                for key in all_metrics:
                    all_metrics[key] += metrics[key]

                num_images += 1

    for key in all_metrics:
        all_metrics[key] /= num_images

    return all_metrics
