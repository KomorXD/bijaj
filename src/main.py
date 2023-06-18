import os
import torch
import segmentation_models_pytorch as smp
import seg_dataset
import augments_and_preproc as aap
import utilities
from torch.utils.data import DataLoader
from segmentation_models_pytorch import utils
from pathlib import Path
from multiprocessing import freeze_support


TRAIN_IMG_DIR = Path("../dataset/valid").resolve()
TRAIN_MASKS_DIR = Path("../dataset/valid_masks").resolve()

VALID_IMG_DIR = Path("../dataset/valid").resolve()
VALID_MASKS_DIR = Path("../dataset/valid_masks").resolve()

TEST_IMG_DIR = Path("../dataset/test").resolve()
TEST_MASKS_DIR = Path("../dataset/test_masks").resolve()

ENCODER = 'se_resnext50_32x4d'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['Animal', 'NonMaskingBackground', "MaskingBackground", "NonMaskingForegroundAttention"]
ACTIVATION = 'softmax2d'
DEVICE = 'cuda'
EPOCHES = 5

MODEL_USED = "Unet"
LOSS_USED = "DiceLoss"
OUTPUT_DIR = Path(f"../output/{MODEL_USED}/{ACTIVATION}/{LOSS_USED}").resolve()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if not Path(OUTPUT_DIR).exists():
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    utilities.fix_ssl_bug_thingy()

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,

    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = seg_dataset.Dataset(
        TRAIN_IMG_DIR,
        TRAIN_MASKS_DIR,
        augmentation=aap.get_training_augmentation(),
        preprocessing=aap.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = seg_dataset.Dataset(
        VALID_IMG_DIR,
        VALID_MASKS_DIR,
        augmentation=aap.get_validation_augmentation(),
        preprocessing=aap.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001),
    ])

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    max_score = 0
    epoch_points = list(range(0, EPOCHES))
    iou_scores_rel = {
        "train": [],
        "valid": []
    }
    loss_rel = {
        "train": [],
        "valid": []
    }

    for i in range(0, EPOCHES):
        print(f'\nEpoch: {i}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        iou_scores_rel["train"].append(train_logs['iou_score'])
        loss_rel["train"].append(train_logs['dice_loss'])

        iou_scores_rel["valid"].append(valid_logs['iou_score'])
        loss_rel["valid"].append(valid_logs['dice_loss'])

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, '../best_model.pth')
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

    best_model = torch.load('../best_model.pth')

    test_dataset = seg_dataset.Dataset(
        TEST_IMG_DIR,
        TEST_MASKS_DIR,
        augmentation=aap.get_validation_augmentation(),
        preprocessing=aap.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    test_dataloader = DataLoader(test_dataset)

    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    test_epoch.run(test_dataloader)

    test_dataset_vis = seg_dataset.Dataset(
        TEST_IMG_DIR,
        TEST_MASKS_DIR,
        augmentation=aap.get_validation_augmentation(),
        classes=CLASSES,
    )

    utilities.plot_and_save_training_data(epoch_points, iou_scores_rel, loss_rel, output_path=OUTPUT_DIR)
    utilities.plot_and_save_results(test_dataset_vis, test_dataset, best_model, output_path=OUTPUT_DIR, count=2)


if __name__ == '__main__':
    freeze_support()
    main()
