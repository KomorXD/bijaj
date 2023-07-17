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

TRAIN_IMG_DIR = Path("../dataset/train").resolve()
TRAIN_MASKS_DIR = Path("../dataset/train_masks").resolve()

VALID_IMG_DIR = Path("../dataset/valid").resolve()
VALID_MASKS_DIR = Path("../dataset/valid_masks").resolve()

TEST_IMG_DIR = Path("../dataset/test").resolve()
TEST_MASKS_DIR = Path("../dataset/test_masks").resolve()

ENCODER = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['Animal', "MaskingBackground", 'NonMaskingBackground', "NonMaskingForegroundAttention"]
ACTIVATION = 'softmax2d'
DEVICE = 'cuda'
EPOCHES = 600
LR = 0.000001

MODEL_USED = "UnetPlusPlus"
LOSS_USED = "CrossEntropyLoss"
OUTPUT_DIR = Path(f"../output/{MODEL_USED}-{ACTIVATION}-{LOSS_USED}-{ENCODER}-{LR}").resolve()
OUT_MODEL = Path(f"../best-2nd-model-{MODEL_USED}-{ACTIVATION}-{LOSS_USED}-{ENCODER}-{LR}.pth").resolve()
IN_MODEL = Path("../train-cand.pth").resolve()

LOSS_LABELS = {
    "CrossEntropyLoss": 'cross_entropy_loss',
    "DiceLoss": "dice_loss",
    "JaccardLoss": "jaccard_loss"
}


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if not Path(OUTPUT_DIR).exists():
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    utilities.fix_ssl_bug_thingy()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    '''model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )'''

    model = torch.load(IN_MODEL)

    train_dataset = seg_dataset.MyDataset(
        TRAIN_IMG_DIR,
        TRAIN_MASKS_DIR,
        augmentation=aap.get_training_augmentation(),
        preprocessing=aap.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = seg_dataset.MyDataset(
        VALID_IMG_DIR,
        VALID_MASKS_DIR,
        augmentation=aap.get_validation_augmentation(),
        preprocessing=aap.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    class_weights = torch.tensor([1.0, 0.2, 0.2, 0.1])
    loss = smp.utils.losses.CrossEntropyLoss(weight=class_weights)
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=LR),
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
    best_epoch = 0
    iou_scores_rel = {
        "train": [],
        "valid": []
    }
    loss_rel = {
        "train": [],
        "valid": []
    }

    try:
        for i in range(0, EPOCHES):
            print(f'\nEpoch: {i} [max IoU score: {max_score:.2f} from epoch #{best_epoch}]')
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            iou_scores_rel["train"].append(train_logs['iou_score'])
            loss_rel["train"].append(train_logs[LOSS_LABELS[LOSS_USED]])

            iou_scores_rel["valid"].append(valid_logs['iou_score'])
            loss_rel["valid"].append(valid_logs[LOSS_LABELS[LOSS_USED]])

            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                best_epoch = i
                torch.save(model, OUT_MODEL)
                print('Model saved!')

    except KeyboardInterrupt:
        print("Epoch loop broken")

    best_model = torch.load(OUT_MODEL)

    test_dataset = seg_dataset.MyDataset(
        TEST_IMG_DIR,
        TEST_MASKS_DIR,
        augmentation=aap.get_test_augmentation(),
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

    test_dataset_vis = seg_dataset.MyDataset(
        TEST_IMG_DIR,
        TEST_MASKS_DIR,
        classes=CLASSES,
    )

    utilities.plot_and_save_training_data(iou_scores_rel, loss_rel, output_path=OUTPUT_DIR)
    utilities.plot_and_save_results(test_dataset_vis, test_dataset, best_model, output_path=OUTPUT_DIR, count=10)


if __name__ == '__main__':
    freeze_support()
    main()
