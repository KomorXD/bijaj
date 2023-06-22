import albumentations as albu


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=0.8, border_mode=0),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        albu.GaussNoise(p=0.2),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=0.5),
                albu.Blur(blur_limit=(3, 5), p=0.5),
                albu.MotionBlur(blur_limit=(3, 5), p=0.5),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=0.5),
                albu.HueSaturationValue(p=0.5),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        albu.PadIfNeeded(320, 320, always_apply=True),
        #albu.RandomCrop(height=256, width=256, always_apply=True),
    ]
    return albu.Compose(test_transform)


def get_test_augmentation():
    test_transform = [
        albu.PadIfNeeded(320, 320, always_apply=True),
        #albu.Normalize(),
    ]
    return albu.Compose(test_transform)
