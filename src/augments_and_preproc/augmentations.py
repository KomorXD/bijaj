import albumentations as albu


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.8, border_mode=0),

        albu.PadIfNeeded(min_height=352, min_width=352, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.GaussNoise(p=0.2),

        albu.OneOf(
            [
                albu.CLAHE(p=0.33),
                albu.RandomBrightnessContrast(p=0.33),
                albu.RandomGamma(p=0.33),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=0.5),
                albu.Blur(blur_limit=(3, 5), p=1),
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
        albu.Resize(320, 320, always_apply=True)
    ]

    return albu.Compose(test_transform)
