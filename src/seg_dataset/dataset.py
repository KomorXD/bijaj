import cv2
import os
import numpy as np
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    CLASSES = [
        "Animal",
        "MaskingBackground",
        "NonMaskingBackground",
        "NonMaskingForegroundAttention",
        "None",
    ]

    COLORS = {
        0: (0, 0, 255),
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (255, 255, 255),
        4: (0, 0, 0),
    }

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.image_ids = sorted(os.listdir(images_dir))
        self.masks_ids = sorted(os.listdir(masks_dir))
        self.ids = self.masks_ids
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.masks_ids]

        self.class_values = [self.CLASSES.index(cls) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        iheight, iwidth, _ = image.shape
        mheight, mwidth, _ = mask.shape

        inew_width = (iwidth // 32) * 32
        inew_height = (iheight // 32) * 32

        mnew_width = (mwidth // 32) * 32
        mnew_height = (mheight // 32) * 32

        image = cv2.resize(image, (inew_width, inew_height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (mnew_width, mnew_height), interpolation=cv2.INTER_AREA)

        masks = []
        for cls_value in self.class_values:
            color = self.COLORS[cls_value]
            new_mask = cv2.inRange(mask, color, color)
            new_mask = np.float32(new_mask)
            new_mask = cv2.normalize(new_mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            masks.append(new_mask)
        mask = np.stack(masks, axis=-1).astype('float32')

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)