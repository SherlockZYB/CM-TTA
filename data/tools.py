import os
import cv2
import torch
import numpy as np
import albumentations as A
import torchvision.transforms as transforms
from imagecorruptions import corrupt, get_corruption_names

# A.RandomCropNearBBox()
# A.BBoxSafeRandomCrop()
# A.RandomSizedBBoxSafeCrop()
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)



weak_transforms = A.Compose(
    [
        # A.Flip(),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        # A.BBoxSafeRandomCrop()
    ],
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    # keypoint_params=A.KeypointParams(format='xy')
)


strong_transforms = A.Compose(
    [
        A.Posterize(),
        A.Equalize(),
        A.Sharpen(),
        A.Solarize(),
        A.RandomBrightnessContrast(),
        A.RandomShadow(),
    ]
)


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes=None, visual=False):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        masks = [transforms.Pad(padding)(mask) for mask in masks]

        # Adjust bounding boxes
        if bboxes is not None:
            bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
            bboxes = [
                [bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h]
                for bbox in bboxes
            ]
            if visual:
                return padding, image, masks, bboxes
            else:
                return image, masks, bboxes
        else:
            if visual:
                return padding, image, masks
            else:
                return image, masks

    def transform_image(self, image):
        # Resize image and masks
        image = self.transform.apply_image(image)
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        return image

    def transform_coord(self, points, image):
        og_h, og_w, _ = image.shape
        coords = points.reshape(1, -1, 2)
        points = self.transform.apply_coords(coords, (og_h, og_w))
        return points.reshape(-1, 2)

    def transform_coords(self, points, image, n):
        og_h, og_w, _ = image.shape
        coords = points.reshape(-1, n, 2)
        points = self.transform.apply_coords(coords, (og_h, og_w))
        return points.reshape(-1, n, 2)


def corrupt_image(image, filename):
    file_name = os.path.basename(os.path.abspath(filename))
    file_path = os.path.dirname(os.path.abspath(filename))
    for corruption in get_corruption_names():
        corrupted = corrupt(image, severity=5, corruption_name=corruption)
        corrupt_path = file_path.replace(
            "val2017", os.path.join("corruption", corruption)
        )
        if not os.path.exists(corrupt_path):
            os.makedirs(corrupt_path, exist_ok=True)
        cv2.imwrite(os.path.join(corrupt_path, file_name), corrupted)


def soft_transform(image: np.ndarray, bboxes: list, masks: list, categories: list):
    weak_transformed = weak_transforms(
        image=image, bboxes=bboxes, masks=masks, category_ids=categories
    )
    image_weak = weak_transformed["image"]
    bboxes_weak = weak_transformed["bboxes"]
    masks_weak = weak_transformed["masks"]

    strong_transformed = strong_transforms(image=image_weak)
    image_strong = strong_transformed["image"]
    return image_weak, bboxes_weak, masks_weak, image_strong


def soft_transform_all(
    image: np.ndarray, bboxes: list, masks: list, points: list, categories: list
):
    weak_transformed = weak_transforms(
        image=image,
        bboxes=bboxes,
        masks=masks,
        category_ids=categories,
        keypoints=points,
    )
    image_weak = weak_transformed["image"]
    bboxes_weak = weak_transformed["bboxes"]
    masks_weak = weak_transformed["masks"]
    keypoints_weak = weak_transformed["keypoints"]

    strong_transformed = strong_transforms(image=image_weak)
    image_strong = strong_transformed["image"]
    return image_weak, bboxes_weak, masks_weak, keypoints_weak, image_strong


def collate_fn(batch):

    if len(batch[0]) == 3:
        images, bboxes, masks = zip(*batch)
        images = torch.stack(images)
        return images, bboxes, masks
    elif len(batch[0]) == 4:
        images, bboxes, masks, name = zip(*batch)
        images = torch.stack(images, dim=0)
        img_niis = [None] * len(name)
        return images, bboxes, masks, name, img_niis
    elif len(batch[0]) == 5:
        images, bboxes, masks, name, img_niis = zip(*batch)
        if(type(images[0]) == list): # 针对AugMix
            images = torch.stack(images[0],dim=0)
        else:
            images = torch.stack(images,dim=0)
        return  images, bboxes, masks, name, img_niis
    elif len(batch[0]) == 9:
        image, bboxes, bboxes_coarse, masks, basename, image_weak, image_strong, bboxes_weak, masks_weak = zip(*batch)
        image = torch.stack(image)
        image_weak = torch.stack(image_weak)
        image_strong = torch.stack(image_strong)   
        return image, bboxes, bboxes_coarse, masks, basename, image_weak, image_strong, bboxes_weak, masks_weak
    else:
        raise ValueError("Unexpected batch format")


def collate_fn_(batch):
    return zip(*batch)


def decode_mask(mask):
    """
    Convert mask with shape [1, h, w] using 1, 2, 3, ... to represent different objects
    to a mask with shape [n, h, w] using a new dimension to represent the number of objects.

    Args:
        mask (torch.Tensor): Mask tensor with shape [1, h, w] using 1, 2, 3, ... to represent different objects.

    Returns:
        torch.Tensor: Mask tensor with shape [n, h, w] using a new dimension to represent the number of objects.
    """
    unique_labels = torch.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    n_objects = len(unique_labels)
    new_mask = torch.zeros((n_objects, *mask.shape[1:]), dtype=torch.int64)
    for i, label in enumerate(unique_labels):
        new_mask[i] = (mask == label).squeeze(0)
    return new_mask


def encode_mask(mask):
    """
    Convert mask with shape [n, h, w] using a new dimension to represent the number of objects
    to a mask with shape [1, h, w] using 1, 2, 3, ... to represent different objects.

    Args:
        mask (torch.Tensor): Mask tensor with shape [n, h, w] using a new dimension to represent the number of objects.

    Returns:
        torch.Tensor: Mask tensor with shape [1, h, w] using 1, 2, 3, ... to represent different objects.
    """
    n_objects = mask.shape[0]
    new_mask = torch.zeros((1, *mask.shape[1:]), dtype=torch.int64)
    for i in range(n_objects):
        new_mask[0][mask[i] == 1] = i + 1
    return new_mask


if __name__ == "__main__":
    mask_encode = np.array([[[0, 0, 1], [2, 0, 2], [0, 3, 3]]])
    mask_decode = np.array(
        [
            [[0, 0, 1], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [1, 0, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 1, 1]],
        ]
    )
    encoded_mask = encode_mask(torch.tensor(mask_decode))
    decoded_mask = decode_mask(torch.tensor(mask_encode))
