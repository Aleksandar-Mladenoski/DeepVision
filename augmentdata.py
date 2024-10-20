# Original file from where I augmented my data, only has one thing additional


import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.functional as functional
from dataset import ImagesDataset

import PIL

def gaussian_noise(img):
    img = img + (0.1**0.5)*torch.randn(1, 100, 100)
    return img


def augment_image(img_np: np.ndarray, index: int):
    v = index % 7
    try:
        img_torch = torch.from_numpy(img_np)
    except TypeError:
        img_torch = img_np
    img_torch = img_torch.to(dtype=torch.float32)
    if v == 0:
        return img_torch, "Original"
    augmentation, str_augmentation = transform_picker(v)
    img_augmented = augmentation(img=img_torch)
    img_augmented = img_augmented/255.0
    return img_augmented, str_augmentation


def transform_picker(v: int):
    match v:
        case 1:
            return torchvision.transforms.GaussianBlur(3), "GaussianBlur"
        case 2:
            return torchvision.transforms.RandomRotation(90), "RandomRotation"
        case 3:
            return torchvision.transforms.RandomVerticalFlip(), "RandomVerticalFlip"
        case 4:
            return torchvision.transforms.RandomHorizontalFlip(), "RandomHorizontalFlip"
        case 5:
            return gaussian_noise, "GaussianNoise"
        case 6:
            t1, _ = transform_picker(np.random.randint(1, 6))
            t2, _ = transform_picker(np.random.randint(1, 6))
            t3, _ = transform_picker(np.random.randint(1, 6))
            return torchvision.transforms.Compose([t1,t2,t3]), "Compose"


class TransformedImagesDataset(Dataset):
    def __init__(self, data_set: ImagesDataset):
        super().__init__()
        self.data_set = data_set

    def __getitem__(self, index: int):
        resized_image, classid, classname, image_filepath = self.data_set.__getitem__(index // 7)
        img_augmented, str_augmentation = augment_image(resized_image, index)
        return img_augmented, str_augmentation, torch.tensor(index), classid, classname, image_filepath

    def __len__(self):
        return self.data_set.__len__()*7


def stacking(batch_as_list: list):
    trans_imgs = [item[0] for item in batch_as_list]
    trans_imgs_stacked = torch.stack(trans_imgs, dim=0)

   
    indexes = [item[2] for item in batch_as_list]
    try:
        indexes_stacked = torch.stack(indexes).view(-1, 1)
    except TypeError:
        indexes_stacked = indexes
    class_ids = [torch.tensor(item[3], dtype=torch.long) for item in batch_as_list]
    class_ids_stacked = torch.stack(class_ids).view(-1, 1)

    trans_names = [item[1] for item in batch_as_list]
    class_names = [item[4] for item in batch_as_list]
    img_paths = [item[5] for item in batch_as_list]
    
    return trans_imgs_stacked, trans_names, indexes_stacked, class_ids_stacked, class_names, img_paths


#if __name__ == "__main__":
    #from matplotlib import pyplot as plt
    #dataset = ImagesDataset("./validated_images", 100, 100, int)
    #transformed_ds = TransformedImagesDataset(dataset)
    #fig, axes = plt.subplots(2, 4)
    #for i in range(0, 8):
    #    trans_img, trans_name, index, classid, classname, img_path = transformed_ds.__getitem__(i)
    #    _i = i // 4
    #    _j = i % 4
    #    axes[_i, _j].imshow(functional.to_pil_image(trans_img))
    #    axes[_i, _j].set_title(f'{trans_name}\n{classname}')
    #fig.tight_layout()
    #plt.show()