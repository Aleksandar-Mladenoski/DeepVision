# This is some more complex augmentation I did for model accuracy improvements, I got quite lazy with the coding
# I found it more convinient to have all these seperate functions outside the match case statement in transform_picker
# It is defenitely not as clean, a little sloppy, but I had no time so ye :) 



import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.functional as functional
from dataset import ImagesDataset

import PIL

def gaussian_noise(img, mean=0.0, std=0.038):
    noise = std * torch.randn(img.size()) + mean
    img = img + noise
    return img

def color_jitter(img):
    return torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)(img)

def affine_transform(img):
    return torchvision.transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10)(img)

def elastic_transform(img):
    return torchvision.transforms.ElasticTransform(alpha=1.0, sigma=0.2)(img)

def cutout(img):
    return torchvision.transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3))(img)

def invert_colors(img):
    return functional.invert(img)

def random_perspective(img):
    return torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=1.0)(img)

def random_crop(img):
    return torchvision.transforms.RandomResizedCrop(size=img.shape[-2:], scale=(0.8, 1.0))(img)

def random_affine(img):
    return torchvision.transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2))(img)

def random_grayscale(img):
    return torchvision.transforms.RandomGrayscale(p=1.0)(img)

def solarize(img, threshold=128):
    if img.max() <= threshold:
        threshold = img.max().item() - 1
    img = img.to(torch.uint8)
    return functional.solarize(img, threshold)

def autocontrast(img):
    return functional.autocontrast(img)

def equalize(img):
    img = img.to(torch.uint8)
    return functional.equalize(img)

def adjust_gamma(img):
    return functional.adjust_gamma(img, gamma=2.0, gain=1.0)

def random_zoom(img):
    scale = np.random.uniform(0.8, 1.2)
    return functional.affine(img, angle=0, translate=(0, 0), scale=scale, shear=0)

def augment_image(img_np: np.ndarray, index: int):
    v = index % 19  
    try:
        img_torch = torch.from_numpy(img_np)
    except TypeError:
        img_torch = img_np
    img_torch = img_torch.to(dtype=torch.float32)
    if v == 0:
        return img_torch, "Original"
    augmentation, str_augmentation = transform_picker(v)
    img_augmented = augmentation(img=img_torch)
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
            return affine_transform, "AffineTransform"
        case 7:
            return elastic_transform, "ElasticTransform"
        case 8:
            return cutout, "Cutout"
        case 9:
            return color_jitter, "ColorJitter"
        case 10:
            t1, _ = transform_picker(np.random.randint(1, 18))
            t2, _ = transform_picker(np.random.randint(1, 18))
            t3, _ = transform_picker(np.random.randint(1, 18))
            return torchvision.transforms.Compose([t1,t2,t3]), "Compose"
        case 11:
            return invert_colors, "InvertColors"
        case 12:
            return random_perspective, "RandomPerspective"
        case 13:
            return random_crop, "RandomCrop"
        case 14:
            return random_affine, "RandomAffine"
        case 15:
            return random_grayscale, "RandomGrayscale"
        case 16:
            return autocontrast, "AutoContrast"
        case 17:
            return adjust_gamma, "AdjustGamma"
        case 18:
            return random_zoom, "RandomZoom"
class TransformedImagesDataset(Dataset):
    def __init__(self, data_set: ImagesDataset):
        super().__init__()
        self.data_set = data_set

    def __getitem__(self, index: int):
        resized_image, classid, classname, image_filepath = self.data_set.__getitem__(index // 19)
        img_augmented, str_augmentation = augment_image(resized_image, index)
        return img_augmented, str_augmentation, torch.tensor(index), classid, classname, image_filepath

    def __len__(self):
        return self.data_set.__len__() * 19

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
