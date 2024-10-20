# This is a simple adaptation of the dataset file provided by the professors, I had reused it and changed some things
# so I could use it with an already saved dataset of augmented images, since that was by far a lot faster than doing it on the fly


import os
import torch
from typing import Optional
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import math
from glob import glob
import csv

class ImagesDataset(Dataset):
    def __init__(
            self,
            image_dir,
            width: int = 100,
            height: int = 100,
            dtype: Optional[type] = None
    ):
        self.image_filepaths = sorted(glob(os.path.join(image_dir, "*.jpg")))
        class_filepath = glob(os.path.join(image_dir, "*.csv"))[0]
        self.filenames_classnames, self.classnames_to_ids = self.load_classnames(class_filepath)
        if width < 100 or height < 100:
            raise ValueError('width and height must be greater than or equal 100')
        self.width = width
        self.height = height
        self.dtype = dtype
        self.index_mapping = {os.path.basename(fp): i for i, fp in enumerate(self.image_filepaths)}
        #self.targets = [self.classes.index(x) for x in self.csv_file['label'].values]

        with open(class_filepath, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')
            self.classid_dict = dict()
            for row in spamreader:
                self.classid_dict.update({row[0]: row[1]})


    @staticmethod
    def load_classnames(class_filepath: str):
        filenames_classnames = np.genfromtxt(class_filepath, delimiter=';', skip_header=1, dtype=str)
        classnames = np.unique(filenames_classnames[:, 1])
        classnames.sort()
        classnames_to_ids = {classname: index for index, classname in enumerate(classnames)}
        return filenames_classnames, classnames_to_ids

    def __getitem__(self, index):
        image_filepath = self.image_filepaths[index]
        image_name = os.path.basename(image_filepath)
        #class_row = next(row for row in self.filenames_classnames if row[0] == image_name)
        #classid = self.classnames_to_ids[classname]

        classname = self.classid_dict[image_name]
        classid = self.classnames_to_ids[classname]
        
        with Image.open(image_filepath) as im:
            image = np.array(im, dtype=self.dtype)
        image = to_grayscale(image)
        resized_image, _ = prepare_image(image, self.width, self.height, 0, 0, 32)
        resized_image = torch.tensor(resized_image, dtype=torch.float32) / 255.0
        
        return resized_image, classid, classname, image_filepath

    def __len__(self):
        return len(self.image_filepaths)
    
def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    if pil_image.ndim == 2:
        return pil_image.copy()[None]
    if pil_image.ndim != 3:
        raise ValueError("image must have either shape (H, W) or (H, W, 3)")
    if pil_image.shape[2] != 3:
        raise ValueError(f"image has shape (H, W, {pil_image.shape[2]}), but it should have (H, W, 3)")
    
    rgb = pil_image / 255
    rgb_linear = np.where(
        rgb < 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4
    )
    grayscale_linear = 0.2126 * rgb_linear[..., 0] + 0.7152 * rgb_linear[..., 1] + 0.0722 * rgb_linear[..., 2]
    
    grayscale = np.where(
        grayscale_linear < 0.0031308,
        12.92 * grayscale_linear,
        1.055 * grayscale_linear ** (1 / 2.4) - 0.055
    )
    grayscale = grayscale * 255
    
    if np.issubdtype(pil_image.dtype, np.integer):
        grayscale = np.round(grayscale)
    return grayscale.astype(pil_image.dtype)[None]

def prepare_image(image: np.ndarray, width: int, height: int, x: int, y: int, size: int):
    if image.ndim < 3 or image.shape[-3] != 1:
        raise ValueError("image must have shape (1, H, W)")
    if width < 32 or height < 32 or size < 32:
        raise ValueError("width/height/size must be >= 32")
    if x < 0 or (x + size) > width:
        raise ValueError(f"x={x} and size={size} do not fit into the resized image width={width}")
    if y < 0 or (y + size) > height:
        raise ValueError(f"y={y} and size={size} do not fit into the resized image height={height}")
    
    image = image.copy()

    if image.shape[1] > height:
        image = image[:, (image.shape[1] - height) // 2: (image.shape[1] - height) // 2 + height, :]
    else: 
        image = np.pad(image, ((0, 0), ((height - image.shape[1])//2, math.ceil((height - image.shape[1])/2)), (0, 0)), mode='edge')
    
    if image.shape[2] > width:
        image = image[:, :, (image.shape[2] - width) // 2: (image.shape[2] - width) // 2 + width]
    else:
        image = np.pad(image, ((0, 0), (0, 0), ((width - image.shape[2])//2, math.ceil((width - image.shape[2])/2))), mode='edge')

    subarea = image[:, y:y + size, x:x + size]
    return image, subarea