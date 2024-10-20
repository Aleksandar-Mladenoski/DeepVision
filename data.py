# My own dataset and dataloader, before I had realised there was a provided one.


from torch.utils.data.dataset import Dataset
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image



'''
Exercise 1 – Submission: a3_ex1.py 60 Points
Write a class ImagesDataset that extends torch.utils.data.Dataset and is responsible for pro-
viding the fixed-size gray-scale images and their additional data which was part of the previous
assignment. The class has the following three instance methods:
• __init__(
self,
image_dir,
width: int = 100,
height: int = 100,
dtype: Optional[type] = None
)

image_dir specifies the directory of validated images as the output directory by function
validate_images from assignment 1. Assume that all image files with the extension ".jpg"
and the class file with the extension ".csv" stored directly in the directory, not be in sub-
directories. The found files must be collected using their absolute paths, and the list of these
must be sorted afterwards in ascending order. The corresponding class names of images are
loaded from the ".csv" file. You can use "numpy.genfromtxt()" or "pandas.read_csv()"
to read the class names file. A list of distinct class names must be sorted in ascending order,
and their index is used as its respective class ID.
width and height specify the fixed size of the resized copy of the images loaded from image_dir.
If width or height are smaller than 100, a ValueError must be raised.
dtype optionally specifies the data type of the loaded images (see __getitem__ below).


• __getitem__(self, index)
This method works as follow:
– Given the specified integer index, the index-th image from the sorted list of image files
(see __init__ above) must be loaded with PIL.Image.open.
– The image is then stored in a NumPy array using the optionally specified dtype (other-
wise, the default data type is used).
– This image array is then transformed into gray-scale using the to_grayscale method
from the previous assignment.
– Afterwards, again from the previous assignment, the method prepare_image(image,
width, height, x=0, y=0, size=32) must be called where width and height are the
fixed size of the resized image. The subarea of the resized image is not used, we therefore
only pass fixed arguments for x, y, and size.
1
Programming in Python II Assignment 3 – Due: 24.04.2024, 11:30 am
The method must then return the following 4-tuple: (image, class_id, class_name,
image_filepath), which are the fixed-size gray-scale copy, class ID (value), class name, and
the absolute file path of the loaded image respectively.



• __len__(self)
Returns the number of samples, i.e., the number of images that were found in __init__.
'''


#print(torch.cuda.is_available())
# target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad_image(image, pad_width, pad_height):

    if isinstance(pad_width, int):
        pad_width = ((0, 0), (pad_width // 2, pad_width // 2 + pad_width % 2))
    elif len(pad_width) == 1:
        pad_width = ((0, 0), (pad_width[0] // 2, pad_width[0] // 2 + pad_width[0] % 2))

    if isinstance(pad_height, int):
        pad_height = ((pad_height // 2, pad_height // 2 + pad_height % 2),)
    elif len(pad_height) == 1:
        pad_height = ((pad_height[0] // 2, pad_height[0] // 2 + pad_height[0] % 2),)

    padded_image = np.pad(image, pad_width + pad_height, mode='edge')

    #PIL_image = Image.fromarray(np.uint8(padded_image[0,:,:])).convert('L')
    #PIL_image = Image.fromarray(padded_image[0,:,:].astype('uint8'), 'L')
    #PIL_image.show()


    return padded_image

def get_subarea(image,x,y,size):
    image_channel, image_height, image_width = image.shape
    subimage = image[:, y:y+size, x:x+size]
    return subimage

    #PIL_image = Image.fromarray(np.uint8(subimage[0,:,:])).convert('L')
    #PIL_image = Image.fromarray(subimage[0,:,:].astype('uint8'), 'L')
    #PIL_image.show()


def crop_image(image, crop_width, crop_height):
    # Crops an image
    img_channel, img_height, img_width = image.shape
    if crop_width % 2 == 0:
        width_left = int((img_width-crop_width)/2)
        width_right = width_left
    else:
        width_left = int((img_width-crop_width)/2) + 1
        width_right = int((img_width-crop_width)/2)


    if crop_height % 2 == 0:
        height_left = int((img_height-crop_height)/2)
        height_right = height_left
    else:
        height_left = int((img_height-crop_height)/2) + 1
        height_right = int((img_height-crop_height)/2)

    cropped_image = image[:, height_left:height_left+crop_height, width_left:width_left + crop_width]

    #PIL_image = Image.fromarray(np.uint8(cropped_image[0,:,:])).convert('L')
    #PIL_image = Image.fromarray(cropped_image[0,:,:].astype('uint8'), 'L')
    #PIL_image.show()

    return cropped_image





def prepare_image(
    image: np.ndarray,
    width: int,
    height: int,
    x: int,
    y: int,
    size: int
) -> tuple[np.ndarray, np.ndarray]:

    # If the input image is smaller in width or height than width or height respectively, pad the
    # image to the desired shape by equally adding pixels to the start and end of that dimension. The
    # pad-pixel used should be the same as the previous border. If an unequal amount of padding
    # needs to be added, add the additional pixel to the end.

    #  The pad-pixel used should be the same as the previous border
    # IMAGE COMES 1,H,W


    img_channel, img_height, img_width = image.shape

    if width > img_width:
        new_img = pad_image(image, 0,width-img_width)
        new_img_channel, new_img_height, new_img_width = new_img.shape
    else:
        new_img = crop_image(image, width, img_height)
        new_img_channel, new_img_height, new_img_width = new_img.shape



    if height > new_img_height:
        final_img = pad_image(new_img, height-new_img_height,0)
    else:
        final_img = crop_image(new_img, new_img_width, height)


    sub_image = get_subarea(final_img, x,y,size)
    return (final_img, sub_image)



def to_grayscale(pil_image: np.ndarray) -> np.ndarray:
    try:
        if len(pil_image.shape) == 2 or pil_image.shape[2] == 1:
            pil_image = np.reshape(pil_image, (1, pil_image.shape[0], pil_image.shape[1]))
            return pil_image
        elif pil_image.shape[2] != 3:
                print('hello')
                raise ValueError
    except IndexError:
        print(pil_image.shape)
    #PIL_image = Image.fromarray(np.uint8(pil_image)).convert('RGB')
    #PIL_image = Image.fromarray(pil_image.astype('uint8'), 'RGB')
    #PIL_image.show()

    #pil_normalized = normalize_array(pil_image, 0,1)
    pil_normalized = pil_image/255

    c_condition = pil_normalized <= 0.04045
    pil_c_linear = np.where(c_condition, pil_normalized/12.92, pow(pil_normalized+0.055/1.055,2.4))
    pil_r_linear = pil_normalized[:, :, 0]
    pil_g_linear = pil_normalized[:, :, 1]
    pil_b_linear = pil_normalized[:, :, 2]

    pil_y_linear = 0.2126*pil_r_linear + 0.7152*pil_g_linear + 0.0722*pil_b_linear

    y_condition = pil_y_linear <= 0.0031308
    pil_y = np.where(y_condition, 12.92*pil_y_linear, 1.055*pow(pil_y_linear, 1/2.4) - 0.055)

    pil_y = pil_y*255

    #PIL_image = Image.fromarray(np.uint8(pil_y)).convert('L')
    #PIL_image = Image.fromarray(pil_y.astype('uint8'), 'L')

    pil_y = np.reshape(pil_y, (1, pil_y.shape[0], pil_y.shape[1]))
    #PIL_image.show()
    #print(pil_y.shape)

    return pil_y

    # Converts raw data from PIL image to grayscale, using  colorimetric conversion.
    # All values must be normalized to [0,1] before the (also normalized ) grayscale output Y is calculated as follows.


def directory_search(input_dir: str):
    try:
        dirs = os.listdir(input_dir)
    except FileNotFoundError:
        raise ValueError(f'{input_dir} is not an existing directory')
    files = list()
    for dir in dirs:
        if os.path.isdir(os.path.join(input_dir, dir)):
            files.extend(directory_search(os.path.join(input_dir, dir)))
        else:
            files.append(os.path.join(input_dir, dir))

    return sorted(files)


class ImagesDataset(Dataset):

    def __init__(
        self,
        image_dir,
        width: int = 100,
        height: int = 100,
        dtype: type = None
    ):
        super().__init__()
        self.dtype = dtype
        self.all_files = sorted(directory_search(image_dir))
        self.csv_index = next((i for i, item in enumerate(self.all_files) if item.endswith('.csv')), None)
        self.csv_file = pd.read_csv(self.all_files[self.csv_index], delimiter=';', header=0, names=['filename', 'label'])
        self.all_files.pop(self.csv_index)
        self.width = width
        self.height = height
        self.classes = sorted(pd.unique(self.csv_file['label'].values))
        self.targets = [self.classes.index(x) for x in self.csv_file['label'].values]
        print(self.classes)
        #for image in self.all_files:
        #    with Image.open(image) as img:
        #        (width, height) = img.size
        #        if width < 100 or height < 100:
        #            raise ValueError




    def __getitem__(self, index):
        with Image.open(self.all_files[index]) as img:
            img_numpy = np.asarray(img, dtype=self.dtype)
            img_grayscale = to_grayscale(img_numpy)
            prepared_image, sub_image = prepare_image(img_grayscale, self.width, self.height, x=0, y=0, size=32)
            file = self.all_files[index]
            absolute_path = file[file.rfind("\\")+1:]
            if not absolute_path.endswith('.jpg'):
                absolute_path = file
            debug = self.csv_file['filename']==absolute_path
            label = self.csv_file.where(self.csv_file['filename'] == absolute_path)['label'].dropna().values
            class_id = self.classes.index(label)


            return prepared_image, class_id, label, self.all_files[index]


    def __len__(self):
        return len(self.all_files)


def main():
    dataset = ImagesDataset("./validated_images(2)", 100, 100, int)
    for resized_image, classid, classname, _ in dataset:
        print(f'image shape: {resized_image.shape}, dtype: {resized_image.dtype}, '
              f'classid: {classid}, classname: {classname}\n')


if __name__ == '__main__':
    main()

def stacking(batch_as_list: list):
    images, classids, classnames, image_filepaths = zip(*batch_as_list)
    tensor_images = list()
    for image in images:
        tensor_images.append(torch.FloatTensor(image))
    images = torch.stack(tensor_images, dim=0)
    classids = torch.tensor(classids, dtype=torch.long).unsqueeze(1)
    classnames = np.asarray(classnames).flatten()
    return images, classids, classnames, image_filepaths


