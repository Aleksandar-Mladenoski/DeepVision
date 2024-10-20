# This is a codefile to save the augmented images done by the complex augmentation ( or the normal one )
# To one folder where I can then later just read off the data and progress quickly.


import os
import csv
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from complex_augment import TransformedImagesDataset
from dataset import ImagesDataset
import numpy as np
from PIL import Image

def save_metadata(metadata, rows: list, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(rows)
        writer.writerows(metadata)


def main():
    dataset = ImagesDataset(r'training_data', 100, 100)
    BATCH_SIZE = 64
    TEST_SIZE = 0.1
    SEED = 42

    train_indices, test_indices, _, _ = train_test_split(
        range(len(dataset)),
        dataset.targets,
        stratify=dataset.targets,
        test_size=TEST_SIZE,
        random_state=SEED
    )

    train_split = Subset(dataset, train_indices)
    test_split = Subset(dataset, test_indices)

    print(len(test_split))


    augment_train_split = TransformedImagesDataset(train_split)

    augmented_images_dir = "project/augment_images"
    os.makedirs(augmented_images_dir, exist_ok=True)
    metadata = []

    SIZE = len(augment_train_split)
    print(SIZE)
    for i in range(SIZE):
        img_augmented, str_augmentation, index, classid, classname, image_filepath = augment_train_split.__getitem__(i)
        image_filename = f"{augmented_images_dir}/{i}.jpg"
        
        show_img = img_augmented.numpy().copy()
        show_img = np.reshape(show_img, (100,100,))*255
        PIL_image = Image.fromarray(show_img.astype('uint8'), 'L')
        #PIL_image.show()

        output_dir = "project/augment_images"
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{index}.jpg")
        PIL_image.save(path, format = 'JPEG')

        transformed_filepath = f"{augmented_images_dir}/{i}.jpg"
        metadata.append([f"{index}.jpg", classname])
    
    save_metadata(metadata=metadata, filename=f"project/augment_images/labels.csv", rows=["name", "label"])


    validation_images_dir = "project/validation_images"
    os.makedirs(validation_images_dir, exist_ok=True)
    metadata = []
    SIZE = len(test_split)
    print(SIZE) 
    
    for i in range(SIZE):
        resized_image, classid, classname, filepath = test_split.__getitem__(i)
        resized_image = resized_image*255.0
        show_img = resized_image.numpy().copy()
        show_img = np.reshape(show_img, (100,100,))
        PIL_image = Image.fromarray(show_img.astype('uint8'), 'L')
        #PIL_image.show()

        #print(classname, classid, resized_image.shape)


        path = os.path.join(validation_images_dir, f"{i}.jpg")
        PIL_image.save(path, format = 'JPEG')

        transformed_filepath = f"{validation_images_dir}/{i}.jpg"
        metadata.append([f"{i}.jpg", classname])
    save_metadata(metadata=metadata, filename=f"project/validation_images/labels.csv", rows=["name", "label"])


if __name__ == "__main__":
    main()