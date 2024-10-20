# This is just some more debugging I did when my augmentations were not working



import os
import csv
from PIL import Image
import numpy as np

# Load the validation metadata
validation_images_dir = "project/validation_images"
metadata_file = os.path.join(validation_images_dir, "labels.csv")

with open(metadata_file, mode='r') as file:
    reader = csv.reader(file, delimiter=';')
    rows = list(reader)

# Verify that the first row contains the headers
assert rows[0] == ["name", "label"], "Invalid metadata headers"

# Verify the integrity of the validation data
for row in rows[1:]:
    image_filename, label = row
    image_filepath = os.path.join(validation_images_dir, image_filename)
    
    # Check if the image file exists
    assert os.path.exists(image_filepath), f"Image file {image_filepath} does not exist"

    # Check if the image can be opened
    with Image.open(image_filepath) as img:
        img_array = np.array(img)
        # Verify image shape
        assert img_array.shape == (100, 100), f"Invalid image shape {img_array.shape} for {image_filepath}"
