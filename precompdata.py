# My first attempt at fixing the augmentation issue, this is old code and should not be used.

from torch.utils.data import Dataset, DataLoader
from data import directory_search
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
def load_metadata(csv_filename):
    metadata_df = pd.read_csv(csv_filename)

    indices = metadata_df['index'].tolist()
    class_ids = metadata_df['class_id'].tolist()
    class_names = metadata_df['class_name'].tolist()
    augmentations = metadata_df['augmentation'].tolist()
    original_filepaths = metadata_df['original_filepath'].tolist()
    transformed_filepaths = metadata_df['transformed_filepath'].tolist()

    metadata_dict = {
        transformed_filepaths[i]: (
            indices[i],
            class_ids[i],
            class_names[i],
            augmentations[i],
            original_filepaths[i]
        )
        for i in range(len(transformed_filepaths))
    }

    return metadata_dict


class preTransformedImagesDataset(Dataset):
    def __init__(
        self,
        image_dir,
        csvfile,
        width: int = 100,
        height: int = 100,
        dtype: type = None
    ):
        super().__init__()
        self.dtype = dtype
        self.all_files = sorted(directory_search(image_dir))
        self.metadata_dict = load_metadata(csvfile)
        self.classes = ['book', 'bottle', 'car', 'cat', 'chair', 'computermouse', 'cup', 'dog', 'flower', 'fork', 'glass', 'glasses', 'headphones', 'knife', 'laptop', 'pen', 'plate',  'shoes', 'spoon', 'tree']
        #self.targets = [self.classes.index(x) for x in self.csv_file['label'].values]
        self.class_id = { self.classes[i] : i for i in range(len(self.classes)) }


    def __getitem__(self, index):
        key = self.all_files[index].replace("\\", r"/")
        #print(key)
        with Image.open(key) as img:
            img_tensor = pil_to_tensor(img).reshape(1,100,100)
            img_tensor = img_tensor.to(torch.float32)
            index, _, class_name, augmentation, original_filepath = self.metadata_dict[key]
            class_name = str(class_name)[2:-2]
            
        return img_tensor, augmentation, index, self.class_id[class_name],class_name, original_filepath


    def __len__(self):
        return len(self.all_files)



if __name__ == "__main__":
    dataset = preTransformedImagesDataset("project/augment_images", "project/metadata.csv")
    
    for i in range(10):
        img_augmented, augmentation, index, class_id, class_name, filepath = dataset.__getitem__(i)
        show_img = img_augmented.numpy().copy()
        show_img = np.reshape(show_img, (100,100,))
        PIL_image = Image.fromarray(show_img.astype('uint8'), 'L')
        PIL_image.show()

        print(augmentation, index, class_id, class_name, filepath)