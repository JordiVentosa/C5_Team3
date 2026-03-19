from vizwiz import VizWiz
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random
import numpy as np

class VizWizDataset(Dataset):

    def __init__(self, data_root : Path = "../data", split='train'):
        """
        Dataset class for the VizWiz dataset using the recomended VizWiz API

        Parameters
        ----------
        data_root : Path
            The root folder where the dataset is located. It must contain both the images and the annotations
        split: 'train' | 'test'
            The split of images to use
        """
        
        super().__init__()

        is_train_split = split == 'train'

        if not isinstance(data_root, Path):
            data_root = Path(data_root)

        self.initial_id = 0 if is_train_split else 23431
        self.target = 'train' if is_train_split else 'val'

        self.img_path = data_root / self.target
        self.annotations_path = data_root / f"annotations/{self.target}.json"
        
        self.manager = VizWiz(annotation_file=self.annotations_path, ignore_precanned=False, ignore_rejected=False)

    # --------------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------------

    def __getitem__(self, index):
        
        img_index = index + self.initial_id

        img = Image.open(self.img_path / f"VizWiz_{self.target}_{index:08d}.jpg").convert('RGB')

        annotation_ids = self.manager.getAnnIds(imgIds=[img_index])

        final_annotation_id = random.choice(annotation_ids)

        final_annotation = self.manager.loadAnns(ids=[final_annotation_id])[0]

        return np.array(img), final_annotation['caption']