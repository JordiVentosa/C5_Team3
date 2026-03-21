from .vizwiz import VizWiz
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import random
import torch
import torchvision.transforms as transforms

chars = ['<SOS>', '<EOS>', '<PAD>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char2idx = {v: k for k, v in enumerate(chars)}
TEXT_MAX_LEN = 201


class VizWizDataset(Dataset):
    def __init__(self, data_root: Path = "../data", split='train', img_size=(224, 224), transform=None):
        super().__init__()
        is_train_split = split == 'train'

        if not isinstance(data_root, Path):
            data_root = Path(data_root)

        self.initial_id = 0 if is_train_split else 23431
        self.target = 'train' if is_train_split else 'val'
        self.img_path = data_root / self.target
        self.annotations_path = data_root / f"annotations/{self.target}.json"
        self.manager = VizWiz(annotation_file=self.annotations_path, ignore_precanned=False, ignore_rejected=False)
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        img_index = index + self.initial_id
        img = Image.open(self.img_path / f"VizWiz_{self.target}_{index:08d}.jpg").convert('RGB')
        
        annotation_ids = self.manager.getAnnIds(imgIds=[img_index])
        final_annotation_id = random.choice(annotation_ids)
        final_annotation = self.manager.loadAnns(ids=[final_annotation_id])[0]
        caption_text = final_annotation['caption']
        
        img_tensor = self.transform(img)
        caption_encoded = self._encode_caption(caption_text)
        
        return img_tensor, caption_encoded, caption_text
    
    def __len__(self):
        return len(self.manager.imgs)
    
    def _encode_caption(self, caption: str) -> torch.Tensor:
        encoded = [char2idx['<SOS>']]
        
        for char in caption:
            if char in char2idx:
                encoded.append(char2idx[char])
        
        encoded.append(char2idx['<EOS>'])
        
        while len(encoded) < TEXT_MAX_LEN:
            encoded.append(char2idx['<PAD>'])
        
        encoded = encoded[:TEXT_MAX_LEN]
        
        return torch.tensor(encoded, dtype=torch.long)