import sys
import os
import numpy as np
import torch
from torch.utils.data import random_split
from pathlib import Path

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from custom_datasets.vizwiz_dataset import VizWizDataset
from text_tokenizers import get_tokenizer

# Must match the seed used in train_lightning.py
SEED = 42

def print_splits_overview(train_imgs, train_caps, val_imgs, val_caps, test_imgs, test_caps):
    """
    Prints the exact number of images and captions for each pipeline split.
    """
    print("\n" + "="*60)
    print("1. DATASET SPLITS OVERVIEW (IMAGES & CAPTIONS)")
    print("="*60)
    print(f"{'Split':<10} | {'Images':<10} | {'Captions':<10}")
    print("-" * 37)
    print(f"{'Train':<10} | {train_imgs:<10} | {train_caps:<10} (90% of 'train' folder)")
    print(f"{'Validation':<10} | {val_imgs:<10} | {val_caps:<10} (10% of 'train' folder)")
    print(f"{'Test':<10} | {test_imgs:<10} | {test_caps:<10} (100% of 'val' folder)")
    print("-" * 37)
    print(f"{'TOTAL':<10} | {train_imgs + val_imgs + test_imgs:<10} | {train_caps + val_caps + test_caps:<10}")


def explore_dataset_statistics(captions):
    """
    Computes and prints basic text statistics strictly for the training split.
    """
    print("\n" + "="*60)
    print("2. TEXT STATISTICS (FROM 90% TRAIN SPLIT)")
    print("="*60)
    
    word_lengths = [len(str(caption).split()) for caption in captions]
    char_lengths = [len(str(caption)) for caption in captions]
    
    print("--- Word-Level ---")
    print(f"Average words per caption: {np.mean(word_lengths):.2f}")
    print(f"Median words per caption:  {np.median(word_lengths):.2f}")
    print(f"Min words in a caption:    {np.min(word_lengths)}")
    print(f"Max words in a caption:    {np.max(word_lengths)}")
    print(f"95th percentile (words):   {np.percentile(word_lengths, 95):.0f}")
    print(f"99th percentile (words):   {np.percentile(word_lengths, 99):.0f}")

    print("\n--- Character-Level ---")
    print(f"Average chars per caption: {np.mean(char_lengths):.2f}")
    print(f"Max chars in a caption:    {np.max(char_lengths)}")
    
    max_idx = np.argmax(word_lengths)
    print("\n--- Longest Outlier ---")
    print(f"({word_lengths[max_idx]} words) -> '{captions[max_idx]}'")


def compare_tokenizers(captions):
    """
    Initializes all tokenizers to extract vocab_size and max_len 
    using the 100% train split, mirroring train_lightning.py.
    """
    print("\n" + "="*60)
    print("3. TOKENIZER COMPARISON (100% TRAIN FOLDER)")
    print("="*60)
    
    token_levels = ["character", "word", "subword"]
    
    for level in token_levels:
        print(f"\nEvaluating '{level}' tokenizer...")
        try:
            if level == 'word':
                tokenizer = get_tokenizer(tokenizer_type=level, min_freq=5)
            else:
                tokenizer = get_tokenizer(tokenizer_type=level)
                
            tokenizer.build_vocab(captions)
            
            print(f"  > Vocabulary Size: {tokenizer.vocab_size}")
            print(f"  > Model Max Length (max_len): {tokenizer.max_len}")
        except Exception as e:
            print(f"  [!] Error loading {level} tokenizer: {e}")
    print("\n" + "="*60 + "\n")


def extract_captions(dataset, indices=None):
    """Helper function to extract captions from dataset or a subset."""
    caps = []
    # If no indices are provided, iterate over the full length of the dataset
    iterable = indices if indices is not None else range(len(dataset))
    
    for idx in iterable:
        # Match the logic in VizWizDataset.__getitem__ to get the correct image ID
        img_index = idx + dataset.initial_id
        
        # Get all annotations for this specific image using the VizWiz manager
        ann_ids = dataset.manager.getAnnIds(imgIds=[img_index])
        anns = dataset.manager.loadAnns(ids=ann_ids)
        
        # Extract the caption text from each annotation
        for ann in anns:
            caps.append(ann['caption'])
            
    return caps


def main():
    data_root = Path("./data")
    torch.manual_seed(SEED)
    
    print("Loading datasets... (This might take a few seconds)")
    full_train_dataset = VizWizDataset(data_root=data_root, split='train', tokenizer=None)
    test_dataset = VizWizDataset(data_root=data_root, split='val', tokenizer=None)
    
    # 1. Split the train folder (90/10) exactly like in training
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_subset, val_subset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # 2. Extract captions for all splits
    print("Extracting captions for statistics...")
    train_subset_caps = extract_captions(full_train_dataset, train_subset.indices)
    val_subset_caps = extract_captions(full_train_dataset, val_subset.indices)
    test_caps = extract_captions(test_dataset)
    
    # Extract 100% train captions for the tokenizer
    full_train_caps = extract_captions(full_train_dataset)

    # --- Print Outputs ---
    
    # Overview Table
    print_splits_overview(
        train_imgs=train_size, train_caps=len(train_subset_caps),
        val_imgs=val_size, val_caps=len(val_subset_caps),
        test_imgs=len(test_dataset), test_caps=len(test_caps)
    )
    
    # Statistics strictly on the 90% training data
    explore_dataset_statistics(train_subset_caps)
    
    # Tokenizer metrics on the 100% train data
    compare_tokenizers(full_train_caps)

if __name__ == "__main__":
    main()