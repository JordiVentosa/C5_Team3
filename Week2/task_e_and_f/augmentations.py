import albumentations as A

def get_augmentations(da_config: int):
    """Build Albumentations pipeline based on DA_CONFIG (0-3)."""
    bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels'], min_area=4, min_visibility=0.2)
    
    if da_config == 0:  # Baseline
        return A.Compose([], bbox_params=bbox_params)
    
    if da_config == 1:  # Photometric (color, lighting, noise)
        return A.Compose([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.6),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomBrightnessContrast(p=0.5)
        ], bbox_params=bbox_params)

    if da_config == 2:  # Geometric (flips, crops, scales)
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(size=(384, 1280), scale=(0.7, 1.0), p=0.7)
        ], bbox_params=bbox_params)

    if da_config == 3:  # Combined + aggressive (photometric + geometric + rotation + occlusion)
        return A.Compose([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(size=(384, 1280), scale=(0.7, 1.0), p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.4),
            A.CoarseDropout(max_holes=6, max_height=40, max_width=40, min_holes=1, min_height=10, min_width=10, p=0.5)
        ], bbox_params=bbox_params)
    
    