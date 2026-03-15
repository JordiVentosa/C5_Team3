import numpy as np


def collate_fn_with_masks(batch, processor):
    batch = [b for b in batch if b is not None and len(b['boxes']) > 0]
    if len(batch) == 0:
        return None

    images = [b["image"] for b in batch]
    orig_boxes = [b["boxes"] for b in batch]
    orig_true_masks = [b["masks"] for b in batch]
    orig_classes = [b["classes"] for b in batch]
    image_ids = [b["image_id"] for b in batch]

    valid_lengths = [len(boxes) for boxes in orig_boxes]
    max_objects = max(valid_lengths)

    padded_boxes = []
    padded_true_masks = []

    for i in range(len(batch)):
        pad_len = max_objects - valid_lengths[i]

        b_boxes = orig_boxes[i] + [[0.0, 0.0, 0.0, 0.0]] * pad_len
        padded_boxes.append(b_boxes)

        if pad_len > 0:
            mask_shape = orig_true_masks[i][0].shape
            dummy_masks = [np.zeros(mask_shape, dtype=np.float32) for _ in range(pad_len)]
            b_masks = orig_true_masks[i] + dummy_masks
        else:
            b_masks = orig_true_masks[i]

        padded_true_masks.append(b_masks)

    inputs = processor(
        images=images,
        input_boxes=padded_boxes,
        return_tensors="pt"
    )

    return inputs, padded_true_masks, orig_boxes, orig_classes, image_ids, images, valid_lengths


def collate_fn_boxes_only(batch, processor):
    batch = [b for b in batch if b is not None and len(b['boxes']) > 0]
    if len(batch) == 0: return None

    images = [b["image"] for b in batch]
    orig_boxes = [b["boxes"] for b in batch]
    orig_masks = [b["masks"] for b in batch]
    image_ids = [b["image_id"] for b in batch]

    valid_lengths = [len(boxes) for boxes in orig_boxes]
    max_objects = max(valid_lengths)

    padded_boxes = []
    for i in range(len(batch)):
        pad_len = max_objects - valid_lengths[i]
        # Pad with dummy boxes
        b_boxes = orig_boxes[i] + [[0.0, 0.0, 0.0, 0.0]] * pad_len
        padded_boxes.append(b_boxes)

    inputs = processor(images=images, input_boxes=padded_boxes, return_tensors="pt")

    return inputs, orig_boxes, orig_masks, image_ids, valid_lengths, images
