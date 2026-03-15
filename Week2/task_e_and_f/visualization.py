import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def save_visualization_basic(image, boxes, pred_masks, true_masks, model_name, img_id, output_dir):
    """Save 5 visualization variations for qualitative analysis."""
    image = np.array(image)
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    np.random.seed(img_id)

    num_objects = max(len(pred_masks), len(true_masks))
    colors = np.random.rand(num_objects, 3)

    pred_overlay = np.zeros((image.shape[0], image.shape[1], 4))
    if len(pred_masks) > 0:
        for idx, mask in enumerate(pred_masks):
            pred_overlay[mask > 0.5] = np.append(colors[idx], 0.75)

    gt_overlay = np.zeros((image.shape[0], image.shape[1], 4))
    if len(true_masks) > 0:
        for idx, mask in enumerate(true_masks):
            gt_overlay[mask > 0.5] = np.append(colors[idx], 0.75)

    def save_fig(img, overlay=None, draw_boxes=False, suffix=""):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)
        if overlay is not None:
            ax.imshow(overlay)
        if draw_boxes:
            for box in boxes:
                x_min, y_min, x_max, y_max = box
                ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                             linewidth=1.5, edgecolor='r', facecolor='none'))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"img_{img_id:03d}_{safe_name}_{suffix}.png"), bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)

    save_fig(image, suffix="1_original")
    save_fig(image, draw_boxes=True, suffix="2_bboxes")
    save_fig(image, overlay=pred_overlay, suffix="3_pred_masks")
    save_fig(image, overlay=pred_overlay, draw_boxes=True, suffix="4_combined")
    save_fig(image, overlay=gt_overlay, suffix="5_gt_masks")


def save_visualization_with_ignore(image, boxes, pred_masks, true_masks, ignore_mask, model_name, img_name, output_dir):
    import cv2

    image = np.array(image)
    safe_name = model_name.replace(" ", "_")
    np.random.seed(42)

    num_objects = max(len(pred_masks), len(true_masks))
    colors = np.random.rand(num_objects, 3)

    pred_overlay = np.zeros((image.shape[0], image.shape[1], 4))
    for idx, mask in enumerate(pred_masks):
        pred_overlay[mask > 0.5] = np.append(colors[idx], 0.85)

    gt_overlay = np.zeros((image.shape[0], image.shape[1], 4))
    for idx, mask in enumerate(true_masks):
        gt_overlay[mask > 0.5] = np.append(colors[idx], 0.85)

    ignore_overlay = None
    ignore_boxes = []
    if ignore_mask is not None and ignore_mask.any():
        ignore_overlay = np.zeros((image.shape[0], image.shape[1], 4))
        ignore_overlay[ignore_mask] = [120/255, 120/255, 120/255, 0.4]

        ignore_u8 = ignore_mask.astype(np.uint8)
        num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(ignore_u8, connectivity=8)
        for comp_id in range(1, num_labels):
            stats_comp = stats[comp_id]
            area = int(stats_comp[cv2.CC_STAT_AREA])
            if area > 10:
                x, y, w, h = int(stats_comp[cv2.CC_STAT_LEFT]), int(stats_comp[cv2.CC_STAT_TOP]), int(stats_comp[cv2.CC_STAT_WIDTH]), int(stats_comp[cv2.CC_STAT_HEIGHT])
                ignore_boxes.append([x, y, x+w, y+h])

    def save_fig(img, overlay=None, draw_boxes=False, draw_ignore=False, suffix=""):
        fig, ax = plt.subplots(1, figsize=(12, 4))
        ax.imshow(img)
        if draw_ignore and ignore_overlay is not None:
            ax.imshow(ignore_overlay)
            for ix_min, iy_min, ix_max, iy_max in ignore_boxes:
                ax.add_patch(patches.Rectangle((ix_min, iy_min), ix_max - ix_min, iy_max - iy_min, linewidth=1.5, edgecolor=(120/255, 120/255, 120/255), facecolor='none'))
                ax.text(ix_min, max(10, iy_min - 5), "ignore class", color='white', fontsize=8, fontweight='bold', bbox=dict(facecolor=(120/255, 120/255, 120/255), alpha=0.7, pad=0.5, edgecolor='none'))
        if overlay is not None:
            ax.imshow(overlay)
        if draw_boxes:
            for x_min, y_min, x_max, y_max in boxes:
                ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1.5, edgecolor='r', facecolor='none'))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"{img_name}_{safe_name}_{suffix}.png"), bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close(fig)

    save_fig(image, draw_ignore=False, suffix="1_original")
    save_fig(image, draw_boxes=True, draw_ignore=True, suffix="2_bboxes")
    save_fig(image, overlay=pred_overlay, draw_ignore=True, suffix="3_pred_masks")
    save_fig(image, overlay=gt_overlay, draw_ignore=True, suffix="4_gt_masks")
    save_fig(image, overlay=pred_overlay, draw_boxes=True, draw_ignore=True, suffix="5_combined")
