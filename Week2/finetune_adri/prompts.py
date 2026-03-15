import numpy as np

# Negative prompt parameters
NEG_MARGIN = 20
NEG_MAX_TRIES = 500


def nearest_foreground_to_point(mask: np.ndarray, yx_point: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.array([yx_point[1], yx_point[0]], dtype=np.float32)
    coords = np.stack([ys, xs], axis=1)
    d2 = np.sum((coords - yx_point[None, :]) ** 2, axis=1)
    best = coords[np.argmin(d2)]
    return np.array([best[1], best[0]], dtype=np.float32)

def prompt_center_positive(mask: np.ndarray):
    ys, xs = np.where(mask)
    centroid_y = int(np.round(np.mean(ys)))
    centroid_x = int(np.round(np.mean(xs)))
    point_xy = nearest_foreground_to_point(mask, np.array([centroid_y, centroid_x]))
    return np.array([point_xy], dtype=np.float32), np.array([1], dtype=np.int64)

def prompt_random_positive(mask: np.ndarray):
    ys, xs = np.where(mask)
    idx = np.random.randint(0, len(xs))
    point_xy = np.array([xs[idx], ys[idx]], dtype=np.float32)
    return np.array([point_xy], dtype=np.float32), np.array([1], dtype=np.int64)

def prompt_three_positives(mask: np.ndarray):
    ys, xs = np.where(mask)
    coords = np.stack([xs, ys], axis=1)
    center_pt, _ = prompt_center_positive(mask)
    chosen = [tuple(center_pt[0].astype(int).tolist())]

    if len(coords) > 1:
        perm = np.random.permutation(len(coords))
        for idx in perm:
            candidate = tuple(coords[idx].astype(int).tolist())
            if candidate not in chosen:
                chosen.append(candidate)
            if len(chosen) == 3: break

    while len(chosen) < 3:
        chosen.append(chosen[-1])

    return np.array(chosen, dtype=np.float32), np.array([1, 1, 1], dtype=np.int64)

def prompt_pos_plus_neg(mask: np.ndarray):
    pos_points, _ = prompt_center_positive(mask)
    pos_xy = pos_points[0]
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    h, w = mask.shape
    x0, x1 = max(0, x_min - NEG_MARGIN), min(w - 1, x_max + NEG_MARGIN)
    y0, y1 = max(0, y_min - NEG_MARGIN), min(h - 1, y_max + NEG_MARGIN)

    neg_xy = None
    for _ in range(NEG_MAX_TRIES):
        x, y = np.random.randint(x0, x1 + 1), np.random.randint(y0, y1 + 1)
        if not mask[y, x]:
            neg_xy = np.array([x, y], dtype=np.float32)
            break

    if neg_xy is None:
        bg_ys, bg_xs = np.where(~mask)
        if len(bg_xs) == 0: neg_xy = pos_xy.copy()
        else:
            idx = np.random.randint(0, len(bg_xs))
            neg_xy = np.array([bg_xs[idx], bg_ys[idx]], dtype=np.float32)

    return np.array([pos_xy, neg_xy], dtype=np.float32), np.array([1, 0], dtype=np.int64)
