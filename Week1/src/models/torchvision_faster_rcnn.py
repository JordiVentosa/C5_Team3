import torch
from torchvision import models, transforms
import torch.nn as nn
import cv2

class FasterRCNNInference(nn.Module):
    def __init__(self, conf_threshold=0.5, device=None):
        super(FasterRCNNInference, self).__init__()
        self.conf_threshold = conf_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # HWC uint8 -> CHW float [0,1]
        ])

    def forward(self, x):
        return self.model(x)

    def predict(self, img_path, conf_threshold=None,FRCNN_ALLOWED_CLASSES = {1, 3}):
        """
        Run inference on a single image path.
        Returns a list of (label_id, score, (x1, y1, x2, y2)) tuples
        filtered to FRCNN_ALLOWED_CLASSES only.
        """
        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)[0]

        results = []
        for label, score, box in zip(outputs["labels"], outputs["scores"], outputs["boxes"]):
            label_id = int(label)
            score    = float(score)
            if label_id not in FRCNN_ALLOWED_CLASSES:
                continue
            if score < threshold:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            results.append((label_id, score, (x1, y1, x2, y2)))

        return results
