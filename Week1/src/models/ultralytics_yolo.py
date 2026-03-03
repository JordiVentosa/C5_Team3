from ultralytics import YOLO

class YOLOInference:
    def __init__(self, model_version='yolov8n.pt'):
        self.model = YOLO(model_version)
    
    def predict(self, image_path, conf_threshold=0.25, save_results=False, imgsz=1248):
        # KITTI-MOTS images are 1242px wide. imgsz must be a multiple of 32,
        # so 1248 is the closest value
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            imgsz=imgsz,
            save=save_results,
            project="runs/inference",
            name="yolo_kitti_mots",
            exist_ok=True
        )
        return results