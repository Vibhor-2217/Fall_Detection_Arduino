from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # 'yolov8s.pt' is a larger model

# Train on your dataset
model.train(data="dataset.yaml", epochs=10, imgsz=640, batch=8)
