from ultralytics import YOLO

# Load the YOLO model (assumes object detection/classification model)
model = YOLO("best.pt")

def predict(image_path):
    results = model(image_path)
    return results[0]  # prediction object (contains boxes, labels, etc.)
