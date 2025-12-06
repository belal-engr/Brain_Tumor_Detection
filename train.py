from ultralytics import YOLO

def main():
    # Load YOLO base model
    model = YOLO("yolov8n.pt")

    model.train(
        data=r"data.yaml",
        epochs=200,
        resume=False,
        device=0,
        workers=8,
        imgsz=416,
        batch=16,
        name="tumor_detection",  # FIXED (no trailing space)
        project="YOLOv8"
    )

if __name__ == "__main__":
    main()
