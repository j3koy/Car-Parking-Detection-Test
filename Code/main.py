from ultralytics import YOLO

# Load the Nano model
model = YOLO('yolov8n.pt')

# Run the webcam (source 0)
# show=True pops up the window
results = model.predict(source='0', show=True)
