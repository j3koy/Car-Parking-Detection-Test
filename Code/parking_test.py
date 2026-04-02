from ultralytics import YOLO
import cv2

# Load the Nano model
model = YOLO('yolov8n.pt')

# Open your webcam
cap = cv2.VideoCapture(0)

print("Starting Parking Monitor (CPU Mode)... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run detection (we removed show=True here)
        results = model.predict(frame, device='cpu', classes=2)

        # MANUAL WINDOW CODE
        # results[0].plot() draws the boxes and labels on the image for you
        annotated_frame = results[0].plot() 
        
        # This force-opens a window named "Parking Monitor"
        cv2.imshow("Parking Monitor", annotated_frame)

        # 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()
