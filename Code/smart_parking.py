import cv2
from ultralytics import YOLO

# 1. Load the YOLOv8 Nano model
model = YOLO('yolov8n.pt')

# 2. Open your Webcam
cap = cv2.VideoCapture(0)

# 3. YOUR LOCKED-IN COORDINATES [x1, y1, x2, y2]
parking_spots = [
    [310, 51, 481, 120], 
    [314, 162, 512, 245],
    [318, 275, 535, 370]
]

print("Smart Parking Monitor is LIVE... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 4. AI Detection (Using CPU with 0.1 confidence for Hot Wheels)
    results = model.predict(frame, device='cpu', classes=2, conf=0.1, verbose=False)

    # results[0].plot() draws the AI's "car" boxes and confidence levels
    annotated_frame = results[0].plot() 

    # Track which spots are full (start with all False)
    spot_status = [False, False, False] 

    # 5. OCCUPANCY LOGIC
    for r in results:
        for box in r.boxes:
            # Get car coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            # Calculate the Center Point (Centroid) of the car
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Check if this center point is inside any of our 3 spot rectangles
            for i, spot in enumerate(parking_spots):
                if spot[0] < cx < spot[2] and spot[1] < cy < spot[3]:
                    spot_status[i] = True

    # 6. VISUALS: DRAW THE STATUS BOARD & BOXES
    # --- DRAW SUMMARY BOARD (Black bar at the top) ---
    cv2.rectangle(annotated_frame, (0, 0), (640, 30), (0, 0, 0), -1) 
    
    available = spot_status.count(False)
    occupied = spot_status.count(True)
    
    cv2.putText(annotated_frame, f"AVAILABLE: {available}", (15, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"OCCUPIED: {occupied}", (180, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # --- DRAW THE PARKING SPOT BOXES ---
    for i, spot in enumerate(parking_spots):
        color = (0, 0, 255) if spot_status[i] else (0, 255, 0) # Red if full, Green if free
        label = "OCCUPIED" if spot_status[i] else "FREE"
        
        # Draw our custom boxes over the AI-annotated frame
        cv2.rectangle(annotated_frame, (spot[0], spot[1]), (spot[2], spot[3]), color, 2)
        cv2.putText(annotated_frame, f"Spot {i+1}: {label}", (spot[0], spot[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 7. SHOW THE FINAL DASHBOARD
    cv2.imshow("Smart Parking Dashboard", annotated_frame)
    
    # Press 'q' to stop the code safely
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
