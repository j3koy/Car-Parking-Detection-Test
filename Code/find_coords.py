import cv2

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked Coordinates: X={x}, Y={y}")

cap = cv2.VideoCapture(0)
cv2.namedWindow("Find Coordinates")
cv2.setMouseCallback("Find Coordinates", click_event)

print("INSTRUCTIONS: Click the TOP-LEFT and BOTTOM-RIGHT corners of your 3 boxes.")

while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow("Find Coordinates", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
