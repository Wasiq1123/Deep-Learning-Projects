import supervision as sv
from trackers import SORTTracker
from ultralytics import YOLO
import keyboard
import cv2

tracker = SORTTracker()
model = YOLO("yolo11m.pt")
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update(detections)
    annotated_frame = annotator.annotate(frame, detections, labels=detections.tracker_id)
    cv2.imshow("Webcam Tracking", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or keyboard.is_pressed('k'):
        break

cap.release()
cv2.destroyAllWindows()