import cv2
import threading
from ultralytics import YOLO

MODEL_NAMES = ["yolov8n.pt", "yolov8n-seg.pt"]
SOURCES = ["C://Users//Anas computer//Downloads//sample-20s.mp4", "C://Users//Anas computer//Downloads//sample-30s.mp4"]

def run_tracker_in_thread(model_name, filename, window_name):
    model = YOLO(model_name)
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"❌ Error opening video source: {filename}")
        return

    # Create named window for each thread
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (600, 600))
        results = model.track(source=frame, persist=True, stream=False)
        annotated_frame = results[0].plot()

        # Show in separate window
        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

# Launch threads
threads = []
for i in range(len(MODEL_NAMES)):
    t = threading.Thread(
        target=run_tracker_in_thread,
        args=(MODEL_NAMES[i], SOURCES[i], f"Tracking Window {i+1}"),
        daemon=True
    )
    threads.append(t)
    t.start()

for t in threads:
    t.join()

cv2.destroyAllWindows()
