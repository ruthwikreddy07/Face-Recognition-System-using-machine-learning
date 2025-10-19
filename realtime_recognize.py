# realtime_recognize_vip.py
import cv2, pickle, face_recognition, numpy as np
from PIL import Image
import winsound  # Windows sound alert (you can skip if on Linux/macOS)
import time

ENC_FILE = "encodings/known_encodings.pkl"
TOLERANCE = 0.45  # lower -> stricter
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Load known encodings
with open(ENC_FILE, "rb") as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_names = data["names"]
examples = data["examples"]

# Pre-load thumbnails for known VIPs
thumbs = {}
for name, path in zip(known_names, examples):
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((120, 120))
        thumbs[name] = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    except:
        thumbs[name] = None

cap = cv2.VideoCapture(0)

# To avoid repeating alerts for same VIP every frame
last_alert_time = {}
ALERT_COOLDOWN = 10  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = small[:, :, ::-1]

    boxes = face_recognition.face_locations(rgb_small, model="hog")
    rgb_small = np.ascontiguousarray(rgb_small)
    encodings = face_recognition.face_encodings(rgb_small, known_face_locations=boxes)

    for (top, right, bottom, left), encoding in zip(boxes, encodings):
        top *= 2; right *= 2; bottom *= 2; left *= 2

        matches = face_recognition.face_distance(known_encodings, encoding)
        name = "Unknown"
        if len(matches) > 0:
            best_idx = np.argmin(matches)
            if matches[best_idx] <= TOLERANCE:
                name = known_names[best_idx]

        # Draw box
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), FONT, 0.8, (255, 255, 255), 2)

        # VIP ALERT section
        if name != "Unknown":
            current_time = time.time()
            if name not in last_alert_time or (current_time - last_alert_time[name] > ALERT_COOLDOWN):
                print(f"🎉 VIP Alert: {name} is here!")
                winsound.Beep(1000, 300)  # Beep sound
                last_alert_time[name] = current_time

            # Display VIP alert on screen
            cv2.putText(frame, f" VIP ALERT: {name}!", (50, 60), FONT, 1.0, (0, 255, 255), 3)

        # Show thumbnail if known
        if name != "Unknown" and thumbs.get(name) is not None:
            th = thumbs[name]
            h, w = th.shape[:2]
            x = right + 10
            y = max(0, top - h - 10)
            if x + w > frame.shape[1]:
                x = left - w - 10
            if x < 0: x = 0
            frame[y:y + h, x:x + w] = th

    cv2.imshow("VIP Recognition System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting face recognition...")
        break

cap.release()
cv2.destroyAllWindows()
