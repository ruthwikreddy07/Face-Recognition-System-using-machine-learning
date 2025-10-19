# Face-Recognition-System-using-Machine-Learning

This project is a real-time "VIP" alert system using face recognition. It captures video from your webcam, detects faces, and compares them against a pre-built database of known individuals.

When a known person ("VIP") is detected, the system:

- Draws a green box around their face.
- Displays their name.
- Shows a "VIP ALERT!" message on the screen.
- Plays a system beep (on Windows) to provide an audible alert.
- Displays a small thumbnail of the VIP's known photo for confirmation.


---

## Features

- **Real-Time Detection:** Identifies faces live from your webcam feed.
- **VIP Alerts:** Visual (on-screen text) and audible (system beep) alerts.
- **Thumbnail Display:** Shows a small example photo of the recognized VIP.
- **Alert Cooldown:** Prevents spamming alerts for the same person (10-second cooldown).
- **Easy-to-Build Database:** Includes a script (`build_encodings.py`) to automatically learn faces from a folder of

