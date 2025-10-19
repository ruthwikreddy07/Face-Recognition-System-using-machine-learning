# build_encodings.py
import os, pickle
import face_recognition
from PIL import Image

DATA_DIR = "dataset"
OUT_FILE = "encodings/known_encodings.pkl"
os.makedirs("encodings", exist_ok=True)

known_encodings = []
known_names = []
example_images = []  # store one sample path per person

for person in sorted(os.listdir(DATA_DIR)):
    person_dir = os.path.join(DATA_DIR, person)
    if not os.path.isdir(person_dir): continue
    encs_for_person = []
    for fname in os.listdir(person_dir):
        path = os.path.join(person_dir, fname)
        try:
            img = face_recognition.load_image_file(path)
            boxes = face_recognition.face_locations(img, model="hog")  # or "cnn" if GPU
            if not boxes: 
                continue
            encs = face_recognition.face_encodings(img, boxes)
            if encs:
                encs_for_person.append(encs[0])
        except Exception as e:
            print("skip", path, e)
    if encs_for_person:
        # average encoding (optional) or keep all
        # Here we'll store the mean encoding for simplicity
        import numpy as np
        mean_enc = np.mean(encs_for_person, axis=0)
        known_encodings.append(mean_enc)
        known_names.append(person)
        # take the first image file as example
        example_images.append(next(iter([os.path.join(person_dir,f) for f in os.listdir(person_dir)]), None))
        print(f"Added {person} -> {len(encs_for_person)} images")

with open(OUT_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names, "examples": example_images}, f)

print("Saved encodings to", OUT_FILE)
