import os
import face_recognition as fr
import cv2
import numpy as np
import time

# Initialize face encodings
encoded = {}
print("Start Training")
for dirpath, dnames, fnames in os.walk("./faces"):
    for f in fnames:
        if f.endswith(".jpg") or f.endswith(".png"):
            face = fr.load_image_file("faces/" + f)
            encoding = fr.face_encodings(face)[0]
            encoded[f.split(".")[0]] = encoding

faces = encoded
faces_encoded = list(faces.values())
known_face_names = list(faces.keys())
print("Training is Ended")

# Initialize variables
face_recognition_active = False
face_recognition_start_time = None
recognized_face_names = []

def face_rec(frame):
    img = frame
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        matches = fr.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"
        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    ip = 1
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, name, (10, 180 + (ip * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        ip += 1

    img = cv2.resize(img, (0, 0), fx=2, fy=2)
    return img, face_names

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()

    if face_recognition_active:
        frames, recognized_face_names = face_rec(frame)
        face_recognition_active = False
        face_recognition_start_time = time.time()
    elif face_recognition_start_time and (time.time() - face_recognition_start_time) <= 5:
        frames = frame
        ip = 1
        for name in recognized_face_names:
            cv2.putText(frames, name, (10, 180 + (ip * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            ip += 1
    else:
        frames = frame

    cv2.imshow("Demo", frames)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('f'):
        face_recognition_active = True

video_capture.release()
cv2.destroyAllWindows()
