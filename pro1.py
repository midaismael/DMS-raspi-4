# Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import mediapipe as mp
import numpy as np
import time

# Head Pose
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the mixer for playing music
mixer.init()
mixer.music.load("music.wav")  # Load the music file

# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function of head pose nose
def headposenose(frame, current_state, start_time, alert_displayed):
    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # To improve performance
    image.flags.writeable = False
    
    # Get the result
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            # The camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])

            # The Distance Matrix
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360

            # Determine the head pose state
            if y < -10:
                state = "Looking left"
            elif y > 10:
                state = "Looking right"
            elif x < -10:
                state = "Looking Down"
            elif x > 10:
                state = "Looking Up"
            else:
                state = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
            p2 = (320, 240)
            
            cv2.line(image, p1, p2, (255, 0, 0), 2)

            # Add the text on the image
            cv2.putText(image, state, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Check if the state has changed
            if state != current_state:
                current_state = state
                start_time = time.time()
                if current_state == "Forward":
                    alert_displayed = False
            else:
                # Check if the user has been in the same state for more than 5 seconds
                if time.time() - start_time > 5 and current_state != "Forward":
                    alert_displayed = True

    # Display the alert if needed
    if alert_displayed:
        cv2.putText(image, "ALERT: Be careful, Look Forward!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image, current_state, start_time, alert_displayed

# Set threshold and frame check parameters
thresh = 0.25
frame_check = 20

# Initialize face detector and facial landmarks predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Define the indices for left and right eye landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Function to encapsulate the drowsiness detection logic
def drowsiness_monitor(frame, flag):
    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=650)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)

    # Loop over detected faces
    for subject in subjects:
        # Predict facial landmarks for each face
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculate eye aspect ratio for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Create convex hulls around the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        # Draw contours around the eyes
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check if the eye aspect ratio is below the threshold
        if ear < thresh:
            flag += 1
            # If sustained low EAR is detected, display an alert and play music
            if flag >= frame_check:
                cv2.putText(frame, "********Drowsiness ALERT!********", (10, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, str(flag), (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0
    return frame, flag


# Capture video from the camera
video_capture = cv2.VideoCapture(0)

# Initialize variables for tracking head pose state and time
current_state = "Forward"
start_time = time.time()
alert_displayed = False

# Initialize flag variable
flag = 0

def read_frame():
  

# Main loop to process video frames
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)

    # Call the drowsiness_monitor function to process the frame
    frame1, flag = drowsiness_monitor(frame, flag)
    
    # Call the headposenose function to process the frame
    frame2, current_state, start_time, alert_displayed = headposenose(frame1, current_state, start_time, alert_displayed)
    
    # Display the processed frame
    cv2.imshow('Head Pose Estimation + Drowsiness', frame2)
    
    # Check for the 'q' key to quit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video capture object and close all windows
cv2.destroyAllWindows()
video_capture.release()

