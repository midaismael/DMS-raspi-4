# Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
from gaze_tracking import GazeTracking
import mediapipe as mp
import numpy as np
from datetime import datetime as  dt
import os
from time import sleep
import face_recognition as fr
import RPi.GPIO as GPIO
from heartrate_monitor import HeartRateMonitor
import time
import argparse
from max30102 import MAX30102
print("Import libraies")

parser = argparse.ArgumentParser(description="Read and print data from MAX30102")
parser.add_argument("-r", "--raw", action="store_true",
                    help="print raw data instead of calculation result")
parser.add_argument("-t", "--time", type=int, default=10,
                    help="duration in seconds to read from sensor, default 30")
args = parser.parse_args()
print(dt.now())

#Gas Sen
 
GPIO.setmode(GPIO.BCM)           # Set's GPIO pins to BCM GPIO numbering
INPUT_PIN = 20           # Sets our input pin, in this example I'm connecting our button to pin 4. Pin 0 is the SDA pin so I avoid using it for sensors/buttons
GPIO.setup(INPUT_PIN, GPIO.IN)           # Set our input pin to be an input
#------------------------------------

#face rec
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
print("Start of Face Recognition")
# Display the resulting image

# Head Pose
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# gaze init
gaze = GazeTracking()

# Initialize the mixer for playing music
mixer.init()
mixer.music.load("music.wav")  # Load the music file

def face_rec(frame):
    img = frame
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_locations = fr.face_locations(img)
    unknown_face_encodings = fr.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = fr.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = fr.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
        ip=1
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            #cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 1)
            font = cv2.FONT_HERSHEY_PLAIN
            #cv2.putText(img, name, (10, 180+(ip*(20))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #Draw a label with a name below the face
            #cv2.rectangle(img, (left-20, bottom -15), (right+20, bottom+20), (255, 0, 0), cv2.FILLED)
            ip=ip+1
            #print(name)
 
            
            #cv2.putText(img, name, (left -20, bottom + 15), font, 1, (255, 255, 255), 1)
            
    img = cv2.resize(img, (0, 0), fx=2, fy=2)
    return img       

# function of eye tracking
def eye(frame):
    # Send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    # Get annotated frame from gaze object
    frame = gaze.annotated_frame()
    text = ""

#     if gaze.is_blinking():
#         text = "Blinking"
    if gaze.is_right():
        text = "Looking right (eye)"
    elif gaze.is_left():
        text = "Looking left (eye)"
    elif gaze.is_center():
        text = "Looking center (eye)" 
    elif gaze.is_up():
        text = "Looking Up (eye)"
    elif gaze.is_down():
        text = "Looking Down (eye)"

    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 2)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (10, 80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 2)
    return frame

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
                state = "Looking left (Head Pose)"
            elif y > 10:
                state = "Looking right (Head Pose)"
            elif x < -10:
                state = "Looking Down (Head Pose)"
            elif x > 10:
                state = "Looking Up (Head Pose)"
            else:
                state = "Forward (Head Pose)"

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
                if current_state == "Forward (Head Pose)":
                    alert_displayed = False
            else:
                # Check if the user has been in the same state for more than 5 seconds
                if time.time() - start_time > 5 and current_state != "Forward (Head Pose)":
                    alert_displayed = True

    # Display the alert if needed
    if alert_displayed:
        cv2.putText(image, "ALERT: Be careful, Look Forward!", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

    return image, current_state, start_time, alert_displayed

# end of head pose nose function

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
    frame = imutils.resize(frame, width=640)

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
            #print(flag)
            # If sustained low EAR is detected, display an alert and play music
            if flag >= frame_check:
                cv2.putText(frame, "********Drowssness ALERT!********", (50, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
                cv2.putText(frame, str(flag), (10, 420),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0

    return frame, flag

#------------------------------------------------------------


#--------- face points and count -------------------
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear
def draw_facial_landmarks(frame, landmarks):
    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        


def facepointscount(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    #print(len(faces))
    for face in faces:
        landmarks = shape_detector(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)
        left_eye_ratio = eye_aspect_ratio(left_eye)
        right_eye_ratio = eye_aspect_ratio(right_eye)

        #print('left_eye_ratio: ',left_eye_ratio,'\t right_eye_ratio: ',right_eye_ratio)

        draw_facial_landmarks(frame, landmarks)
        
        # check left and right eye blinking

        if left_eye_ratio < blink_threshold and right_eye_ratio < blink_threshold:
            cv2.putText(frame, "Both Eye Blinking and sleeping", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif left_eye_ratio < blink_threshold:
            cv2.putText(frame, "Left Eye Blinking", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif right_eye_ratio < blink_threshold:
            cv2.putText(frame, "Right Eye Blinking", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Active", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    
    if len(faces) >1 :
        cv2.putText(frame, "Number of Detected Faces " + str(len(faces)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif len(faces)==1:
        cv2.putText(frame, "Number of Detected Faces " + str(len(faces)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    #cv2.imshow('face points', frame)
    return frame


face_landmark_path='shape_predictor_68_face_landmarks.dat'
shape_detector = dlib.shape_predictor(face_landmark_path)
face_detector = dlib.get_frontal_face_detector()
blink_threshold=0.2

#------ end of face points and ccount -------


# Capture video from the camera
video_capture = cv2.VideoCapture(0)

# Initialize variables
face_recognition_active = False
face_recognition_start_time = None
recognized_face_names = []

# Initialize flag variable
flag = 0

# Main loop to process video frames
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    st=dt.now()
    # Call the drowsiness_monitor function to process the frame
    frame1, flag = drowsiness_monitor(frame, flag)
    et1=dt.now()

    # Display the processed frame
    #cv2.imshow("Drowseness + hp", frame1)
    frame2, current_state, start_time, alert_displayed = headposenose(frame1, current_state, start_time, alert_displayed)
    et2=dt.now()
    
    frame3 = eye(frame2)
    et3=dt.now()
    frame4 = facepointscount(frame3)
    et4=dt.now()
    frame5 = face_rec(frame4)
    et5=dt.now()
    if face_recognition_active:
        frames, recognized_face_names = face_rec(frame4)
        face_recognition_active = False
        face_recognition_start_time = time.time()
    elif face_recognition_start_time and (time.time() - face_recognition_start_time) <= 5:
        frames = frame4
        ip = 1
        for name in recognized_face_names:
            cv2.putText(frames, "Driver: " + name, (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            ip += 1
    else:
        frames = frame4
    et5=dt.now()
    key = cv2.waitKey(1) #& 0xFF
    if key == ord('g'):
        if (GPIO.input(INPUT_PIN) == True): # Physically read the pin now
            #print("Gas Situation: Normal")
            text = "Gas Situation: Normal"
        else:
            #print("Co Situation: Danger")
            text = "Co Situation: Danger"
        cv2.putText(frame5, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Head Pose + Drowseness + Eye Tracking + Face points + Face recognition', frame5)
    #-------------------------------------------
    #Gas Sen
    #Heart sen
    #print("drowsness: "+str(et1-st))
    #print("head position: "+str(et2-et1))
    #print("eye tracking: "+str(et3-et2))
    #print("face point count: "+str(et4-et3))
    #print("face recognition: "+str(et5-et4))
    #print("Total elapsed time: "+str(et5-st))
    # Check for the 'q' key to quit the loop	
    key = cv2.waitKey(1) #& 0xFF
    if key == ord('g'):
        if (GPIO.input(INPUT_PIN) == True): # Physically read the pin now
            #print("Gas Situation: Normal")
            text = "Gas Situation: Normal"
        else:
            #print("Co Situation: Danger")
            text = "Co Situation: Danger"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
    if key == ord("h"):
        print('sensor starting...')
        hrm = HeartRateMonitor(print_raw=args.raw, print_result=(not args.raw))
        hrm.start_sensor()
        try:
            time.sleep(args.time)
        except KeyboardInterrupt:
            print('keyboard interrupt detected, exiting...')

        hrm.stop_sensor()
        print('sensor stoped!') 

    if key == ord("q"):
        break

# Release the video capture object and close all windows
cv2.destroyAllWindows()
video_capture.release()







