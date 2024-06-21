import time
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

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
        if (ear < thresh):
            flag += 1
            print(flag)
            # If sustained low EAR is detected, display an alert and play music
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0

    return frame, flag

# Capture video from the camera
video_capture = cv2.VideoCapture(0)

# Initialize flag variable
flag = 0

# Initialize FPS variables
fps_start_time = time.time()
fps_frame_count = 0

# Desired frame rate
desired_fps = 30  # Adjust as needed
frame_duration = 1.0 / desired_fps

# Main loop to process video frames
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    # Call the drowsiness_monitor function to process the frame
    frame, flag = drowsiness_monitor(frame, flag)

    # Display the processed frame
    cv2.imshow("Frame", frame)

    # FPS calculation
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1.0:
        current_fps = fps_frame_count / (time.time() - fps_start_time)
        print(f"FPS: {current_fps:.2f}")
        fps_frame_count = 0
        fps_start_time = time.time()

    # Wait to maintain the desired frame rate
    elapsed_time = time.time() - fps_start_time
    if elapsed_time < frame_duration:
        time.sleep(frame_duration - elapsed_time)

    # Check for the 'q' key to quit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the video capture object and close all windows
cv2.destroyAllWindows()
video_capture.release()

