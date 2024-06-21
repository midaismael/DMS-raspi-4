import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
video_capture = cv2.VideoCapture(0)

def eye(frame):
    # Send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    # Get annotated frame from gaze object
    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 1)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (10, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (147, 58, 31), 1)
    return frame

while True:
    # Get a new frame from the webcam
    _, frame = video_capture.read()

    # Call the eye function to perform gaze tracking
    frame = eye(frame)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break

video_capture.release()
cv2.destroyAllWindows()

