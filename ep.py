"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import RPi.GPIO as GPIO
from time import sleep
GPIO.setmode(GPIO.BCM)
buzz = 26
GPIO.setup(buzz,GPIO.OUT)

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
d=0
while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
        d=d+0.1
    elif gaze.is_right():
        text = "Looking right"
        d=0
    elif gaze.is_left():
        text = "Looking left"
        d=0
    elif gaze.is_center():
        text = "Looking center"
        d=0
    else:
        text = "Distracted"
        d=d+1
    if d>5:
        print('Alarm')
        GPIO.output(buzz,GPIO.HIGH)
        #d=0
    else:
        GPIO.output(buzz,GPIO.LOW)
    
    
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (147, 58, 31), 1)
    print(text)
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (10, 80), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
