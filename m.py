import cv2
import numpy as np
import dlib
from imutils import face_utils
import time
import pygame  # Importing pygame for playing sound

# Initializing pygame mixer
pygame.mixer.init()
sleep_sound = pygame.mixer.Sound('alert.mp3')

# Initializing the camera and detector
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Status variables
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Time tracking for sleep
sleep_start_time = None
is_sleep_sound_playing = False

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

def update_status(left_blink, right_blink):
    global sleep, drowsy, active
    if left_blink == 1 or right_blink == 1:
        sleep = 0
        active = 0
        drowsy += 1
        if drowsy > 6:
            return "Drowsy !", (0, 255, 255)
    else:
        drowsy = 0
        sleep = 0
        active += 1
        if active > 6:
            return "Active :)", (0, 255, 0)
    return "", (0, 0, 0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if not faces:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        if left_blink == 0 and right_blink == 0:
            if sleep_start_time is None:
                sleep_start_time = time.time()
            sleep += 1
            drowsy = 0
            active = 0
            if time.time() - sleep_start_time >= 3:
                if not is_sleep_sound_playing:
                    sleep_sound.play(-1)  # Start playing the sound repeatedly
                    is_sleep_sound_playing = True
                status = "Sleeping !!!"  # Updating status when the person is sleeping
                color = (0, 0, 255)
        else:
            if is_sleep_sound_playing:
                sleep_sound.stop()  # Stop the sound when the person wakes up
                is_sleep_sound_playing = False
            sleep_start_time = None
            status, color = update_status(left_blink, right_blink)

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
