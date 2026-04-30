import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# ---------------- AUDIO ----------------
pygame.mixer.init()
pygame.mixer.music.load("assets/music.mp3")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)

WIDTH, HEIGHT = 900, 550

# ---------------- VIDEOS ----------------
video1 = cv2.VideoCapture("assets/clip.mp4")
video2 = cv2.VideoCapture("assets/clipp.mp4")
video3 = cv2.VideoCapture("assets/clippp.mp4")

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

face = mp_face.FaceMesh(max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6)

mp_draw = mp.solutions.drawing_utils

# ---------------- VARIABLES ----------------
mode = 0
system_on = False
music_on = False

last_finger = -1
cooldown = 0.3
last_time = 0

# ---------------- FUNCTIONS ----------------
def count_fingers(hand):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand[4][1] > hand[3][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for i in range(1, 5):
        if hand[tip_ids[i]][2] < hand[tip_ids[i]-2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers.count(1)

def get_frame(vid):
    ret, v = vid.read()
    if not ret or v is None:
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, v = vid.read()
    if v is None:
        return None
    return cv2.resize(v, (180,180))

# ---------------- SHAPES ----------------
def draw_square(frame, vid):
    v = get_frame(vid)
    if v is None: return
    frame[30:210, 30:210] = v
    cv2.rectangle(frame, (30,30), (210,210), (0,255,0), 3)

def draw_circle(frame, vid):
    v = get_frame(vid)
    if v is None: return

    mask = np.zeros((180,180), dtype=np.uint8)
    cv2.circle(mask, (90,90), 90, 255, -1)

    roi = frame[30:210, 30:210]
    mask3 = cv2.merge([mask]*3)

    bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask3))
    fg = cv2.bitwise_and(v, mask3)

    frame[30:210, 30:210] = cv2.add(bg, fg)
    cv2.circle(frame, (120,120), 90, (255,0,255), 3)

def draw_triangle(frame, vid):
    v = get_frame(vid)
    if v is None: return

    pts = np.array([[120,30],[30,210],[210,210]], np.int32)

    mask = np.zeros((180,180), dtype=np.uint8)
    pts_shift = np.array([[90,0],[0,180],[180,180]], np.int32)
    cv2.fillPoly(mask, [pts_shift], 255)

    roi = frame[30:210, 30:210]
    mask3 = cv2.merge([mask]*3)

    bg = cv2.bitwise_and(roi, cv2.bitwise_not(mask3))
    fg = cv2.bitwise_and(v, mask3)

    frame[30:210, 30:210] = cv2.add(bg, fg)
    cv2.polylines(frame, [pts], True, (0,255,255), 3)

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_result = hands.process(rgb)
    face_result = face.process(rgb)

    total_fingers = 0

    # -------- HAND DETECTION --------
    if hand_result.multi_hand_landmarks:
        for handLms in hand_result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * WIDTH), int(lm.y * HEIGHT)
                lm_list.append((id, cx, cy))

            total_fingers += count_fingers(lm_list)
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # -------- GESTURE CONTROL --------
    current_time = time.time()

    if total_fingers != last_finger and (current_time - last_time > cooldown):
        last_time = current_time
        last_finger = total_fingers

        if total_fingers == 5:
            system_on = True

        elif total_fingers == 10:
            system_on = False
            pygame.mixer.music.stop()
            music_on = False

        elif system_on:
            if total_fingers == 1:
                mode = 1
            elif total_fingers == 2:
                mode = 2
            elif total_fingers == 3:
                mode = 3

    # -------- SMILE DETECTION (PLAY / PAUSE) --------
    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:

            left = face_landmarks.landmark[61]
            right = face_landmarks.landmark[291]

            lx, ly = int(left.x * WIDTH), int(left.y * HEIGHT)
            rx, ry = int(right.x * WIDTH), int(right.y * HEIGHT)

            smile_dist = abs(lx - rx)

            # Debug text
            cv2.putText(frame, f"Smile: {smile_dist}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            cv2.circle(frame, (lx, ly), 3, (0,255,0), -1)
            cv2.circle(frame, (rx, ry), 3, (0,255,0), -1)

            if smile_dist > 60 and system_on:
                if not music_on:
                    pygame.mixer.music.play(-1)
                    music_on = True

            elif smile_dist < 50:
                if music_on:
                    pygame.mixer.music.pause()
                    music_on = False

    # -------- DRAW VIDEO --------
    if system_on:
        if mode == 1:
            draw_square(frame, video1)
        elif mode == 2:
            draw_circle(frame, video2)
        elif mode == 3:
            draw_triangle(frame, video3)

    # -------- DASHBOARD --------
    overlay = frame.copy()
    cv2.rectangle(overlay, (550, 320), (880, 530), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "GESTURE CONTROL PANEL", (560, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.putText(frame, "5 Fingers : START", (560, 380),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.putText(frame, "10 Fingers : STOP", (560, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.putText(frame, "Smile 😊 : MUSIC", (560, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    cv2.putText(frame, "1/2/3 Fingers : Video", (560, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.putText(frame, f"Fingers: {total_fingers}", (560, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    status_color = (0,255,0) if system_on else (0,0,255)
    cv2.putText(frame, f"System: {'ON' if system_on else 'OFF'}", (560, 485),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    cv2.putText(frame, f"Music: {'ON' if music_on else 'OFF'}", (560, 510),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # -------- DISPLAY --------
    cv2.imshow("AI Gesture Media Player ✨", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
video1.release()
video2.release()
video3.release()
cv2.destroyAllWindows()