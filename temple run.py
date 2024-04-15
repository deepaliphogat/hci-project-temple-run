import time
import pip
import cv2
import mediapipe as mp
import postestimationmodule as pem
from pynput.keyboard import Key, Controller

cap = cv2.VideoCapture(0)
detector = pem.PoseDetector()
keyboard = Controller()

counter = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (750, 550))
    img = cv2.flip(img, 1)

    detector.find_pose(img, draw=False)
    lm_list = detector.get_position(img, draw=False)

    if lm_list:
        p1, p2 = lm_list[1][1:], lm_list[23][1:]
        left, right = lm_list[18][1:], lm_list[19][1:]

        l, _, _ = detector.find_distance(p1, p2)
        l1, _, _ = detector.find_distance(left, right)

        if l1 < 80:
            if counter == 0:
                keyboard.press(Key.up)
                keyboard.release(Key.up)

        if p1[1] > 250:
            if counter == 0:
                keyboard.press(Key.down)
                keyboard.release(Key.down)

        if left[0] < 150:
            keyboard.press(Key.left)
            keyboard.release(Key.left)

        if right[0] > 600:
            keyboard.press(Key.right)
            keyboard.release(Key.right)

        counter += 1

    if counter == 11:
        counter = 0

    cv2.imshow("Temple Run", img)
    cv2.waitKey(1)
