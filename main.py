import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from cvzone.FaceMeshModule import FaceMeshDetector

lastBlink = 0


def main():
    global lastBlink

    lastBlink = 0

    sens = get_sensitivity()

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    detector = FaceMeshDetector(maxFaces=1)

    cap = cv2.VideoCapture(0)

    ratioList = []
    counter = 0

    while cap.isOpened():
        start_time = time.time()

        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            face_2d, face_3d = extract_face_landmarks(face_landmarks, image)
            x, y = calculate_head_pose(face_3d, face_2d, image)
            move_mouse(x, y, sens, image)
            image, faces = detector.findFaceMesh(image, draw=False)
            detect_blink(faces, ratioList, counter, image,
                         start_time, detector, x, y)

        cv2.imshow("Image", image)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_sensitivity():
    while True:
        sens = float(input("Choose your sensitivity Level between 1 - 3: "))
        if 1 <= sens <= 3:
            return sens
        else:
            print("Please enter a valid sensitivity level.")


def extract_face_landmarks(face_landmarks, image):
    img_h, img_w, _ = image.shape
    face_2d = []
    face_3d = []

    for idx, lm in enumerate(face_landmarks.landmark):
        if idx in [33, 263, 1, 61, 291, 199]:
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    return face_2d, face_3d


def calculate_head_pose(face_3d, face_2d, image):
    img_w, _, _ = image.shape
    cam_matrix = np.array(
        [[img_w, 0, img_w / 2], [0, img_w, img_w / 2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix)

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360
    y = angles[1] * 360

    return x, y


def move_mouse(x, y, sens, image):
    text = ""
    if y < -11:
        text = "Looking Left"
        pyautogui.moveRel(y * sens, 0)
    elif y > 11:
        text = "Looking Right"
        pyautogui.moveRel(y * sens, 0)
    elif x < -3:
        text = "Looking Down"
        pyautogui.moveRel(0, -x * sens * 1.3)
    elif x > 11:
        text = "Looking Up"
        pyautogui.moveRel(0, -x * sens)
    else:
        text = "Forward"

    cv2.putText(image, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


def detect_blink(faces, ratioList, counter, image, start_time, detector, x, y):
    global lastBlink

    image, faces = detector.findFaceMesh(image, draw=False)

    if faces and abs(x) < 3 and abs(y) < 11:
        face = faces[0]
        leftUp = face[159]
        leftDown = face[145]
        leftLeft = face[454]
        leftRight = face[234]
        lengthVer, _ = detector.findDistance(leftUp, leftDown)
        lengthHor, _ = detector.findDistance(leftLeft, leftRight)
        ratio = int((lengthVer / lengthHor) * 100)
        ratioList.append(ratio)

        if len(ratioList) > 3:
            ratioList.pop(0)

        ratioAvg = sum(ratioList) / len(ratioList)

        if ratioAvg < 35 and counter == 0:
            if time.perf_counter() - lastBlink < 2:
                pyautogui.click(clicks=1, button='right')

            else:
                pyautogui.click(clicks=1, button='left')

            lastBlink = time.perf_counter()
            counter = 1

            end_time = time.time()
            click_response_time = end_time - start_time
            print("Click Response Time", click_response_time)

    if counter != 0:
        counter -= 1
        if counter > 10:
            counter = 0


if __name__ == '__main__':
    main()
