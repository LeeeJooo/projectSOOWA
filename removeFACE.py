import cv2 as cv
import mediapipe as mp

# cascade classifier (다단계 분류)를 이용한 객체 검출(Object Detection)
# Haar cascade classifier 이란?
# 다수의 객체 이미지(이를 positive 이미지라 함)와 객체가 아닌 이미지(이를 negative 이미지라 함)를 cascade 함수로 트레이닝 시켜 객체 검출을 달성하는 머신러닝 기반의 접근 방법
# cascade로 사람 얼굴을 검출할 것임
# 얼굴 검출을 위해 많은 수으 ㅣ얼굴 이미지와 얼굴이 없는 이미지를 classifier에 트레이닝 시켜 얼굴에 대한 특징들을 추출해서 데이터로 저장
# 얼굴 검출을 위한 Haar-Cascade 트레이닝 데이터를 읽어 CascadeClassifier 객체를 생성
cascade = cv.CascadeClassifier(cv.samples.findFile("haarcascade_frontalface_alt.xml"))  # 사람 얼굴 정면에 대한 Haar-Cascade 학습 데이터

# MediaPipe holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv.VideoCapture(0)    # VideoCapture 객체 생성

# 라이브로 들어오는 비디오를 frame 별로 캡쳐하고 이를 화면에 display
# 특정 키를 누를 때까지 무한 루프
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, img = cap.read()

        # 비디오 프레임을 제대로 읽었다면 ret 값이 True가 되고 실패하면 False가 된다
        if ret == False:
            break

        #img_result = img.copy()
        img = cv.flip(img, 1)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result = holistic.process(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        # Draw Landmark
        # Right Hand
        mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # Left Hand
        mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # Pose
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)  # 이미지 흑백처리
        gray = cv.equalizeHist(gray)    # 히스토그램 평활화(Histogram Equalization)를 적용하여 이미지의 콘트라스트를 향상시킴

        # 얼굴 위치를 리스트로 리턴 (x, y, w, h) / (x, y ):얼굴의 좌상단 위치, (w, h): 가로 세로 크기
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                         flags=cv.CASCADE_SCALE_IMAGE)
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # 얼굴 영역에 검정 사각형 만들기
        height, width = img.shape[:2]
        for x1, y1, x2, y2 in rects:
            cv.rectangle(img, (x1 - 10, y1+y2+10), (x1+x2+10, y1-20), (153, 102, 204), -1)      # bgr



        cv.imshow("Result", img)
        cv.waitKey(1)