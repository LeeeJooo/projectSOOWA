import cv2
import mediapipe as mp
import numpy as np
import time, os

cloudONE = {0:'white', 1:'be', 2:'here', 3:'probably'}
cloudONE = {0:'white', 3:'probably'}
cloudTWO = {0:'cloud_1', 1:'cloud_2', 2:'above', 3:'if'}    # cloud_1: 오른손 위, cloud_2:왼손 위
cloudTWO = {3:'if'}
shineONE = {0:'eye', 1:'light'}
shineTWO = {0:'bright', 1:'morning'}

rainbowONE = {0:'wish', 1:'rainbow'}
rainbowTWO = {0:'sincerely', 1:'if', 2:'rise', 3:'will_1', 4:'will_2'}

likeONE = {0:'I', 1:'you', 2:'like'}
likeTWO = {0:'always'}

snowONE = {0:'white', 1:'color', 2:'leaf'}
snowTWO = {0:'flower', 1:'snow_', 2:'as_', 3:'fly'}

forsythiaONE = {0:'yellow_', 1:'color', 2:'black'}
forsythiaTWO = {0:'flower', 1:'shadow_', 2:'below'}

springONE = {0:'leaf'}
springTWO = {0:'season', 1:'warm', 2:'wind', 3:'because_1', 4:'because_2', 5:'flower', 6:'together', 7:'bright', 8:'smile'}

galaxyONE = {0:'blue', 1:'color', 2:'star'}
galaxyTWO = {0:'bright', 1:'gorup'}

############# 설정 ###############
actions = cloudTWO
hand = 'two'         # one or two
word = 'SL'          # SL or KW

seq_length = 10         # LSTM에 넣을 window 크기
timeONE, timeTWO = 0, 0
TIME = 200
#################################


# MediaPipe holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('test', exist_ok=True)       # dataset을 저장할 폴더 지정


# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # gesture 마다 녹화
        for idx, action in actions.items():
            data1, data2 = np.empty((0, 90)), np.empty((0, 144))
            dataONE, dataTWO = np.empty((0, 0, 90)), np.empty((0, 0, 144))

            ret, img = cap.read()

            img = cv2.flip(img, 1)

            # 어떤 action을 녹화할 것인지 표시
            cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            cv2.imshow('img', img)
            cv2.waitKey(3000)   # 3초간 대기 : 녹화 준비

            while timeONE < TIME and timeTWO < TIME :
                ret, img = cap.read()

                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = holistic.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                Angle, distance = [], []
                right, left = 0, 0     # 손 인식 여부 확인
                RHAngle, LHAngle, PAngle = np.zeros((15,)), np.zeros((15,)), np.zeros((10,))
                RHjoint, LHjoint, Pjoint, Fjoint = np.zeros((21, 3)), np.zeros((21, 3)), np.zeros((33, 3)), np.zeros((468,3))
                # Get Right Hand angle info
                if result.right_hand_landmarks is not None:
                    right = 1
                    for i in range(21):
                        RHjoint[i] = [result.right_hand_landmarks.landmark[i].x, result.right_hand_landmarks.landmark[i].y, result.right_hand_landmarks.landmark[i].z]

                    #joint = np.array([RHjoint.flatten()])

                    # Compute angles between joints
                    v1 = RHjoint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = RHjoint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    RHAngle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                    RHAngle = np.degrees(RHAngle)  # Convert radian to degree
                    Angle = np.append(Angle, RHAngle)
                    #print("RHAngle: ", len(Angle))

                # Get left Hand angle info
                if result.left_hand_landmarks is not None:
                    left = 1

                    for i in range(21):
                        LHjoint[i] = [result.left_hand_landmarks.landmark[i].x, result.left_hand_landmarks.landmark[i].y, result.left_hand_landmarks.landmark[i].z]

                    if right == 1:
                        distance = np.append(distance, LHjoint[0]-RHjoint[0])  # # 오른손목_왼손목 간 거리

                    # Compute angles between joints
                    v1 = LHjoint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = LHjoint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    LHAngle = np.arccos(np.einsum('nt,nt->n',
                                                  v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                  v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                    LHAngle = np.degrees(LHAngle)  # Convert radian to degree
                    #print("left: ", len(LHAngle))
                    Angle = np.append(Angle, RHAngle)

                # Get distance info between face&hand
                if result.face_landmarks is not None:
                    facdidx = [368, 159]
                    handidx = [0, 4, 8, 12, 16, 20]

                    for i in facdidx:
                        Fjoint[i] = Fjoint[i] = [result.face_landmarks.landmark[i].x, result.face_landmarks.landmark[i].y, result.face_landmarks.landmark[i].z]

                    if right == 1:    # 오른쪽손 인식
                        for i in handidx:
                            distance = np.append(distance, Fjoint[368]-RHjoint[i])     # 왼쪽눈-오른쪽손
                            distance = np.append(distance, Fjoint[159]-RHjoint[i])     # 오른쪽눈-오른쪽손

                    if left == 1:    # 왼쪽손 인식
                        for i in handidx:
                            distance = np.append(distance, Fjoint[368]-LHjoint[i])     # 왼쪽눈-왼쪽손
                            distance = np.append(distance, Fjoint[159]-LHjoint[i])     # 오른쪽눈-왼쪽손

                # Get Arm&Face angle info AND distance info between finger&face
                if result.pose_landmarks is not None:
                    for i in range(33):
                        Pjoint[i] = [result.pose_landmarks.landmark[i].x, result.pose_landmarks.landmark[i].y, result.pose_landmarks.landmark[i].z]

                    ## angle info ##
                    # Compute angles between joints
                    v1 = Pjoint[
                         [11, 13, 15, 17, 19, 21, 12, 14, 16, 18, 20, 22, 15, 7, 15, 8, 17, 7, 17, 8, 19, 7, 19, 8, 21,
                          7, 21, 8, 16, 7, 16, 8, 18, 7, 18, 8, 20, 7, 20, 8, 22, 7, 22, 8], :]  # Parent joint
                    v2 = Pjoint[
                         [13, 15, 17, 19, 21, 11, 14, 16, 18, 20, 22, 12, 7, 9, 8, 10, 7, 9, 8, 10, 7, 9, 8, 10, 7, 9,
                          8, 10, 7, 9, 8, 10, 7, 9, 8, 10, 7, 9, 8, 10, 7, 9, 8, 10], :]  # Child joint
                    v = v2 - v1  # [20,3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                    # Get angle using arcos of dot product
                    PAngle = np.arccos(np.einsum('nt,nt->n',
                                                 v[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10,
                                                    12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                                                    32, 34, 36, 38, 40, 42], :],
                                                 v[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11,
                                                    13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
                                                    33, 35, 37, 39, 41, 43], :]))  # [26,]
                    PAngle = np.degrees(PAngle)  # Convert radian to degree     # [26,]
                    Angle = np.append(Angle, PAngle)

                    ## distance info ##
                    dis1 = Pjoint[[17, 21, 17, 21, 18, 22, 18, 22], :]
                    dis2 = Pjoint[[9, 9, 0, 0, 10, 10, 0, 0], :]
                    dis = dis1 - dis2
                    for i in range(4):
                        distance = np.append(distance, dis1[2 * i] - dis[2 * i + 1])  # [12,]
                    Angle = np.append(Angle, distance)


                # 한손 인식
                if hand == 'one' and len(Angle) == 89:
                    angleONE = np.array([Angle], dtype=np.float32)  # (89,)
                    angleONE = np.append(angleONE, idx)  # label 추가 >> 1차원배열로 바뀌어서 오류남
                    data1 = np.append(data1, np.array([angleONE]), axis=0)
                    timeONE += 1
                    print("timeONE: ",timeONE)

                # 두 손 인식
                elif hand == 'two' and len(Angle) == 143:
                    angleTWO = np.array([Angle], dtype=np.float32)
                    angleTWO = np.append(angleTWO, idx)  # label 추가
                    data2 = np.append(data2, np.array([angleTWO]), axis=0)
                    timeTWO += 1
                    print("timeTWO: ", timeTWO)

                else:
                    print("no gesture")

                # Draw Landmark
                # Right Hand
                mp_drawing.draw_landmarks(img, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                # Left Hand
                mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                # Pose
                mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                cv2.imshow('img', img)
                if cv2.waitKey(1) == ord('q'):
                    break

            if hand == 'one':
                timeONE = 0
                print("<",hand,"_",word,"_",action,"> ", data1.shape)
                dataONE = []
                for seq in range(len(data1) - seq_length):
                    dataONE.append(data1[seq:seq + seq_length])
                dataONE = np.array(dataONE)
                print("<", hand, "_", word, "_", action,"> ", dataONE.shape)
                np.save(os.path.join('dataset', f'{hand}_{word}_{action}_{created_time}'), dataONE)

            if hand == 'two':
                timeTWO = 0
                print("<", hand, "_", word, "_", action,"> ", data2.shape)
                dataTWO = []
                for seq in range(len(data2) - seq_length):
                    dataTWO.append(data2[seq:seq + seq_length])
                dataTWO = np.array(dataTWO)
                print("<", hand, "_", word, "_", action,"> ", dataTWO.shape)
                np.save(os.path.join('dataset', f'{hand}_{word}_{action}_{created_time}'), dataTWO)

        break