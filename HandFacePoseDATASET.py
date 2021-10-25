import cv2
import mediapipe as mp
import numpy as np
import time, os

SLTWO1 = {0:'above', 1:'below', 2:'resemble', 3:'fly', 4:'rise',
          5:'together', 6:'as_', 7:'if', 8:'will', 9:'sincerely'}
SLTWO2 = {10:'because_1', 11:'because_2',      #because_1:오른손주먹, because_2:왼손주먹
          12: 'smile', 13:'bright', 14:'morning', 15:'group', 16:'jewel',
          17:'explosion', 18:'always', 19:'flower'}
SLTWO3 = {20:'fruit', 21:'shadow_', 22:'season', 23:'warm', 24:'hot',
          25:'wind', 26:'winter', 27:'cloud_1', 28:'cloud_2', 29:'snow_',
          30:'fall', 31:'wave_'}       #cloud_1: 오른손이 위, #cloud_2: 왼손이 위
SLRIGHT1 = {0:'I', 1:'you', 2:'be', 3:'wish', 4:'like', 5:'shy',
           6:'here', 7:'probably', 8:'eye', 9:'light'}
SLRIGHT2 = {10:'color', 11:'white_', 12:'blue_', 13:'yellow_', 14:'red_',
            15:'black_', 16:'thorn', 17:'leaf', 18:'water', 19:'rainbow', 20:'star'}
SLtwoONE = {0:'above_ONE', 1:'below_ONE', 2:'together_ONE', 3:'if_ONE', 4:'sincerely_ONE', 5:'because_ONE',
            6:'bright_ONE', 7:'morning_ONE', 8:'jewel_ONE', 9:'flower_ONE', 10:'fruit_ONE',
            11:'season_ONE', 12:'fall_ONE', 13:'rise_ONE'}
KWTWO = {0:'finger_heart_TWO', 1:'small_heart', 2:'middle_heart', 3:'big_heart',
         4:'hi_TWO', 5:'flower_cup', 6:'V_TWO_palm', 7:'V_TWO_back', 8:'fuckyou_TWO'}
KWONE1 = {0:'finger_heart_LEFT', 1:'finger_heart_RIGHT', 2:'hi_ONE_LEFT', 3:'hi_TWO_RIGHT',
          4:'meosseug_LEFT', 5:'meosseug_RIGHT', 6:'kosseug_LEFT'}
KWONE2 = {7:'kosseug_RIGHT', 8:'fuckyou_LEFT',
          9:'fuckyou_RIGHT', 10:'jawV_LEFT', 11:'jawV_RIGHT', 12:'V_ONE_LEFT', 13:'V_ONE_RIGHT'}

actions = SLTWO1     # SLTWO1,SLTWO2,SLTWO3 / KWTWO or SLRIGHT1,SLRIGHT2 / SLtwoONE / KWONE1,KWONE2
hand = 'two'         # one or two
word = 'SL'          # SL or KW

seq_length = 10         # LSTM에 넣을 window 크기
timeONE, timeTWO = 0, 0
TIME = 100

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