import cv2
import mediapipe as mp
import numpy as np
import time, os

actionsONE = {0:'long for', 1:'rainbow'}
actionsTWO = {0:'sincerely', 1:'if', 2:'rise', 3:'will'}
actions = actionsONE
hand = 'ONE'
word = 'SL'

seq_length = 10         # LSTM에 넣을 window 크기
timeONE, timeTWO = 0, 0
TIME = 500

# MediaPipe holistic model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)       # dataset을 저장할 폴더 지정




# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # gesture 마다 녹화
        for idx, action in actions.items():
            data53, data68 = [], []

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

                Angle = []
                right, left = 0, 0     # 손 인식 여부 확인
                RHAngle, LHAngle, PAngle = np.zeros((15,)), np.zeros((15,)), np.zeros((10,))
                RHjoint, LHjoint, Pjoint, Fjoint = np.zeros((21, 3)), np.zeros((21, 3)), np.zeros((33, 3)), np.zeros((468,3))

                # Get Right Hand angle info
                if result.right_hand_landmarks is not None:
                    right = 1
                    for i in range(21):
                        RHjoint[i] = [result.right_hand_landmarks.landmark[i].x, result.right_hand_landmarks.landmark[i].y, result.right_hand_landmarks.landmark[i].z]
                    joint = np.array([RHjoint.flatten()])

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

                # Get left Hand angle info
                if result.left_hand_landmarks is not None:
                    left = 1
                    for i in range(21):
                        LHjoint[i] = [result.left_hand_landmarks.landmark[i].x, result.left_hand_landmarks.landmark[i].y, result.left_hand_landmarks.landmark[i].z]
                        # visibility: landmark가 이미지 상에서 보이는지 안보이는지를 판단
                    if right == 1:
                        joint = np.append(joint, LHjoint.flatten())
                        #print("RIGHT & LEFT: ", len(joint))     #126
                    elif right == 0:
                        joint = np.array(LHjoint.flatten())
                        #print("ONLY LEFT: ", len(joint))        #63

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
                    Angle = np.append(Angle, RHAngle)

                # Get Arm&Face angle info AND distance info between finger&face
                if result.pose_landmarks is not None:
                    for i in range(33):
                        Pjoint[i] = [result.pose_landmarks.landmark[i].x, result.pose_landmarks.landmark[i].y, result.pose_landmarks.landmark[i].z]
                        # visibility: landmark가 이미지 상에서 보이는지 안보이는지를 판단

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
                                                 v[
                                                 [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                                                  32, 34, 36, 38, 40, 42], :],
                                                 v[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29,
                                                    31, 33, 35, 37, 39, 41, 43], :]))  # [26,]
                    PAngle = np.degrees(PAngle)  # Convert radian to degree
                    Angle = np.append(Angle, PAngle)
                    #print("PAngle:", len(PAngle))
                    #print(PAngle)


                    ## distance info ##
                    dis1 = Pjoint[[17, 21, 17, 21, 18, 22, 18, 22], :]
                    dis2 = Pjoint[[9, 9, 0, 0, 10, 10, 0, 0], :]
                    dis = dis1 - dis2
                    distance = []
                    for i in range(4):
                        distance = np.append(distance, dis1[2 * i] - dis[2 * i + 1])  # [12,]
                    Angle = np.append(Angle, distance)

                # 한손 인식
                if hand == 'one' and len(Angle) == 53:
                    #print("no gesture")
                    #continue
                    angle53 = np.array([Angle], dtype=np.float32)   # (53,)
                    print("idx : ", idx)
                    angle53 = np.append(angle53, idx)  # label 추가
                    #print("angle53.shape: ", angle53.shape)     # (54,)
                    #print("joint.shape: ", joint.shape)         # (1,63)
                    d53 = np.concatenate([joint.flatten(), angle53])
                    #print("d53.shape: ", d53.shape)       # (117,)
                    data53.append(d53)
                    timeONE += 1
                    print("timeONE: ",timeONE)
                    #cv2.imshow('img', img)
                    #continue

                # 두 손 인식
                elif hand == 'two' and len(Angle) == 68:
                    angle68 = np.array([Angle], dtype=np.float32)
                    angle68 = np.append(angle68, idx)  # label 추가
                    print("angle68.shape: ", angle68.shape)     # (69,)
                    print("joint.shape: ", joint.shape)         # (126,)
                    d68 = np.concatenate([joint.flatten(), angle68])
                    print("d68.shape: ", d68.shape)             # (195,)
                    data68.append(d68)
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
                data53 = np.array(data53)
                #print("<",hand,"_",word,"_",action,"> ", data53.shape)
                full_seq_data53 = []
                for seq in range(len(data53) - seq_length):
                    full_seq_data53.append(data53[seq:seq + seq_length])
                full_seq_data53 = np.array(full_seq_data53)
                print("<", hand, "_", word, "_", action,"> ", full_seq_data53.shape)
                np.save(os.path.join('dataset', f'{hand}_{word}_{action}_{created_time}'), full_seq_data53)

            if hand == 'two':
                timeTWO = 0
                data68 = np.array(data68)
                print("<", hand, "_", word, "_", action,"> ", data68.shape)
                full_seq_data68 = []
                for seq in range(len(data68) - seq_length):
                    full_seq_data68.append(data68[seq:seq + seq_length])
                full_seq_data68 = np.array(full_seq_data68)
                print("<", hand, "_", word, "_", action,"> ", full_seq_data68.shape)
                np.save(os.path.join('dataset', f'{hand}_{word}_{action}_{created_time}'), full_seq_data68)

            #print(action, data.shape)

            #print("len(data): ", len(data[1]))
            #if data.shape == 69:
                #np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)      # .npy 형태로 저장

            # Create sequence data, 30개씩 모으기
            #full_seq_data = []
            #for seq in range(len(data) - seq_length):
                #full_seq_data.append(data[seq:seq + seq_length])

            #full_seq_data = np.array(full_seq_data)
            #print(action, full_seq_data.shape)

        #    np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
        break