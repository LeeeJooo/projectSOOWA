import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from collections import defaultdict
from itertools import chain

## 수정!! ##
CLOUD, SHINE, RAINBOW, LIKE, SNOW, FORSYTHIA, SPRING, GALAXY, ONE, TWO, ALL = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
actionsONE, actionsTWO = defaultdict(int), defaultdict(int)
Actions = []

cloudONE = {0:'white', 1:'be', 2:'here', 3:'probably', 4:'aboveONE'}
cloudTWO = {0:'cloud', 1:'cloud', 2:'above', 3:'if'}    # cloud_1: 오른손 위, cloud_2:왼손 위

shineONE = {0:'eye', 1:'light', 2:'morning'}
shineTWO = {0:'bright', 1:'morning'}

rainbowONE = {0:'wish', 1:'rainbow'}
rainbowTWO = {0:'sincerely', 1:'if', 2:'rise', 3:'will', 4:'will'}

likeONE = {0:'I', 1:'you', 2:'like'}
likeTWO = {0:'always', 1:'bright', 2:'above'}

snowONE = {0:'white', 1:'leaf', 2:'color'}
snowTWO = {0:'flower', 1:'snow_', 2:'as_', 3:'fly'}

forsythiaONE = {0:'yellow', 1:'black', 2:'shadow_', 3:'below'}
forsythiaTWO = {0:'flower', 1:'shadow_', 2:'below'}

springONE = {0:'leaf', 1:'wish', 2:'rainbow'}
springTWO = {0:'season', 1:'warm', 2:'wind', 3:'because', 4:'because', 5:'flower', 6:'together', 7:'bright', 8:'smile'}

galaxyONE = {0:'blue', 1:'color', 2:'star'}
galaxyTWO = {0:'bright', 1:'gorup'}


## 추가 !! ##
SLONE = {0:'white', 1:'be', 2:'here', 3:'probably', 4:'above',
         5:'eye', 6:'light', 7:'morning', 8:'wish', 9:'rainbow',
         10:'I', 11:'you', 12:'like', 13:'leaf', 14:'color',
         15:'yellow', 16:'black', 17:'shadow', 18:'below', 19:'blue',
         20:'star'}
KWONE = {0:'finger_heart', 1:'finger_heart', 2:'hi', 3:'hi', 4:'meosseug',
         5:'meosseug', 6:'kosseug', 7:'kosseeug', 8:'fuckyou', 9:'fuckyou',
         10:'jawV', 11:'jawV', 12:'V', 13:'V'}

SLTWO = {0:'cloud', 1:'cloud', 2:'above', 3:'if', 4:'bright',
         5:'morning', 6:'sincerely', 8:'rise', 9:'will',
         10:'will', 11:'always', 12:'flower', 13:'snow', 14:'as',
         15:'fly', 16:'shadow', 17:'below', 18:'season', 19:'warm',
         20:'wind', 21:'because', 22:'because', 23:'together', 24:'smile',
         25:'group', 26:'jewel'}
KWTWO = {0:'finger_heart', 1:'small_heart', 2:'middle_heart', 3:'big_heart',
         4:'hi', 5:'flower_cup', 6:'V', 7:'V', 8:'fuckyou'}

k=0
for v in chain(cloudTWO.values(), cloudONE.values()):
    CLOUD[k] = v
    k+=1
k=0
for v in chain(shineTWO.values(), shineONE.values()):
    SHINE[k] = v
    k+=1
k=0
for v in chain(rainbowTWO.values(), rainbowONE.values()):
    RAINBOW[k] = v
    k+=1
k=0
for v in chain(likeTWO.values(), likeONE.values()):
    LIKE[k] = v
    k+=1
k=0
for v in chain(snowTWO.values(), snowONE.values()):
    SNOW[k] = v
    k+=1
k=0
for v in chain(forsythiaTWO.values(), forsythiaONE.values()):
    FORSYTHIA[k] = v
    k+=1
k=0
for v in chain(springTWO.values(), springONE.values()):
    SPRING[k] = v
    k+=1
k=0
for v in chain(galaxyTWO.values(), galaxyONE.values()):
    GALAXY[k] = v
    k+=1

## 추가!! ##
## ALL ##
k=0
for v in chain(SLONE.values(), KWONE.values()):
    ONE[k] = v
    k+=1
k=0
for v in chain(SLTWO.values(), KWTWO.values()):
    TWO[k] = v
    k+=1
k=0
for v in chain(TWO.values(), ONE.values()):
    ALL[k] = v
    k+=1

#################### 설정 #################
actionsTWO = TWO       # TWO, cloudTWO, shineTWO, rainbowTWO, likeTWO, snowTWO, forsythiaTWO, springTWO, galaxyTWO
actionsONE = ONE       # ONE, cloudONE, shineONE, rainbowONE, likeONE, snowONE, forsythiaONE, springONE, galaxyONE
for v in ALL.values():    # ALL, CLOUD, SHINE, RAINBOW, LIKE, SNOW, FORSYTHIA, SPIRNG
    Actions.append(v)

model1 = load_model('models/ALLONE_4.h5')   # cloudONE / shineONE/ rainbowONE / likeONE / snowONE / forsythiaONE / springONE / galaxyONE
model2 = load_model('models/ALLTWO_4.h5')   # cloudTWO / shineTWO / rainbowTWO / likeTWO / snowTWO / forsythiaTWO / springTWO / galaxyTWO
###########################################

seq_length = 10
seqONE, seqTWO = [], []

# MediaPipe hands model
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, img = cap.read()
        img0 = img.copy()

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        Angle, distance = [], []
        right, left = 0, 0  # 손 인식 여부 확인
        RHAngle, LHAngle, PAngle = np.zeros((15,)), np.zeros((15,)), np.zeros((10,))
        RHjoint, LHjoint, Pjoint, Fjoint = np.zeros((21, 3)), np.zeros((21, 3)), np.zeros((33, 3)), np.zeros((468,3))


        # Get RIGHT HAND angle info
        if results.right_hand_landmarks is not None:
            right = 1
            for i in range(21):
                RHjoint[i] = [results.right_hand_landmarks.landmark[i].x, results.right_hand_landmarks.landmark[i].y,
                              results.right_hand_landmarks.landmark[i].z]

            # Compute angles between joints
            v1 = RHjoint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = RHjoint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1  # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            # Get angle using arcos of dot product
            RHAngle = np.arccos(np.einsum('nt,nt->n',
                                          v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                          v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            RHAngle = np.degrees(RHAngle)  # Convert radian to degree
            Angle = np.append(Angle, RHAngle)


        # get LEFT HAND angle info
        if results.left_hand_landmarks is not None:
            left = 1
            for i in range(21):
                LHjoint[i] = [results.left_hand_landmarks.landmark[i].x, results.left_hand_landmarks.landmark[i].y,
                              results.left_hand_landmarks.landmark[i].z]

            if right == 1:
                distance = np.append(distance, LHjoint[0] - RHjoint[0])  # # 오른손목_왼손목 간 거리

            # Compute angles between joints
            v1 = LHjoint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = LHjoint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1  # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            # Get angle using arcos of dot product
            LHAngle = np.arccos(np.einsum('nt,nt->n',
                                          v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                          v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

            LHAngle = np.degrees(LHAngle)  # Convert radian to degree
            Angle = np.append(Angle, LHAngle)


        # Get distance info between face&hand
        if results.face_landmarks is not None:
            facdidx = [10, 368, 159]
            handidx = [0, 4, 8, 12, 16, 20]

            for i in facdidx:
                Fjoint[i] = Fjoint[i] = [results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y, results.face_landmarks.landmark[i].z]

            if right == 1:  # 오른쪽손 인식
                for i in handidx:
                    distance = np.append(distance, Fjoint[368] - RHjoint[i])  # 왼쪽눈-오른쪽손
                    distance = np.append(distance, Fjoint[159] - RHjoint[i])  # 오른쪽눈-오른쪽손
                # print("right", len(distance))   # len(distance) = 36    (right or left)

            if left == 1:  # 왼쪽손 인식
                for i in handidx:
                    distance = np.append(distance, Fjoint[368] - LHjoint[i])  # 왼쪽눈-왼쪽손
                    distance = np.append(distance, Fjoint[159] - LHjoint[i])  # 오른쪽눈-왼쪽손
                # print("+ left", len(distance))  # len(distance) = 72    (right + left)
            TOP = Fjoint[10]

        # Get Arm&Face angle info AND distance info between finger&face
        if results.pose_landmarks is not None:

            for i in range(33):
                Pjoint[i] = [results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y, results.pose_landmarks.landmark[i].z]

            ## angle info ##
            # Compute angles between joints
            v1 = Pjoint[[11,13,15,17,19,21, 12,14,16,18,20,22, 15,7,15,8, 17,7,17,8, 19,7,19,8, 21,7,21,8, 16,7,16,8, 18,7,18,8, 20,7,20,8, 22,7,22,8],:] # Parent joint
            v2 = Pjoint[[13,15,17,19,21,11, 14,16,18,20,22,12, 7,9,8,10, 7,9,8,10, 7,9,8,10, 7,9,8,10, 7,9,8,10, 7,9,8,10, 7,9,8,10, 7,9,8,10],:] # Child joint
            v = v2 - v1 # [20,3]
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            # Get angle using arcos of dot product
            PAngle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,3,4, 6,7,8,9,10,  12,14,16,18,20,22,24,26, 28,30,32,34,36,38,40,42],:],
                v[[1,2,3,4,5, 7,8,9,10,11, 13,15,17,19,21,23,25,27, 29,31,33,35,37,39,41,43],:])) # [26,]
            PAngle = np.degrees(PAngle) # Convert radian to degree
            Angle = np.append(Angle, PAngle)

            ## distance info ##
            dis1 = Pjoint[[17,21,17,21, 18,22,18,22],:]
            dis2 = Pjoint[[9,9,0,0,     10,10,0,0],:]
            dis = dis1-dis2
            for i in range(4):
                distance = np.append(distance, dis1[2*i]-dis[2*i+1])    #[12,]
            Angle = np.append(Angle, distance)

            # Draw Landmark
            # Right Hand
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            # Left Hand
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            # Pose
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Inference gesture
            if len(Angle) == 143:    # with BOTH HANDS
                print("TWO HAND")
                angleTWO = np.array([Angle], dtype=np.float32)
                angleTWO = np.array(angleTWO.flatten())
                seqTWO.append(angleTWO)

                input_data = np.expand_dims(np.array(seqTWO[-seq_length:], dtype=np.float32), axis=0)
                y_pred = model2.predict(input_data).squeeze()
                i_pred = int(np.argmax(y_pred))

                conf = y_pred[i_pred]


                if conf < 0.99:
                    continue

                action = actionsTWO[i_pred]

                cv2.putText(img, f'{action.upper()}', org=(50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(51, 000, 204), thickness=2)


            elif len(Angle) == 89:
                print("ONE HAND")
                angleONE = np.array([Angle], dtype=np.float32)
                angleONE = np.array(angleONE.flatten())
                seqONE.append(angleONE)
                input_data = np.expand_dims(np.array(seqONE[-seq_length:], dtype=np.float32), axis=0)
                y_pred = model1.predict(input_data).squeeze()
                i_pred = int(np.argmax(y_pred))

                conf = y_pred[i_pred]

                if conf < 0.99:
                    continue

                action = Actions[i_pred+len(actionsTWO)]
                cv2.putText(img, f'{action.upper()}', org=(50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 25, 25), thickness=2)

        cv2.imshow('Holistic Model Detections', img)
        if cv2.waitKey(10) and 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()