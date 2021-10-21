import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model1 = load_model('models/ONEmodel5.h5')
model2 = load_model('models/TWOmodel5.h5')

actionsONE = {0:'long for', 1:'rainbow'}
actionsTWO = {0:'sincerely', 1:'if', 2:'rise', 3:'will'}
Actions = {0:'sincerely', 1:'if', 2:'rise', 3:'will', 4:'longfor', 5:'rainbow'}

seq_length = 30
seqONE, seqTWO, action_seq = [], [], []
this_action = '?'

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

        Angle = []
        right, left = 0, 0  # 손 인식 여부 확인
        RHAngle, LHAngle, PAngle = np.zeros((15,)), np.zeros((15,)), np.zeros((10,))
        RHjoint, LHjoint, Pjoint = np.zeros((21, 3)), np.zeros((21, 3)), np.zeros((33, 3))


        # Get RIGHT HAND angle info
        if results.right_hand_landmarks is not None:
            right = 1
            for i in range(21):
                RHjoint[i] = [results.right_hand_landmarks.landmark[i].x, results.right_hand_landmarks.landmark[i].y,
                              results.right_hand_landmarks.landmark[i].z]
            joint = np.array([RHjoint.flatten()])

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
                joint = np.append(joint, LHjoint.flatten())
            elif right == 0:
                joint = np.array(LHjoint.flatten())

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
            distance = []
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
            if len(Angle) == 68:    # with BOTH HANDS
                print("TWO HAND")
                angle68 = np.array([Angle], dtype=np.float32)
                d68 = np.concatenate([joint, angle68.flatten()])
                seqTWO.append(d68)

                input_data = np.expand_dims(np.array(seqTWO[-seq_length:], dtype=np.float32), axis=0)
                y_pred = model2.predict(input_data).squeeze()
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.8:
                    continue

                action = actionsTWO[i_pred]
                action_seq.append(action)
                if len(action_seq) < 3:
                    continue

                #this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

                cv2.putText(img, f'{this_action.upper()}', org=(50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, color=(255, 25, 25), thickness=2)


            elif len(Angle) == 53:
                print("ONE HAND")
                angle53 = np.array([Angle], dtype=np.float32)
                d53 = np.concatenate([joint.flatten(), angle53.flatten()])
                seqONE.append(d53)
                input_data = np.expand_dims(np.array(seqONE[-seq_length:], dtype=np.float32), axis=0)
                y_pred = model1.predict(input_data).squeeze()
                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.8:
                    continue

                action = Actions[i_pred+len(actionsTWO)]
                action_seq.append(action)
                if len(action_seq) < 2:
                    continue

                #this_action = '?'
                if action_seq[-1] == action_seq[-2] :
                    this_action = action

                cv2.putText(img, f'{this_action.upper()}', org=(50, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 25, 25), thickness=2)

        cv2.imshow('Holistic Model Detections', img)
        if cv2.waitKey(10) and 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()