import mediapipe as mp
import cv2 
import numpy as np
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic

df = pd.read_csv("coords.csv")
x = df.drop('class', axis=1)
y = df['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =0.3, random_state=1234)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
# pipelines ={
#     'lr':make_pipeline(StandardScaler(), LogisticRegression()),
#     'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
#     'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
#     'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
# }   
# fit_models = {}
# for algo, pipeline in pipelines.items():
#     model = pipeline.fit(x_train, y_train)
#     fit_models[algo] = model

# print(fit_models['rf'].predict(x_test))
# for algo, pipeline in fit_models.items():
#     yhat = model.predict(x_test)
#     print(algo, accuracy_score(y_test, yhat))
# with open('body_language.pkl', 'wb') as f:
#     pickle.dump(fit_models['rf'], f)

with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1),
                                    )

        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                     )

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                   mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )
        try:
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            row = pose_row + face_row
            x =pd.DataFrame([row])
            body_language_class = model.predict(x)[0]
            body_language_prob = model.predict_proba(x)[0]
            print(body_language_class, body_language_prob)
        except:
            pass
        cv2.imshow("Raw camera feeback", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
