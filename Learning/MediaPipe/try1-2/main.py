import cv2 as cv
import mediapipe as mp
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_DIR = os.path.join(BASE_DIR, "xmls")

face_cas  = cv.CascadeClassifier(os.path.join(XML_DIR, "haarcascade_frontalface_default.xml"))
smile_cas = cv.CascadeClassifier(os.path.join(XML_DIR, "haarcascade_smile.xml"))
eye_cas   = cv.CascadeClassifier(os.path.join(XML_DIR, "haarcascade_eye.xml"))

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1)

drawing_spec_hand = mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1)
mp_hand_mesh = mp.solutions.hands

video = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as face_mesh:
    with mp_hand_mesh.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

        while True:
            ret, frame = video.read()
            if not ret:
                break

            

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            detact_Face = face_cas.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in detact_Face:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)

                roi_gray  = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                detact_Eye = eye_cas.detectMultiScale(roi_gray, 1.1, 10)
                if len(detact_Eye) > 0:
                    cv.putText(frame, "Eye detected", (x, y-50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

                detact_smile = smile_cas.detectMultiScale(roi_gray, 1.7, 20)
                if len(detact_smile) > 0:
                    cv.putText(frame, "Smile detected", (x, y-20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = face_mesh.process(rgb)
            rgb.flags.writeable = True

            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,  
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )
            img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img.flags.writeable = False
            result = hands.process(img)
            img.flags.writeable = True
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hand_mesh.HAND_CONNECTIONS
                    )

            frame = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            frame=cv.flip(frame,1)
            cv.imshow("WebCam", frame)
            k = cv.waitKey(10)
            if k == ord('q'):
                break

video.release()
cv.destroyAllWindows()
