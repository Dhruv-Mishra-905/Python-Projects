import cv2 as cv
import mediapipe as mp
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_DIR = os.path.join(BASE_DIR, "xmls")

face_cas  = cv.CascadeClassifier(os.path.join(XML_DIR, "haarcascade_frontalface_default.xml"))

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1)

video = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

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

        cv.imshow("WebCam", frame)
        k = cv.waitKey(10)
        if k == ord('q'):
            break

video.release()
cv.destroyAllWindows()
