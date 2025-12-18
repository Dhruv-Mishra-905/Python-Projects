import cv2 as cv
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1)
mp_hand_mesh = mp.solutions.hands

video = cv.VideoCapture(0)

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
            print("Webcam not working")
            break

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
        cv.imshow("WebCam", frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

video.release()
cv.destroyAllWindows()
