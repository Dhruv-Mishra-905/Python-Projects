import cv2 as cv
import mediapipe as mp
import numpy as np

overlay_img = cv.imread(r"C:\Users\dhruv\OneDrive\Desktop\CV\img\drac.png", cv.IMREAD_UNCHANGED)

mp_face_mesh = mp.solutions.face_mesh

def overlay_transparent(background, overlay, x, y, scale=1):
    h, w = overlay.shape[:2]
    overlay = cv.resize(overlay, (int(w * scale), int(h * scale)))

    bh, bw = background.shape[:2]

    if x >= bw or y >= bh:
        return background

    h, w = overlay.shape[:2]

    if x + w > bw:
        w = bw - x
        overlay = overlay[:, :w]

    if y + h > bh:
        h = bh - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        return background

    mask = overlay[:, :, 3:] / 255.0
    background[y:y+h, x:x+w] = (1.0 - mask) * background[y:y+h, x:x+w] + mask * overlay[:, :, :3]
    
    return background


video = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as mesh:

    while True:
        ret, frame = video.read()
        if not ret:
            break

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = mesh.process(rgb)

        if result.multi_face_landmarks:
            # Extract face bounds
            h, w, _ = frame.shape
            landmarks = result.multi_face_landmarks[0].landmark

            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            face_width = x_max - x_min
            face_height = y_max - y_min

            scale = face_width / overlay_img.shape[1] * 1.2  # adjust size

            frame = overlay_transparent(
                frame,
                overlay_img,
                x_min - int(face_width * 0.1),   # offset
                y_min - int(face_height * 0.4),
                scale
            )

        cv.imshow("WebCam", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

video.release()
cv.destroyAllWindows()
