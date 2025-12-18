import cv2 as cv
import numpy as np
from PIL import Image
import os

def create_ds(f_id,name):
    web = cv.VideoCapture(0)
    web.set(3,640)
    web.set(4,480)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    XML_DIR = os.path.join(BASE_DIR, "xmls")

    face_cas  = cv.CascadeClassifier(os.path.join(XML_DIR, "haarcascade_frontalface_default.xml"))

    f_dir = 'dataset'
    f_name = name
    
    
    path=os.path.join(f_dir,f_name)
    if not os.path.exists(path):
        os.makedirs(path)
        
    counter=0
    while True:
        ret,frame=web.read()
        frame=cv.flip(frame,1)
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        multiFaces=face_cas.detectMultiScale(gray,1.3,5)
        for x,y,w,h in multiFaces:
            cv.rectangle(frame,(x,y),(x+w , y+h),(0,255,00), 2 )
            counter +=1
            cv.imwrite("{}/{}.{}.{}{}".format(path,counter,f_id,name,".jpg"),gray[y:y+h,x:x+w])
            
            cv.imshow("webCam",frame)
        k=cv.waitKey(10)
        if k==ord('q'):
            break
        elif counter>40:
            break
    web.release()
    cv.destroyAllWindows()

create_ds(1,"Dhruv")

def train():
    database='dataset'
    img_dir=[x[0] for x in os.walk(database)][1:]
    
    recog = cv.face.LBPHFaceRecognizer_create()
    detect = cv.CascadeClassifier(r"C:\Users\dhruv\OneDrive\Desktop\CV\xmls\haarcascade_frontalface_default.xml")
    
    faceSamples = []
    ids = []
    
    for path in img_dir:
        imgPaths = [os.path.join(path, f) for f in os.listdir(path)]
        for imgPath in imgPaths:
            PIL_img = Image.open(imgPath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            
            filename = os.path.basename(imgPath)       # "1.1.Dhruv.jpg"
            id = int(filename.split('.')[1])           # f_id
            
            faces = detect.detectMultiScale(img_numpy)
            for x, y, w, h in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)
    
    recog.train(faceSamples, np.array(ids))
    recog.write('trainer.yml')
    
    print(f"\n[INFO] {len(np.unique(ids))} faces trained. Exiting.")
    return len(np.unique(ids))

train()

def overlap(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2

    end_x = min(x1 + w1, x2 + w2)
    start_x = max(x1, x2)
    width = end_x - start_x

    end_y = min(y1 + h1, y2 + h2)
    start_y = max(y1, y2)
    height = end_y - start_y

    if width <= 0 or height <= 0:
        return 0

    return (width * height) / min(w1 * h1, w2 * h2)


def recog():
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    faceCasc = cv.CascadeClassifier(r"C:\Users\dhruv\OneDrive\Desktop\CV\xmls\haarcascade_frontalface_default.xml")

    font = cv.FONT_HERSHEY_SIMPLEX
    last_label = None
    face_count = 0

    web = cv.VideoCapture(0)
    web.set(3, 640)
    web.set(4, 480)

    minW = int(0.1 * web.get(3))
    minH = int(0.1 * web.get(4))

    while True:
        ret, frame = web.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        faces = faceCasc.detectMultiScale(gray, 1.3, 5, minSize=(minW, minH))
        final_faces = []
        for f in faces:
            keep = True
            for ff in final_faces:
                if overlap(f, ff) > 0.70:
                    keep = False
                    break
            if keep:
                final_faces.append(f)

        for (x, y, w, h) in final_faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            predicted_id, confi = recognizer.predict(gray[y:y + h, x:x + w])
            confid_pct = f"{round(100 - confi)}%"

            if confi < 70:
                label = str(predicted_id)
            else:
                label = "unknown"

            cv.putText(frame, label, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv.putText(frame, confid_pct, (x + 5, y + h - 5), font, 1, (255, 255, 255), 2)

        cv.imshow("webCam", frame)
        if cv.waitKey(10) == ord('q'):
            break

    web.release()
    cv.destroyAllWindows()
recog()