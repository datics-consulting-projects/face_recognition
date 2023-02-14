import cv2
import numpy as np
import face_recognition
from age_detection import f_my_age
from gender_detection import f_my_gender
from race_detection import f_my_race
from emotion_detection import f_emotion_detection
from my_face_recognition import f_main

# instanciar detectores
age_detector = f_my_age.Age_Model()
gender_detector = f_my_gender.Gender_Model()
# race_detector = f_my_race.Race_Model()
# emotion_detector = f_emotion_detection.predict_emotions()
# rec_face = f_main.rec()


# ----------------------------------------------


def get_face_info(im):
    # face detection
    boxes_face = face_recognition.face_locations(im, model="hog")
    out = []
    if len(boxes_face) != 0:
        for box_face in boxes_face:
            # segmento rostro
            box_face_fc = box_face
            x0, y1, x1, y0 = box_face
            box_face = np.array([y0, x0, y1, x1])
            face_features = {
                #    "name":[],
                "age": [],
                "gender": [],
                "race": [],
                "emotion": [],
                "bbx_frontal_face": box_face
            }

            face_image = im[x0:x1, y0:y1]

            # -------------------------------------- face_recognition ---------------------------------------
            # face_features["name"] = rec_face.recognize_face2(im,[box_face_fc])[0]

            # -------------------------------------- age_detection ---------------------------------------
            age = age_detector.predict_age(face_image)
            face_features["age"] = str(round(age, 2))

            # -------------------------------------- gender_detection ---------------------------------------
            face_features["gender"] = gender_detector.predict_gender(face_image)

            # -------------------------------------- race_detection ---------------------------------------
            # face_features["race"] = race_detector.predict_race(face_image)

            # -------------------------------------- emotion_detection ---------------------------------------
            # _,emotion = emotion_detector.get_emotion(im,[box_face])
            # face_features["emotion"] = emotion[0]

            # -------------------------------------- out ---------------------------------------       
            out.append(face_features)
    else:
        face_features = {
            #    "name":[],
            "age": [],
            "gender": [],
            "race": [],
            "emotion": [],
            "bbx_frontal_face": []
        }
        out.append(face_features)
    return out


def bounding_box(out, img, anonymized=False):
    tempImg = img.copy()
    mask = np.full((img.shape[0], img.shape[1], 1), 0, dtype=np.uint8)
    for data_face in out:
        box = data_face["bbx_frontal_face"]
        if len(box) == 0:
            continue
        else:
            x0, y0, x1, y1 = box
            if anonymized:
                y0, y1 = max(0, y0), min(img.shape[0] - 1, y1)
                x0, x1 = max(0, x0), min(img.shape[1] - 1, x1)
                bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)
                tempImg[y0:y1, x0:x1] = cv2.blur(
                    tempImg[y0:y1, x0:x1],
                    (abs(x1 - x0) // bf, abs(y1 - y0) // bf)
                )
                cv2.ellipse(mask, (((x0 + x1) // 2, (y0 + y1) / 2), (abs(x1 - x0), abs(y1 - y0)), 0), 255, -1)
            else:
                img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

            '''
            thickness = 1
            fontSize = 0.5
            step = 13
            
            try:
                cv2.putText(img, "age: " +data_face["age"], (x0, y0-7), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
            try:
                cv2.putText(img, "gender: " +data_face["gender"], (x0, y0-step-10*1), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
            try:
                cv2.putText(img, "race: " +data_face["race"], (x0, y0-step-10*2), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
            try:
                cv2.putText(img, "emotion: " +data_face["emotion"], (x0, y0-step-10*3), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
            try:
                cv2.putText(img, "name: " +data_face["name"], (x0, y0-step-10*4), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0,255,0), thickness)
            except:
                pass
            '''
    if anonymized:
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(img, img, mask=mask_inv)
        img2_fg = cv2.bitwise_and(tempImg, tempImg, mask=mask)
        img = cv2.add(img1_bg, img2_fg)
    return img
