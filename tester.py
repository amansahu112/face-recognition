import cv2
import os
import numpy as np
import project4 as fr

test_img=cv2.imread('resources/download.jpg')
faces_detected,gray_img=fr.faceDetection(test_img)
print('face_detected:',faces_detected)

# faces,faceID=fr.labels_for_training_data('images')
# face_recognizer=fr.tarin_classifier(faces,faceID)
# face_recognizer.save('trainingData.yml')

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.yml')

name={0:"SHIRLEY",1:"PRIYANKA"}

for face in faces_detected:
    (x, y, w, h)=face
    roi_gray=gray_img[y:y+h,x:x+w]
    label,confidence=face_recognizer.predict(roi_gray)
    print("CONFIDENCE:",confidence)
    print("LABEL:",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    if (confidence>37):
        continue
    else:
        fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow('image',resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
