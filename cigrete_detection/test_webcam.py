import numpy as np
import cv2
import os
import time

face_cascade = cv2.CascadeClassifier('pre_trained_models/haarcascade_frontalface_default.xml')

svm = cv2.ml.SVM_load('pre_trained_models/trained_svm.xml')

thresh = 150

path = 'real_time_test'

#hog descriptor
winSize = (32,32)
blockSize = (32,32)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = -1
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradients = False
 
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,
                        nbins,derivAperture,winSigma,histogramNormType,
                        L2HysThreshold,gammaCorrection,nlevels, signedGradients)
#hog descriptor

cap = cv2.VideoCapture(2)

while 1:
    ret, img = cap.read()
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in faces:
        y = int(y+0.6*h)
        h = int(h*0.7)
        x = int(x+0.1*w)
        w = int(w-0.2*w)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color = img[y:y+h, x:x+w]
        roi = cv2.resize(roi_color, (32, 32))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #roi = cv2.threshold(roi, thresh, 255, cv2.THRESH_BINARY)[1]
        features = hog.compute(roi)
        features = features.transpose()
        label = svm.predict(features)[1].ravel()
        if label == 0:
            print('Smoking')
        else:
            print('Nothing')

        cv2.imwrite(os.path.join(path , 'test%d(%f).jpg')% (x,label), roi_color)

        time.sleep(0.5)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
