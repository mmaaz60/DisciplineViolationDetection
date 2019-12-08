#class_0 = cigarete
#class_1 = ballpoint
#class_2 = simple_face
import cv2
import numpy as np
import glob
import os

class_0 = [cv2.imread(file) for file in glob.glob("class_0/*.jpg")]
class_1 = [cv2.imread(file) for file in glob.glob("class_1/*.jpg")]
#class_2 = [cv2.imread(file) for file in glob.glob("class_2/*.jpg")]

path_0 = 'class_0_classifier_data' # Path to directory containing cigrete samples
path_1 = 'class_1_classifier_data' # Path to directory containing non-cigrete samples
#path_2 = 'class_2_classifier_data'

thresh = 150

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
hog_descriptors_class_0 = []
labels_class_0 = []
hog_descriptors_class_1 = []
labels_class_1 = []
hog_descriptors_class_2 = []
labels_class_2 = []

#hog descriptor

#svm
def svmInit(C=12.5, gamma=0.50625):
  model = cv2.ml.SVM_create()
  model.setGamma(gamma)
  model.setC(C)
  model.setKernel(cv2.ml.SVM_RBF)
  model.setType(cv2.ml.SVM_C_SVC)
  
  return model

def svmTrain(model, samples, responses):
  model.train(samples, cv2.ml.ROW_SAMPLE, responses)
  return model

def svmPredict(model, samples):
  return model.predict(samples)[1].ravel()

def svmEvaluate(model, samples, labels):
    predictions = svmPredict(model, samples)
    #accuracy = (labels == predictions).mean()
    #print('Percentage Accuracy: %.2f %%' % (accuracy*100))
    print(predictions)
    confusion = np.zeros((3, 3), np.int32)
    for i, j in zip(labels, predictions):
        confusion[int(i), int(j)] += 1
    print('confusion matrix:')
    print(confusion)

    accuracy = float(confusion[0,0]+confusion[1,1]+confusion[2,2])/float(len(labels))
    print('Percentage Accuracy: %.2f %%' % (accuracy*100))

    return confusion

#svm

print('Calculating HoG descriptor for class_0')
for i in range(0,len(class_0)):
    img = class_0[i]
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(os.path.join(path_0 , 'pos%d.jpg')% i, img)
    hog_descriptors_class_0.append(hog.compute(img))
    labels_class_0.append([0])

print('Calculating HoG descriptor for class_1')
for i in range(0,len(class_1)):
    img = class_1[i]
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(os.path.join(path_1 , 'pos%d.jpg')% i, img)
    hog_descriptors_class_1.append(hog.compute(img))
    labels_class_1.append([1])

##print('Calculating HoG descriptor for class_2')
##for i in range(0,len(class_2)):
##    img = class_2[i]
##    img = cv2.resize(img, (32, 32))
##    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##    #img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
##    cv2.imwrite(os.path.join(path_2 , 'pos%d.jpg')% i, img)
##    hog_descriptors_class_2.append(hog.compute(img))
##    labels_class_2.append([2])
    
#training and test set descriptors from class_0 images
class_0_train = int(0.7*len(labels_class_0))
hog_class_0_train, hog_class_0_test = np.split(hog_descriptors_class_0, [class_0_train])
labels_class_0_train, labels_class_0_test = np.split(labels_class_0, [class_0_train])


#training and test set descriptors from class_1 images
class_1_train = int(0.7*len(labels_class_1))
hog_class_1_train, hog_class_1_test = np.split(hog_descriptors_class_1, [class_1_train])
labels_class_1_train, labels_class_1_test = np.split(labels_class_1, [class_1_train])

#training and test set descriptors from class_2 images
##class_2_train = int(0.7*len(labels_class_2))
##hog_class_2_train, hog_class_2_test = np.split(hog_descriptors_class_2, [class_2_train])
##labels_class_2_train, labels_class_2_test = np.split(labels_class_2, [class_2_train])

#data for training
training_descriptors = np.concatenate((hog_class_0_train, hog_class_1_train), axis = 0)
training_labels = np.concatenate((labels_class_0_train, labels_class_1_train), axis = 0)

#data for testing
testing_descriptors = np.concatenate((hog_class_0_test, hog_class_1_test), axis = 0)
testing_labels = np.concatenate((labels_class_0_test, labels_class_1_test), axis = 0)

#training SVM    
model = svmInit()
svmTrain(model, training_descriptors, training_labels)

#testing SVM
confusion_matrix = svmEvaluate(model, testing_descriptors, testing_labels)

model.save('trained_svm.xml')


##cv2.waitKey(0)
##cv2.destroyAllWindows()
