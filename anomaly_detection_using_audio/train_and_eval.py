import numpy as np
import scipy.io.wavfile as wav
import python_speech_features
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
#0=shouting
#1=whistling
#2=door_slamming

samples = 16
nFrames = 50
nCepticals = 13
flen = .2
fstr = .01

directory_whistle = "data/whistle"
directory_shout = "data/scream"
directory_door_bang = "data/door_slam"
directory_no = "data/negative_samples"

def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return (np.float64(countZ) / np.float64(count-1.0))

def stEnergy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))

mfcc1 = []
mfcc2 = []
mfcc3 = []
mfcc4 = []

zcr1 = []
zcr2 = []
zcr3 = []
zcr4 = []

ste1 = []
ste2 = []
ste3 = []
ste4 = []


files_shout = [f for f in os.listdir(directory_shout) if f.endswith(".wav")]
files_whistle = [f for f in os.listdir(directory_whistle) if f.endswith(".wav")]
files_banging = [f for f in os.listdir(directory_door_bang) if f.endswith(".wav")]
files_no = [f for f in os.listdir(directory_no) if f.endswith(".wav")]

for  q in range (len(files_shout)):
	fs,data = wav.read(directory_shout+'/'+files_shout[q])
	if (data.shape[0]<5*44100):
		data = np.hstack((data,np.zeros((5*44100-len(data)))))

	mfcc = python_speech_features.mfcc(signal = data, samplerate = fs, nfft = 23000, numcep = nCepticals, ceplifter = 0,
										winstep = fstr, winlen = flen)
    
	zcr = stZCR(data)
	zcr1.append(zcr)

	ste = stEnergy(data)
	ste1.append(ste)

	mfcc1.append(mfcc)

Big_Matrix = np.reshape(np.array(mfcc1[0]).transpose(), (1, np.array(mfcc1[0]).shape[1]*np.array(mfcc1[0]).shape[0]))	

for h in range(1,len(mfcc1)):
	t = np.reshape(np.array(mfcc1[h]).transpose(), (1, np.array(mfcc1[h]).shape[1]*np.array(mfcc1[h]).shape[0]))
	if (t.shape[1]>Big_Matrix.shape[1]):
		t=t[:,np.arange(Big_Matrix.shape[1])]		
	if (t.shape[1]<Big_Matrix.shape[1]):
		Big_Matrix=Big_Matrix[:,np.arange(t.shape[1])]	
	
	Big_Matrix = np.concatenate ((Big_Matrix,t))

lablezz= np.zeros((1,len(mfcc1)))  # """""""""         lable of zero is for shouting                   """"""""" 

print (Big_Matrix.shape)
print(lablezz.shape)


for  q in range (len(files_whistle)):
	fs,data = wav.read(directory_whistle+'/'+files_whistle[q])
	if (data.shape[0]<5*44100):
		data = np. hstack ((data,np.zeros((5*44100 - len(data)))))

	mfcc = python_speech_features.mfcc(signal = data, samplerate = fs,nfft = 23000,numcep = nCepticals,ceplifter = 0,
											winstep = fstr,winlen = flen)

	zcr = stZCR(data)
	zcr2.append(zcr)

	ste = stEnergy(data)
	ste2.append(ste)
		
	mfcc2.append(mfcc)

for h in range(len(mfcc2)):
	t = np.reshape(np.array(mfcc2[h]).transpose(), (1, np.array(mfcc2[h]).shape[1]*np.array(mfcc2[h]).shape[0]))
	if (t.shape[1]>Big_Matrix.shape[1]):
		t=t[:,np.arange(Big_Matrix.shape[1])]
	if (t.shape[1]<Big_Matrix.shape[1]):
		Big_Matrix=Big_Matrix[:,np.arange(t.shape[1])]

	Big_Matrix = np. concatenate((Big_Matrix,t))	
	
	
lablezz= np. hstack ((lablezz,np.ones((1,len(mfcc2))))) # """""""""         lable of one is for whistling                """"""""" 

print (Big_Matrix.shape)
print(lablezz.shape)

for  q in range (len(files_banging)):
	fs,data = wav.read(directory_door_bang+'/'+files_banging[q])
	if (data.shape[0]<5*44100):
		data = np.hstack((data,np.zeros((5*44100-len(data),2))))

	mfcc = python_speech_features.mfcc(signal = data, samplerate = fs,nfft = 23000,numcep = nCepticals,ceplifter = 0,
											winstep = fstr,winlen = flen)

	
	zcr = stZCR(data)
	zcr3.append(zcr)

	ste = stEnergy(data)
	ste2.append(ste)
	
	mfcc3.append(mfcc)

for h in range(len(mfcc3)):
	t = np.reshape(np.array(mfcc3[h]).transpose(), (1, np.array(mfcc3[h]).shape[1]*np.array(mfcc3[h]).shape[0]))
	if (t.shape[1]>Big_Matrix.shape[1]):
		t=t[:,np.arange(Big_Matrix.shape[1])]
	if (t.shape[1]<Big_Matrix.shape[1]):
		Big_Matrix=Big_Matrix[:,np.arange(t.shape[1])]

	Big_Matrix = np. concatenate((Big_Matrix,t))	

lablezz= np. hstack ((lablezz,np.full((1, len(mfcc3)), 2)))# """""""""         lable of two is for door banging                """"""""" 

print (Big_Matrix.shape)
print(lablezz.shape)

for  q in range (len(files_no)):
	fs,data = wav.read(directory_no+'/'+files_no[q])
	if (data.shape[0]<5*44100):
		data = np.hstack((data,np.zeros((5*44100-len(data)))))
	mfcc = python_speech_features.mfcc(signal = data, samplerate = fs,nfft = 23000,numcep = nCepticals,ceplifter = 0,
											winstep = fstr,winlen = flen)
	zcr = stZCR(data)
	zcr4.append(zcr)

	ste = stEnergy(data)
	ste4.append(ste)
	
	mfcc4.append(mfcc)

for h in range(len(mfcc4)):
	t = np.reshape(np.array(mfcc4[h]).transpose(), (1, np.array(mfcc4[h]).shape[1]*np.array(mfcc4[h]).shape[0]))
	if (t.shape[1]>Big_Matrix.shape[1]):
		t=t[:,np.arange(Big_Matrix.shape[1])]
	if (t.shape[1]<Big_Matrix.shape[1]):
		Big_Matrix=Big_Matrix[:,np.arange(t.shape[1])]

	Big_Matrix = np. concatenate((Big_Matrix,t))	

lablezz= np. hstack ((lablezz,np.full((1, len(mfcc4)), 3)))# """""""""         lable of two is for no               """"""""" 

print (Big_Matrix.shape)
print(lablezz.shape)


zcr = np.concatenate((zcr1,zcr2,zcr3,zcr4),axis=0)
zcr = zcr.reshape((len(zcr),1))

ste = np.concatenate((ste1,ste2,ste3,ste4),axis=0)
ste = ste.reshape((len(ste),1))

Big_Matrix = np.hstack((Big_Matrix, zcr, ste))
print (Big_Matrix.shape)


label = lablezz.transpose()

X_train, X_test, y_train, y_test = train_test_split(Big_Matrix, label, test_size=0.3, random_state=3)
classifier = svm.SVC(max_iter = 100000, kernel = 'linear',probability=True)

y = y_train.ravel()
y_train = np.array(y).astype(int)

print(classifier.fit(X_train,y_train))
prediction = classifier.predict(X_test)

C = confusion_matrix(y_test, prediction)
print(C)
accuracy = (float(C.trace())/float(len(y_test)))*100
print(accuracy)

model_name = 'trained_svm_audio.xml'
joblib.dump(classifier, model_name)
