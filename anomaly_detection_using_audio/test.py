import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import python_speech_features
from sklearn import svm
from sklearn.externals import joblib

samples = 16
nFrames = 50
nCepticals = 39
flen = .2
fstr = .01

def stZCR(frame):
    """Computes zero crossing rate of frame"""
    count = len(frame)
    countZ = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return (np.float64(countZ) / np.float64(count-1.0))

def stEnergy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))

model_name = 'pre_trained_models/trained_svm_audio.xml'
classifier = joblib.load(model_name)

while(1):
    fs=44100
    duration = 5  # seconds
    myrecording = sd.rec(duration * fs, samplerate=fs, channels=2,dtype='float64')
    print "Recording Audio"
    sd.wait()
    print "Audio recording complete"


    mfcc = python_speech_features.mfcc(signal = myrecording, samplerate = fs, nfft = 23000, numcep = nCepticals, ceplifter = 0,
                                                                                            winstep = fstr, winlen = flen)

    zcr = stZCR(myrecording)
    ste = stEnergy(myrecording)

    mfcc = np.reshape(np.array(mfcc).transpose(), (1, np.array(mfcc).shape[1]*np.array(mfcc).shape[0]))
    mfcc = mfcc[:,0:25480]
    zcr = zcr.reshape((1,1))
    ste = ste.reshape((1,1))

    data = np.hstack((mfcc, zcr, ste))

    label = classifier.predict(data)

    if label == 0:
        print('door')
    if label == 1:
        print('Whistling')
    if label == 2:
        print('no')
    if label == 3:
        print('Nothing')
