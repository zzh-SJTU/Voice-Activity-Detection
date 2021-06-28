# -*- coding: utf-8 -*-
"""
Created on Wed May 12 23:24:50 2021

@author: zzh
"""
from python_speech_features import mfcc
from python_speech_features import logfbank
from sklearn.mixture import GaussianMixture
import scipy.io.wavfile as wav
import os
import wave
import pylab as pl
import numpy as np
from sklearn import svm
from scipy import signal
from pathlib import Path
from matplotlib import pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft
from sklearn import metrics
from pickle import dump
from pickle import load
import pickle
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
def save_variable(v,filename):
  f=open(filename,'wb')
  pickle.dump(v,f)
  f.close()
  return filename

def load_variavle(filename):
  f=open(filename,'rb')
  r=pickle.load(f)
  f.close()
  return r

def compute_eer(target_scores, nontarget_scores):
    """Calculate EER following the same way as in Kaldi.

    Args:
        target_scores (array-like): sequence of scores where the
                                    label is the target class
        nontarget_scores (array-like): sequence of scores where the
                                    label is the non-target class
    Returns:
        eer (float): equal error rate
        threshold (float): the value where the target error rate
                           (the proportion of target_scores below
                           threshold) is equal to the non-target
                           error rate (the proportion of nontarget_scores
                           above threshold)
    """
    assert len(target_scores) != 0 and len(nontarget_scores) != 0
    tgt_scores = sorted(target_scores)
    nontgt_scores = sorted(nontarget_scores)

    target_size = float(len(tgt_scores))
    nontarget_size = len(nontgt_scores)
    target_position = 0
    for target_position, tgt_score in enumerate(tgt_scores[:-1]):
        nontarget_n = nontarget_size * target_position / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontgt_scores[nontarget_position] < tgt_score:
            break
    threshold = tgt_scores[target_position]
    eer = target_position / target_size
    return eer, threshold


def get_metrics(prediction, label):
    """Calculate several metrics for a binary classification task.

    Args:
        prediction (array-like): sequence of probabilities
            e.g. [0.1, 0.4, 0.35, 0.8]
        labels (array-like): sequence of class labels (0 or 1)
            e.g. [0, 0, 1, 1]
    Returns:
        auc: area-under-curve
        eer: equal error rate
    """  # noqa: H405, E261
    assert len(prediction) == len(label), (len(prediction), len(label))
    fpr, tpr, thresholds = metrics.roc_curve(label, prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # from scipy.optimize import brentq
    # from scipy.interpolate import interp1d
    # fnr = 1 - tpr
    # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    eer, thres = compute_eer(
        [pred for i, pred in enumerate(prediction) if label[i] == 1],
        [pred for i, pred in enumerate(prediction) if label[i] == 0],
    )
    return auc, eer


def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax,Ymin,Ymax):
    plt.hist(myList,100)
    plt.xlabel(Xlabel)
    plt.xlim(Xmin,Xmax)
    plt.ylabel(Ylabel)
    plt.ylim(Ymin,Ymax)
    plt.title(Title)
    plt.show()
def parse_vad_label(line, frame_size: float = 0.032, frame_shift: float = 0.008):
    """Parse VAD information in each line, and convert it to frame-wise VAD label.
       将标签文件转换成 [0, ..., 0, 1, ..., 1, 0, ..., 0, 1, ..., 1]的形式
    Args:
        line (str): e.g. "0.2,3.11 3.48,10.51 10.52,11.02"
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
    Returns:
        frames (List[int]): frame-wise VAD label

    Examples:
        >>> label = parse_vad_label("0.3,0.5 0.7,0.9")
        [0, ..., 0, 1, ..., 1, 0, ..., 0, 1, ..., 1]
        >>> print(len(label))
        110

    NOTE: The output label length may vary according to the last timestamp in `line`,
    which may not correspond to the real duration of that sample.

    For example, if an audio sample contains 1-sec silence at the end, the resulting
    VAD label will be approximately 1-sec shorter than the sample duration.

    Thus, you need to pad zeros manually to the end of each label to match the number
    of frames in the feature. E.g.:
        >>> feature = extract_feature(audio)    # frames: 320
        >>> frames = feature.shape[1]           # here assumes the frame dimention is 1
        >>> label = parse_vad_label(vad_line)   # length: 210
        >>> import numpy as np
        >>> label_pad = np.pad(label, (0, np.maximum(frames - len(label), 0)))[:frames]
    """
    frame2time = lambda n: n * frame_shift + frame_size / 2
    frames = []
    frame_n = 0
    for time_pairs in line.split():
        start, end = map(float, time_pairs.split(","))
        assert end > start, (start, end)
        while frame2time(frame_n) < start:
            frames.append(0)
            frame_n += 1
        while frame2time(frame_n) <= end:
            frames.append(1)
            frame_n += 1
    return frames


def prediction_to_vad_label(
    prediction,
    frame_size: float = 0.032,
    frame_shift: float = 0.008,
    threshold: float = 0.5,
):
    """Convert model prediction to VAD labels.
         将预测的list（每一帧的概率）转化为时刻到时刻的判断
    Args:
        prediction (List[float]): predicted speech activity of each **frame** in one sample
            e.g. [0.01, 0.03, 0.48, 0.66, 0.89, 0.87, ..., 0.72, 0.55, 0.20, 0.18, 0.07]
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
        threshold (float): prediction values that are higher than `threshold` are set to 1,
                            and those lower than or equal to `threshold` are set to 0
    Returns:
        vad_label (str): converted VAD label
            e.g. "0.31,2.56 2.6,3.89 4.62,7.99 8.85,11.06"

    NOTE: Each frame is converted to the timestamp according to its center time point.
    Thus the converted labels may not exactly coincide with the original VAD label, depending
    on the specified `frame_size` and `frame_shift`.
    See the following exmaple for more detailed explanation.

    Examples:
        >>> label = parse_vad_label("0.31,0.52 0.75,0.92")
        >>> prediction_to_vad_label(label)
        '0.31,0.53 0.75,0.92'
    """
    frame2time = lambda n: n * frame_shift + frame_size / 2
    speech_frames = []
    prev_state = False
    start, end = 0, 0
    end_prediction = len(prediction) - 1
    for i, pred in enumerate(prediction):
        state = pred > threshold
        if not prev_state and state:
            # 0 -> 1
            start = i
        elif not state and prev_state:
            # 1 -> 0
            end = i
            speech_frames.append(
                "{:.2f},{:.2f}".format(frame2time(start), frame2time(end))
            )
        elif i == end_prediction and state:
            # 1 -> 1 (end)
            end = i
            speech_frames.append(
                "{:.2f},{:.2f}".format(frame2time(start), frame2time(end))
            )
        prev_state = state
    return " ".join(speech_frames)

def modify(pred):
    temp=np.zeros(len(pred))
    for index2 in range(len(pred)):
        temp[index2]=pred[index2]
    for index,item in enumerate(pred):
        if index>16 and index<len(pred)-13:
            a=temp[index-15:index+15].mean()
            if a<0.5:
                pred[index]=0
            else: pred[index]=1
        if index<10:
            pred[index]=0
    return pred
##############################################
# Examples of how to use the above functions #
##############################################
def read_label_from_file(
    path="data/dev_label.txt", frame_size: float = 0.032, frame_shift: float = 0.008
):
    """Read VAD information of all samples, and convert into
    frame-wise labels (not padded yet).

    Args:
        path (str): Path to the VAD label file.
        frame_size (float): frame size (in seconds) that is used when
                            extarcting spectral features
        frame_shift (float): frame shift / hop length (in seconds) that
                            is used when extarcting spectral features
    Returns:
        data (dict): Dictionary storing the frame-wise VAD
                    information of each sample.
            e.g. {
                "1031-133220-0062": [0, 0, 0, 0, ... ],
                "1031-133220-0091": [0, 0, 0, 0, ... ],
                ...
            }
    """
    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.strip().split(maxsplit=1)
            if len(sps) == 1:
                print(f'Error happened with path="{path}", id="{sps[0]}", value=""')
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = parse_vad_label(v, frame_size=frame_size, frame_shift=frame_shift)
    return data
filelist=os.listdir("D:\\vad\\wavs\\train")
filelist1=os.listdir("D:\\vad\\wavs\\dev")
filelist2=os.listdir("D:\\vad\\wavs\\test")
total_energy_of_speech=0
total_energy_nonspeech=0
frame_of_zero=0
frame_of_one=0
accu_list=[]
totalpredictlist=[]
totallabel=[]
feat_list_0 = []
feat_list_1 = []
gmm0 = GaussianMixture(n_components=30,tol=0.0001,max_iter=10000, random_state=0)
gmm1 = GaussianMixture(n_components=30,tol=0.0001,max_iter=10000, random_state=0)
#定义GMM模型
dic_label=read_label_from_file(
    path=r"D:/vad/data/train_label.txt", frame_size= 0.032, frame_shift= 0.008
)
dic_label1=read_label_from_file(
    path=r"D:/vad/data/dev_label.txt", frame_size= 0.032, frame_shift= 0.008
)

for index,file in enumerate(filelist):
    if(index==2000):break
    if(index % 50==1):
        print(index)
    if(file[0]=='.'):
            filename=file[2:-4]
    else: filename=file[0:-4]
    if(file[0]=='.'):
        f = wave.open(r"D:\vad\wavs\train\\"+file[2:], "rb")
    else:  f = wave.open(r"D:\vad\wavs\train\\"+file, "rb")
    if(file[0]=='.'):
        fsize = os.path.getsize("D:\\vad\\wavs\\train"+'\\'+file[2:])
    else: fsize = os.path.getsize("D:\\vad\\wavs\\train"+'\\'+file)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data = wave_data /7250
    b,a=signal.butter(6,0.008,'highpass')
    wave_data=signal.filtfilt(b,a,wave_data)
    dic_test=dic_label[filename]
    counter=0
    for index1 in range(1,len(dic_test)):
        data_frame=wave_data[counter:counter+512]
        mfcc_feat = mfcc(data_frame,framerate)
        fbank_feat = logfbank(data_frame,framerate)
        mfcc_feat=mfcc_feat.reshape(26)
        fbank_feat=fbank_feat.reshape(52)
        feat_frame=np.append(mfcc_feat,fbank_feat)
        #特征提取与合并
        if dic_test[index1]==0:
            feat_list_0 .append(feat_frame)
        else:
            feat_list_1 .append(feat_frame)
        counter=counter+128
print('start to fit')
filename0 = save_variable(feat_list_0, 'feat_of_0.txt')
filename1 = save_variable(feat_list_1, 'feat_of_1.txt')

#feat_list_0 = load_variavle('feat_of_0.txt')
#feat_list_1 = load_variavle('feat_of_1.txt')
print('Finish store data, start to fit')
#模型拟合
gmm0.fit(feat_list_0)
#dump(gmm0, open('gmm0.pkl', 'wb'))
gmm1.fit(feat_list_1)  
#dump(gmm1, open('gmm1.pkl', 'wb'))
print('Finish Training, start to test on dev') 
#gmm0 = load(open('gmm0.pkl', 'rb'))
#gmm1 = load(open('gmm1.pkl', 'rb'))
for index,file in enumerate(filelist1):
    if(index % 50==1):
        print(index)
    if(file[0]=='.'):
            filename=file[2:-4]
    else: filename=file[0:-4]
    if(file[0]=='.'):
        f = wave.open(r"D:\vad\wavs\dev\\"+file[2:], "rb")
    else:  f = wave.open(r"D:\vad\wavs\dev\\"+file, "rb")
    if(file[0]=='.'):
        fsize = os.path.getsize("D:\\vad\\wavs\\dev"+'\\'+file[2:])
    else: fsize = os.path.getsize("D:\\vad\\wavs\\dev"+'\\'+file)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data=wave_data/7250
    b,a=signal.butter(6,0.008,'highpass')
    wave_data=signal.filtfilt(b,a,wave_data)
    dic_test=dic_label1[filename]
    totallabel.extend(dic_test)
    predictlist=np.zeros(len(dic_test))
    counter=0
    for index1 in range(1,len(dic_test)):
        data_frame=wave_data[counter:counter+512]
        mfcc_feat = mfcc(data_frame,framerate)
        fbank_feat = logfbank(data_frame,framerate)
        mfcc_feat=mfcc_feat.reshape(26)
        fbank_feat=fbank_feat.reshape(52)
        feat_frame=np.append(mfcc_feat,fbank_feat)
        score1=gmm0.score(feat_frame.reshape(1,-1))
        score2=gmm1.score(feat_frame.reshape(1,-1))
        #每一帧的预测
        counter+=128
        if score1>score2:
            predictlist[index1]=0
        else:
            predictlist[index1]=1
    same=0
    predictlist=modify(predictlist)
    for i in range(len(dic_test)):
        if predictlist[i]==dic_test[i]:
            same+=1
    predictlist=predictlist.tolist()
    totalpredictlist.extend(predictlist)
    accu=same/len(dic_test)
    accu_list.append(accu)
    if(index%100==0):
        print(accu)
print('Performance on the dev set')
print('total accuracy')
print(sum(accu_list)/len(accu_list))
print(get_metrics(totalpredictlist, totallabel))
print('start to write file')
#文件写入
for index,file in enumerate(filelist2):
    if(index % 200==1):
        print(index)
    if(file[0]=='.'):
            filename=file[2:-4]
    else: filename=file[0:-4]
    if(file[0]=='.'):
        f = wave.open(r"D:\vad\wavs\test\\"+file[2:], "rb")
    else:  f = wave.open(r"D:\vad\wavs\test\\"+file, "rb")
    if(file[0]=='.'):
        fsize = os.path.getsize("D:\\vad\\wavs\\test"+'\\'+file[2:])
    else: fsize = os.path.getsize("D:\\vad\\wavs\\test"+'\\'+file)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data=wave_data/7250
    b,a=signal.butter(6,0.008,'highpass')
    wave_data=signal.filtfilt(b,a,wave_data)
    totallabel.extend(dic_test)
    predictlist=[]
    counter=0
    for index1 in range(int(len(wave_data)/128-3)-1):
        data_frame=wave_data[counter:counter+512]
        mfcc_feat = mfcc(data_frame,framerate)
        fbank_feat = logfbank(data_frame,framerate)
        mfcc_feat=mfcc_feat.reshape(26)
        fbank_feat=fbank_feat.reshape(52)
        feat_frame=np.append(mfcc_feat,fbank_feat)
        score1=gmm0.score(feat_frame.reshape(1,-1))
        score2=gmm1.score(feat_frame.reshape(1,-1))
        counter+=128
        if score1>score2:
            predictlist.append(0)
        else:
            predictlist.append(1)
    predictlist=modify(predictlist)
    time_to_time=prediction_to_vad_label(predictlist)
    with open("D:\\vad\\test_label.txt","a" )as f:
        f.writelines(filename+' '+time_to_time)
        f.write('\n')    
print('file has been written')