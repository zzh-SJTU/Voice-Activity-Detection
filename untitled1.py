#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 23:25:47 2021

@author: zzh
"""
import os
import wave
import pylab as pl
import numpy as np
from scipy import signal


# 打开WAV文档
f = wave.open(r"/Users/zzh/Downloads/vad/wavs/dev/54-121080-0009.wav", "rb")
fsize = os.path.getsize("/Users/zzh/Downloads/vad/wavs/dev/54-121080-0009.wav")
print("fsize: ", fsize)

# 读取格式信息
# (nchannels, sampwidth, framerate, nframes, comptype, compname)
# 通道数 采样字节长度 采样频率 总帧数（总的采样数）
params = f.getparams()

nchannels, sampwidth, framerate, nframes = params[:4]
print("nchannels: ", nchannels)
print("sampwidth: ", sampwidth)
print("framerate: ", framerate)
print("nframes: ", nframes)
# 读取波形数据，读取并返回以 bytes 对象表示的最多 n 帧音频。
str_data = f.readframes(nframes)
# print(str_data)
f.close()
# 将波形数据转换为数组
wave_data = np.fromstring(str_data, dtype=np.short)
# print(wave_data)
print("size : ", wave_data.size)
# for i in range(wave_data.size):
#     if i % 10 == 0:
#         print();
#     print(wave_data[i])
    
if nchannels == 2:
    wave_data.shape = -1, 2 
    wave_data = wave_data.T 
    time = np.arange(0, nframes) * (1.0 / framerate) 
    # 绘制波形 
    pl.subplot(211) 
    pl.plot(time, wave_data[0]) 
    pl.subplot(212) 
    pl.plot(time, wave_data[1], c="g") 
    pl.xlabel("time (seconds)") 
    pl.show()

elif nchannels == 1:
    wave_data.shape = -1, 1 
    wave_data = wave_data.T 
    time = np.arange(0, nframes) * (1.0 / framerate) 
    # 绘制波形 
    # pl.subplot(211) 
    pl.plot(time, wave_data[0], 'c') 
    pl.savefig("1.png")
    pl.xlabel("time (seconds)") 
    pl.show()
    print(max(wave_data[0]))
