import tkinter as tk

import librosa
import pyaudio
import numpy as np
from tkinter import TclError
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style
import random
import struct
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Model,load_model
root = tk.Tk()
root.title("AudioClassification")
root.geometry("1340x720+0+0")
def start():
    pass


def stop():
    pass


# --------------------------------------------------------------------------------------------------
frame1 = tk.Frame(root, padx=10, pady=10)
frame2 = tk.LabelFrame(root, padx=10, pady=10)
# --------------------------------------------------------------------------------------------------
f1=Figure(figsize=(6,8),dpi=90)
a=f1.add_subplot()
def animate1(i):
    xValue = pred
    a.clear()
    a.barh([1, 2, 3, 4, 5, 6, 7, 8], xValue, align='center',
           tick_label=['B/C', 'Classical', 'Disco', 'HipHop', 'Jazz', 'Pop', 'Reggae', 'Rock/Metal'])
    for i in range(8):
        a.text(s=str(xValue[i])[:4], x=xValue[i], y=i + 1)
    a.axes.get_xaxis().set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['bottom'].set_visible(False)
canvas = FigureCanvasTkAgg(f1, frame1)
canvas.draw()
canvas.get_tk_widget().pack()
#--------------------------------------------------------------------------------------------------4
f2=Figure(figsize=(9,8),dpi=90)
ax=f2.add_subplot(2,1,1)
ax1=f2.add_subplot(2,1,2)
def animate2(i):
    ax.clear()
    chunk = 33000
    Format = pyaudio.paFloat32
    channels = 1
    rate = 22050
    p = pyaudio.PyAudio()
    chosen_device_index = -1
    for x in range(p.get_device_count()):
        info = p.get_device_info_by_index(x)
        if info['name'] == 'pulse':
            chosen_device_index = info['index']
            print(chosen_device_index)
    stream = p.open(format=Format,
                    channels=channels,
                    rate=rate,
                    input_device_index=chosen_device_index,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    x_ = np.arange(0, chunk)
    data = stream.read(chunk)
    data_float = struct.unpack(str(chunk) + 'f', data)
    x=librosa.feature.melspectrogram(np.array(data_float), n_fft=1024,
        hop_length=256, n_mels=128)
    global pred,model
    x=x.reshape(1,128,129,1)
    dopreds(x,model)
    line, = ax.plot(x_, data_float)
    ax.set_ylim([-1, 1])
    data = struct.unpack(str(chunk) + 'f', stream.read(chunk))
    line.set_ydata(data)
    #-----------------------------------------------------------------------------------------------------------
    '''ax1.clear()
    chunk = 4000
    Format = pyaudio.paInt16
    channels = 1
    rate = 16000
    p = pyaudio.PyAudio()
    chosen_device_index = -1
    for x in range(p.get_device_count()):
        info = p.get_device_info_by_index(x)
        if info['name'] == 'pulse':
            chosen_device_index = info['index']
            print(chosen_device_index)
    stream = p.open(format=Format,
                    channels=channels,
                    rate=rate,
                    input_device_index=chosen_device_index,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    x = np.arange(0, chunk)
    data = stream.read(chunk)
    data_int16 = struct.unpack(str(chunk) + 'h', data)
    line, = ax1.plot(x, data_int16)
    ax1.set_ylim([-2 ** 15, (2 ** 15) - 1])
    data = struct.unpack(str(chunk) + 'h', stream.read(chunk))
    line.set_ydata(data)
    '''


def dopreds(x,model):
    global pred
    preds=model.predict(x)
    preds=np.array([preds[0,0]+preds[0,2],preds[0,1],preds[0,3],preds[0,4],preds[0,5],preds[0,7],preds[0,8],preds[0,9]+preds[0,6]])
    pred=(preds+(3*pred))/4


pred=np.zeros(8)
canvas = FigureCanvasTkAgg(f2, frame2)
canvas.draw()
canvas.flush_events()
canvas.get_tk_widget().pack()

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
frame1.grid(row=0, column=0,fill=None)
frame2.grid(row=0, column=1,fill=None)
# --------------------------------------------------------------------------------------------------
model=load_model('models/custom_cnn_2d_78.h5')
# ---------------------------------------------------------------------------------------------------------------
ani1 = animation.FuncAnimation(f1,animate1,interval=1500)
ani2 = animation.FuncAnimation(f2,animate2)
root.mainloop()
