import tkinter as tk
import customtkinter as ctk

import torch

import numpy as np

import cv2
from PIL import Image, ImageTk
import vlc

#create app
app=tk.Tk()
app.geometry('600x600')
app.title('recognize currency')
ctk.set_appearance_mode('dark')

vidFrame=tk.Frame(height=480,width=600) #create the container where the vedio will be displayed
vidFrame.pack()
vid=ctk.CTkLabel(vidFrame)
vid.pack()

#le modèle est entrainé sur google collab, j'ai donc enregistré le fichier last.pt qui comporte les poids et je l'ai mis sur le dossier
#it loads the file rom github
model=torch.hub.load('ultralytics/yolov5','custom',path=r"C:\Users\Hp\Desktop\mes projets\reconnaissance de monnaie\built_app\moroccan_currency_detector\yolov5\runs\train\exp4\weights\last.pt" ,
                      force_reload=True)
#force_reload=True is used to force the system to reload the file even if it was already installed
cap=cv2.VideoCapture(0) #connexion à ma camera

counter=0
counterlabel=ctk.CTkLabel(app,height=40,width=120,text=counter)
counterlabel.pack()

def reset_counter():
    global counter
    counter=0
resetbutton=ctk.CTkButton(app,height=40,width=120,text='reset counter',command=reset_counter)
resetbutton.pack()



def detect():
    global counter
    ret, frame=cap.read()
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=model(frame)
    img=np.squeeze(results.render())

    #print(len(results.xywh[0]))
    #print(results.xywh[0])
    if len(results.xywh[0])>0:
        precision=results.xywh[0][0][4]
        classe=results.xywh[0][0][5]
        if precision.item()>=0.1 and classe.item()==2.0:
            p=vlc.MediaPlayer(r"C:\Users\Hp\Desktop\mes projets\reconnaissance de monnaie\built_app\moroccan_currency_detector\mixkit-classic-alarm-995.wav")
            p.play()
            counter+=1
    imgarr=Image.fromarray(img)
    imgtk=ImageTk.PhotoImage(imgarr)
    vid.imgtk=imgtk  #associate the imgtk image with the vid object.
    vid.configure(image=imgtk)
    vid.after(10,detect)
    counterlabel.configure(text=counter)


detect()


app.mainloop()


