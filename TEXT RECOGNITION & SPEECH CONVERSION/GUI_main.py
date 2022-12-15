import tkinter as tk
from tkinter.filedialog import askopenfilename
import processing as pc
from PIL import Image, ImageTk
from tkinter import *
from tkinter import messagebox
from tkinter import *
from PIL import ImageTk, Image 
from tkinter import filedialog
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
import os,fnmatch
from skimage.data import coins
from skimage.morphology import label, remove_small_objects
from skimage.measure import regionprops, find_contours
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy
from scipy import stats
#from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
import tkinter.scrolledtext as st
import tkinter.scrolledtext as scrolledtext
from sklearn.svm import SVC
#from skimage.measure import compare_ssim
import warnings

image_file = None
originimage = None
proceimage = None


def resize(w, h, w_box, h_box, pil_image):
    
    f1 = 1.0 * w_box / w  # 1.0 forces float division in Python2
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    # print(f1, f2, factor) # test
    # use best down-sizing filter
    width = int(w * factor)
    height = int(h * factor)
    
    return pil_image.resize((width, height), Image.ANTIALIAS)


def open_image():
    global image_file
    
    filepath = askopenfilename()
    
    image_file = Image.open(filepath)
    file2 = open("ip.txt","w")
    file2.writelines(filepath)
    file2.close()
    w_box = 500
    h_box = 350
    showimg(image_file, imgleft, w_box, h_box)
    showimg(image_file, imgright, w_box, h_box)


def showimg(PIL_img, master, width, height):
    
    w, h = PIL_img.size
   
    img_resize = resize(w, h, width, height, PIL_img)
    # Image 2 ImageTk
    Tk_img = ImageTk.PhotoImage(image=img_resize)
    
    master.config(image=Tk_img)
    master.image = Tk_img



def Otsu():
    PIL_gary,PIL_Otsu = pc.Otus_hold(image_file)
    w_box = 500
    h_box = 350
    showimg(PIL_gary, imgleft, w_box, h_box)
    showimg(PIL_Otsu, imgright, w_box, h_box)
    histleft.config(image=None)
    histleft.image = None
    histright.config(image=None)
    histright.image = None
###############################################

def selection():
    
    choice = var.get()
    if choice == 1:
        m = 'Low'
    elif choice == 2:
        m = 'High'
    elif choice == 3:
        pass
    return m


######################################################################
def run():
        import main
def detect():
        import Detector

root = tk.Tk()
root.title('OCR')
root.geometry('1100x700')
root.config(bg='white')

menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='OPEN', command=open_image)
operate = tk.Menu(menubar, tearoff=0)
operate.add_command(label='OTSU',command=Otsu)
operate.add_command(label='Classify',command=run)
#operate.add_command(label='Detect',command=detect)

menubar.add_cascade(label='FILE', menu=filemenu)
menubar.add_cascade(label='Process', menu=operate)

frm = tk.Frame(root, bg='white')
frm.pack()
frm_left = tk.Frame(frm, bg='white')
frm_right = tk.Frame(frm, bg='white')
frm_left.pack(side='left')
frm_right.pack(side='right')

imgleft = tk.Label(frm_left, bg='white')
histleft = tk.Label(frm_left, bg='white')

imgright = tk.Label(frm_right, bg='white')
histright = tk.Label(frm_right, bg='white')
imgleft.pack()
histleft.pack()
imgright.pack()
histright.pack()
#################################

root.config(menu=menubar)
root.mainloop()
