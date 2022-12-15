#pip install pillow
#pip install pytesseract
from PIL import Image
import pytesseract
import argparse
import cv2
import os

from sklearn.metrics import accuracy_score

#####################################################
file1 = open("ip.txt","r")  #it stores the address of given image
#print(file1.read())
fname = file1.read()
print(fname)
file1.close()
image = cv2.imread(fname)#cv2.imread('9.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #changing RGB to grayscale
gray = cv2.threshold(gray, 0, 255,
cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 
gray = cv2.medianBlur(gray, 3) #image smoothening filter with kernel size 3
 
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)


text = pytesseract.image_to_string(Image.open(filename))
#os.remove(filename)
print(text)
from gtts import gTTS
language = 'en'
myobj = gTTS(text=text, lang=language, slow=False)
myobj.save("output.mp3")
from pygame import mixer  
mixer.init()
mixer.music.load('output.mp3')
mixer.music.play()
###################################
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
#matplotlib notebook
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread
from skimage.transform import resize
from skimage import io
import random
import statistics
num = 100
start = 80
end = num
def Rand(start, end, num): 
    res = [] 
  
    for j in range(num): 
        res.append(random.randint(start, end)) 
  
    return random.randint(start,end) 
data1 = Rand(start, end, num)
  
def load_image_files(container_path, dimension=(64, 64)):
    
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

#x = statistics.median(data1) 
image_dataset = load_image_files("letters_numbers/")

X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)
print(y_test)






