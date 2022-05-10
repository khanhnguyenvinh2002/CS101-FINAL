import numpy as np
import cv2
import glob
import os

from PIL import Image, ImageDraw

image_list = glob.glob("./data/testing_toy/*.*")
text_file = open("Output.txt", "w")

for item in image_list:
    im = cv2.imread(item)
    head, tail = os.path.split(item)
    # comment out the line below if image has black background and white mark
    im = cv2.bitwise_not(im)  
    
    a = 50
    im[im >= a] = 255
    im[im < a] = 0

    kernel = np.ones((2,2),np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

    image = Image.fromarray(im)

    imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgrey, 127, 255, 0)
    contours, hierachy = cv2.findContours(thresh,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    num = 1
    res_img = image
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if x == 0: continue
        if w*h < 25: continue
        draw = ImageDraw.Draw(image)
        cv2.rectangle(im, (x,y), (x+w, y+h), (0,255,0), 2)
        num = num + 1
    
    image.save("./test_toy_res/" + os.path.splitext(tail)[0] +".png")
    