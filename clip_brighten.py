import numpy as np
import cv2
import glob
import os

from PIL import Image, ImageDraw
import PIL.ImageOps

# for i in range(0,10):
name = "./data/testing_toy/004.png"
im = cv2.imread(name)
head, tail = os.path.split(name)
# convert the image into white and black image
im = cv2.bitwise_not(im)

# set the morphology kernel size, the number in tuple is the bold pixel size
kernel = np.ones((2,2),np.uint8)
im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

kernel = np.ones((2,2),np.uint8)
# convert the image into white and black image
a = 70
im[im >=a] = 255
im[im < a] = 0

# make a tmpelate image for next crop
image = Image.fromarray(im)
image.save("./test_toy_res/" + os.path.splitext(tail)[0] +".png")