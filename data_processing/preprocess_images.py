from PIL import Image
import os
from process_images import ImageProcessing
import random

root = os.getcwd() + "/data/symbols"
for subdir in os.listdir(root):
     if not subdir.startswith("."):
         for f in os.listdir(root + "/" +subdir):
             if f.endswith(".png"):
                 im1,im2 = ImageProcessing(root + "/" +subdir+"/").process_image(f)
                 im2.save(root + "/" +subdir+"/"+f)
