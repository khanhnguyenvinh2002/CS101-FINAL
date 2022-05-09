from PIL import Image, ImageFilter
import cv2

class ImageProcessing:
    def __init__(self, datart):
        self.dataroot = datart

    def process_image(self, image_name):
        im = Image.open(self.dataroot +image_name).convert('L')
        im.save(self.dataroot +image_name)
        im = cv2.imread(self.dataroot +image_name)

        image = Image.fromarray(im)

        width = float(image.size[0])
        height = float(image.size[1])

        new_image = Image.new('L', (28, 28), (0))

        if width > height:
            new_height = int(round((28.0/width*height),0))
            if (new_height == 0):
                new_height = 1
            img = image.resize((28,new_height), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((28 - new_height)/2),0))
            new_image.paste(img, (0, wtop)) 
        else:
            new_width = int(round((28.0/height*width),0)) 
            if (new_width == 0): 
                new_width = 1
            img = image.resize((new_width,28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((28 - new_width)/2),0))
            new_image.paste(img, (wleft, 0)) 

        tv = list(new_image.getdata())
        tva = [x * 1.0/255.0 for x in tv]
        return tva, new_image