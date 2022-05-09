
import cv2

from PIL import Image, ImageDraw

def isDot(bounding_box_0):
    (x, y), (xw, yh) = bounding_box_0
    area = (yh - y) * (xw - x)
    return area < 200 and 0.5 < (xw - x)/(yh - y) < 2 and abs(xw - x) < 20 and abs(yh - y) < 20 

def isVerticalBar(bounding_box_0):
    (x, y), (xw, yh) = bounding_box_0
    return (yh - y) / (xw - x) > 2

def isHorizontalBar(bounding_box_0):
    (x, y), (xw, yh) = bounding_box_0
    return (xw - x) / (yh - y) > 2

def isSquare(bounding_box_0):
    (x, y), (xw, yh) = bounding_box_0
    return (xw - x) > 8 and (yh - y) > 8 and 0.5 < (xw - x)/(yh - y) < 2

def isDivisionMark(bounding_box_0, bounding_box_1, bounding_box_2):
    (x, y), (xw, yh) = bounding_box_0
    (x1, y1), (xw1, yh1) = bounding_box_1
    (x2, y2), (xw2, yh2) = bounding_box_2
    
    return (isHorizontalBar(bounding_box_0) and isDot(bounding_box_1) and isDot(bounding_box_2)
            and x < x1 < x2 < xw and max(y1, y2) > y and min(y1, y2) < y
            and max(y1, y2) - min(y1, y2) < 1.2 * abs(xw - x))

def isLetterI(bounding_box_0, bounding_box_1):
    (x, y), (xw, yh) = bounding_box_0
    (x1, y1), (xw1, yh1) = bounding_box_1
    return (((isDot(bounding_box_0) and isVerticalBar(bounding_box_1)) or (isDot(bounding_box_1) and isVerticalBar(bounding_box_0)))
            and abs(x1 - x) < 10) 

def isEquationMark(bounding_box_0, bounding_box_1):
    (x, y), (xw, yh) = bounding_box_0
    (x1, y1), (xw1, yh1) = bounding_box_1
    return isHorizontalBar(bounding_box_0) and isHorizontalBar(bounding_box_1) and abs(x1 - x) < 20 and abs(xw1 - xw) < 20

def isFraction(bounding_box_0, bounding_box_1, bounding_box_2):
    (x, y), (xw, yh) = bounding_box_0
    (x1, y1), (xw1, yh1) = bounding_box_1
    (x2, y2), (xw2, yh2) = bounding_box_2
    cenX = x + (xw - x) / 2
    cenX1 = x1 + (xw1 - x1) / 2
    cenX2 = x2 + (xw2 - x2) / 2
    case1 = not isDot(bounding_box_0) and not isDot(bounding_box_1) and isHorizontalBar(bounding_box_2) and (y < y2 < yh1 or y1 < y2 < yh)
    case2 = not isDot(bounding_box_2) and not isDot(bounding_box_0) and isHorizontalBar(bounding_box_1) and (y2 < y1 < yh or y < y1 < yh2)
    case3 = not isDot(bounding_box_1) and not isDot(bounding_box_2) and isHorizontalBar(bounding_box_0) and (y1 < y < yh2 or y2 < y < yh1)
    return (case1 or case2 or case3) and  max(cenX, cenX1, cenX2) - min(cenX, cenX1, cenX2) < 50 
    
def initialize_boxes(im):
    im[im >= 127] = 255
    im[im < 127] = 0
    imgrey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(imgrey, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    res = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if x == 0: continue
        if w*h < 25: continue
            
        res.append([(x,y), (x+w, y+h)])
    return res

def connect(im, res):
    finalRes = []
    res.sort()
    i = 0
    while (i < len(res) - 1):
        (x, y), (xw, yh) = res[i]
        (x1, y1), (xw1, yh1) = res[i+1]

        equation = isEquationMark(res[i],  res[i + 1])
        letterI = isLetterI(res[i], res[i+1])
        divisionMark = False
        fraction = False
        if i < len(res) - 2:
            (x2, y2), (xw2, yh2) = res[i+2]
            divisionMark = isDivisionMark(res[i], res[i+1], res[i+2])
            fraction = isFraction(res[i], res[i+1], res[i+2])

        if (equation or letterI) and not fraction:
            finalRes.append([(min(x, x1), min(y, y1)), (max(xw, xw1), max(yh, yh1))])
            i += 2
        elif (divisionMark) and not fraction:
            finalRes.append([(min(x, x1, x2), min(y, y1, y2)), (max(xw, xw1, xw2), max(yh, yh1, yh2))])
            i += 3
        else:
            finalRes.append(res[i])
            i += 1

    while i < len(res):
        finalRes.append(res[i])
        i += 1

    return finalRes

        
def createSymbol(path):
    im = cv2.imread(path)
    image = Image.fromarray(im)
    raw_response = initialize_boxes(im)
    boxes = connect(im, raw_response)
    boxes = sorted(boxes, key=lambda box: (box[1][1]-box[0][1]) * (box[1][0]-box[0][0]))
    
    symbol_list= []
    for box in boxes:
        (x, y), (xw, yh) = box
        x -= 1
        y -= 1
        xw += 1
        yh += 1
        w = xw - x - 2
        h = yh - y - 2
        symbolImage = image.crop((x, y, xw, yh))
        if w < 10 and h < 10 and float(w)/h < 1.5 and float(h)/w < 1.5 :
            symbol_info = (symbolImage, "dot", x, y, xw, yh)
        else :
            symbol_info = (symbolImage, "not_dot", x, y, xw, yh)

        symbol_list.append(symbol_info)
        draw = ImageDraw.Draw(image)
        draw.rectangle((x, y, xw, yh), fill = 'black')
    return symbol_list
