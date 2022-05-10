from PIL import Image, ImageFilter
from numpy import False_
from sqlalchemy import false, true

sy = ['dot', 'tan', ')', '(', '+', '-', '|', 'sqrt', '1', '0', '3', '2', '4', '6','8', 'mul', 'pi', 'sin', 'A', 'cube root', 'co', 'os', 'mn', 'frac', 'cos', 'delta', 'a', 'c', 'b', 'bar', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', 'z', 'v', 'l', 'w', 'div', 'z_no_line', 'z_line']

slash_sy = ['tan', 'sqrt', 'mul', 'pi', 'sin', 'frac', 'cos', 'delta', 'bar', 'div','^','_', 'cube root']

variable = ['1', '0', '3', '2', '4', '6', '8', 'pi', 'A', 'a', 'c', 'b', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', 'z', 'v', 'l', 'w', '(', ')', 'dot', '|', 'mn', 'z_no_line', 'z_line']
brules = {}

def prepare_image(image):
    im = image.convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (0))

    if width > height:
        nheight = int(round((28.0/width*height),0))
        nheight = max(3, nheight)
        img = im.resize((28,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) 
        newImage.paste(img, (0, wtop)) 
    else:
        nwidth = int(round((28.0/height*width),0))
        nwidth = max(3, nwidth)
        img = im.resize((nwidth,28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) 
        newImage.paste(img, (wleft, 0))
    tv = list(newImage.getdata()) 
    tva = [ 1-(255-x)*1.0/255.0 for x in tv]
    return tva, newImage

def update(im_name, symbol_list):
    im = Image.open(im_name)
    list_len = len(symbol_list)
    for i in range(list_len):
        if i >= len(symbol_list): break
        
        symbol = symbol_list[i]
        predict_result = symbol[1]
        
        # deal with cos mark
        if predict_result == "c":
            if i < (len(symbol_list) - 2):
                s1 = symbol_list[i+1]
                s2 = symbol_list[i+2]
                if symbol_list[i+1][1] == "0" and symbol_list[i+2][1] == "s":
                    updateCos(symbol, s1, s2, symbol_list, im, i)
                    continue

            if i < (len(symbol_list) - 1):
                s1 = symbol_list[i+1]
                if symbol_list[i+1][1] == "os":
                    updateC_os(symbol, s1, symbol_list, im, i)
                    continue
            
        if predict_result == "(":
            if i < (len(symbol_list) - 2):
                s1 = symbol_list[i+1]
                s2 = symbol_list[i+2]
                if symbol_list[i+1][1] == "0" and symbol_list[i+2][1] == "s":
                    updateCos(symbol, s1, s2, symbol_list, im, i)
                    continue

            if i < (len(symbol_list) - 1):
                s1 = symbol_list[i+1]
                if symbol_list[i+1][1] == "os":
                    updateC_os(symbol, s1, symbol_list, im, i)
                    continue
        
        if predict_result == "co":
            if i < (len(symbol_list) - 1):
                s1 = symbol_list[i+1]
                if symbol_list[i+1][1] == "s":
                    updateCo_s(symbol, s1, symbol_list, im, i)
                    continue
        # deal with cos mark
        if predict_result == "s":
            if i < (len(symbol_list) - 2):
                s1 = symbol_list[i+1]
                s2 = symbol_list[i+2]
                if (symbol_list[i+1][1] == "1" or symbol_list[i+1][1] == "|" ) and symbol_list[i+2][1] == "n":
                    updateSin(symbol, s1, s2, symbol_list, im, i)
                    continue
        
        # deal with bar
        if predict_result == "-":
            if i < (len(symbol_list) - 2):
                s1 = symbol_list[i+1]
                s2 = symbol_list[i+2]
                if isVSame(symbol, s1) and (not isVSame(symbol, s2)):
                    updateBar(symbol, symbol_list, im, i)
                    continue
        
        # deal with fraction
        if predict_result == "-":
            j = i
            upPart = 0
            underPart = 0
            while j < len(symbol_list):
                tmp = symbol_list[j]
                if tmp[2] > symbol[2] - 10 and tmp[4] - 10 < symbol[4] and tmp[5] > symbol[3] - 10: upPart += 1
                if tmp[2] > symbol[2] - 10 and tmp[4] - 10 < symbol[4] and tmp[3] - 10 < symbol[5]: underPart += 1
                j += 1
            if upPart > 0 and underPart > 0:
                updateFrac(symbol, symbol_list, im, i)
                continue
                        
        # deal with dots
        if predict_result == "dot":
            if i < (len(symbol_list) - 2):
                s1 = symbol_list[i+1]
                s2 = symbol_list[i+2]
                if symbol_list[i+1][1] == "dot" and symbol_list[i+2][1] == "dot":
                    updateDots(symbol, s1, s2, symbol_list, im, i)
                    continue
        
        # deal with i
        if predict_result == "dot":
            if i < (len(symbol_list) - 1):
                s1 = symbol_list[i+1]
                if (s1[1] == "1" or s1[1] == "|")and abs(s1[2] - symbol[2]) < 30:
                    updateI(symbol, s1, symbol_list, im, i)
                    continue
            
            if i > 1:
                s1 = symbol_list[i-1]
                if (s1[1] == "1" or s1[1] == "|")and abs(s1[2] - symbol[2]) < 30:
                    updateI(symbol, s1, symbol_list, im, i-1)
                    continue
        #deal with z
        if predict_result == "z_no_line":
            symbol_list[i] = (im, "z", symbol[2], symbol[3], symbol[4], symbol[5])
            continue
        if predict_result == "z_line":
            symbol_list[i] = (im, "z", symbol[2], symbol[3], symbol[4], symbol[5])
            continue

    return symbol_list

def categorize(symbol_list):
    s = []
    i = 0
    flag = False
    flag_sqrt = False
    flag_frac = False
    while (i < len(symbol_list)):
    
        symbol = symbol_list[i]
        value = symbol[1]
        
        if value == "s" and not flag:
            return 8
        if value in variable:
            flag = True
        elif value == "sin" or value == "si":
            return 8
        elif value == "cos" or value == "co":
            i = i + 1
            while (i < len(symbol_list)):
                if symbol_list[i][1] == "frac":
                    return 10
                i+=1
            return 9
        if value == 'cube root':
            return 3
        elif value == 'sqrt':
            flag_sqrt = True
            if flag_frac:
                return 6
            i = i + 1
            while (i < len(symbol_list) and isInner(symbol, symbol_list[i])):
                # sqrt with expression
                if isSqrtExp(symbol, symbol_list[i]):
                    return 3
                else:
                    if symbol_list[i][1] == "frac":
                        return 5
                i = i + 1
                # contains some precedent
            if flag:
                return 3
        elif value == "frac":
            flag_frac = True
            if flag_sqrt:
                return 6
            upper = []
            under = []
            i = i + 1
            while (i < len(symbol_list) and (isUpperFrac(symbol, symbol_list[i]) or isUnderFrac(symbol, symbol_list[i]))):
                if isUpperFrac(symbol, symbol_list[i]): 
                    if symbol_list[i][1] == "sqrt":
                        return 6
                if isUnderFrac(symbol, symbol_list[i]): 
                    if symbol_list[i][1] == "sqrt":
                        return 6
                if symbol_list[i][1] == "sqrt":
                    return 6
                i = i + 1
            return 4

        elif value == "a" and i< len(symbol_list)-1 and symbol_list[i+1][1] == "s":
            temp = i + 1
            while (temp < len(symbol_list)):
                if symbol_list[temp][1] == "frac":
                    return 10
                temp+=1
        # is upper symbol
        elif i < len(symbol_list) - 1 and isUpperSymbol(symbol, symbol_list[i+1]) and (symbol[1] in variable) and (symbol_list[i+1][1] in variable): 
            i = i+1
            cnt = 0
            while (i < len(symbol_list) and isUpperSymbol(symbol, symbol_list[i])):
                cnt+=1
                i = i + 1
            if cnt == 1: return 1
            elif cnt > 1: return 7
        i = i + 1
    if flag_sqrt:
        return 2
    return 9
def toLatex(symbol_list):
    s = []
    i = 0
    while (i < len(symbol_list)):
        symbol = symbol_list[i]
        value = symbol[1]
        
        if value == 'frac':
            upper = []
            under = []
            i = i + 1
            while (i < len(symbol_list) and (isUpperFrac(symbol, symbol_list[i]) or isUnderFrac(symbol, symbol_list[i]))):
                if isUpperFrac(symbol, symbol_list[i]): upper.append(symbol_list[i])
                if isUnderFrac(symbol, symbol_list[i]): under.append(symbol_list[i])
                i = i + 1
            if len(upper) > 0 and upper[len(upper) - 1][1] not in variable:
                upper.pop()
                i = i - 1
            if len(under) > 0 and under[len(under) - 1][1] not in variable:
                under.pop()
                i = i - 1
            upper_string = '{' + toLatex(upper) + '}'
            under_string = '{' + toLatex(under) + '}'
            s.append('\\frac'+upper_string+under_string)
            continue
        elif value == 'sqrt':
            outer = []
            inner = []
            i = i + 1
            while (i < len(symbol_list) and isInner(symbol, symbol_list[i])):
                if isSqrtExp(symbol, symbol_list[i]):
                    outer.append(symbol_list[i])
                else:
                    inner.append(symbol_list[i])
                i = i + 1
            outer_string = '{' + toLatex(outer) + '}'
            if len(outer) == 0:
                outer_string = ""
            if len(inner) > 0 and inner[len(inner) - 1][1] not in variable:
                inner.pop()
                i = i - 1
            inner_string = '{' + toLatex(inner) + '}'
            s.append('\\sqrt'+outer_string+inner_string)
            continue
        elif value == 'cube root':
            inner = []
            i = i + 1
            while (i < len(symbol_list) and isInner(symbol, symbol_list[i])):
                inner.append(symbol_list[i])
                i = i + 1
            if len(inner) > 0 and inner[len(inner) - 1][1] not in variable:
                inner.pop()
                i = i - 1
            inner_string = '{' + toLatex(inner) + '}'
            s.append('\\sqrt'+"{3"+"}"+inner_string)
            continue
        elif value in slash_sy: 
            s.append('\\' + value)
            base = i
        elif i > 0 and (s[len(s) - 1] in slash_sy): 
            # need to consider about range within squrt and frac
            s.append('{'+value+'}')
        elif i < len(symbol_list) - 1 and isUpperSymbol(symbol, symbol_list[i+1]) and (symbol[1] in variable) and (symbol_list[i+1][1] in variable): 
            s.append(value)
            s.append('^{')
            i = i+1
            while (i < len(symbol_list) and isUpperSymbol(symbol, symbol_list[i])):
                s.append(symbol_list[i][1])
                i = i + 1
            s.append('}')
            continue
        else: 
            s.append(value)
            base = i
        i = i + 1
    return "".join(s)
                    
def isVSame(cur, next):
    cur_center_x = cur[2] + (cur[4] - cur[2])/2
    next_center_x = next[2] + (next[4] - next[2])/2
    if abs(cur_center_x - next_center_x) < 30: return True
    else: return False

def isInner(cur, next):
    if next[3] < cur[5] and next[2] > cur[2] and next[2] < cur[4]: return True
    else: return False
    
def isUpperFrac(cur, next):
    if next[5] < cur[5] and next[2] - cur[2] > -10 and next[4] - cur[4] < 10: return True
    else: return False

def isSqrtExp(cur, next):
    cur_center = cur[3] + (cur[5] - cur[3])/2
    next_center = next[3] + (next[5] - next[3])/2
    cur_center_x = cur[2] + (cur[4] - cur[2])/2
    # 
    if next_center < cur_center   and next[2] < cur_center_x - 20: return True 
    else: return False

def isUnderFrac(cur, next):
    if next[5] > cur[5] and next[2] - cur[2] > -10 and next[4] - cur[4] < 10: return True
    else: return False
    
def isUpperSymbol(cur, next):
    cur_center = cur[3] + (cur[5] - cur[3])/2
    next_center = next[3] + (next[5] - next[3])/2
    cur_center_x = cur[2] + (cur[4] - cur[2])/2
    # 
    if next_center < cur_center - (next[5] - next[3])/2  and next[2] > cur_center_x: return True 
    else: return False

def area(symbol):
    return (symbol[4] - symbol[2]) * (symbol[5] - symbol[3])
    
def updateDots(symbol,s1,s2,symbol_list, im, i):
    new_x = min(symbol[2], s1[2], s2[2])
    new_y = min(symbol[3], s1[3], s2[3])
    new_xw = max(symbol[4], s1[4], s2[4])
    new_yh = max(symbol[5], s1[5], s2[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "dots", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+2)
    symbol_list.pop(i+1)
    
def updateCos(symbol,s1,s2,symbol_list, im, i):
    new_x = min(symbol[2], s1[2], s2[2])
    new_y = min(symbol[3], s1[3], s2[3])
    new_xw = max(symbol[4], s1[4], s2[4])
    new_yh = max(symbol[5], s1[5], s2[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "cos", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+2)
    symbol_list.pop(i+1)

def updateC_os(symbol,s1,symbol_list, im, i):
    new_x = min(symbol[2], s1[2])
    new_y = min(symbol[3], s1[3])
    new_xw = max(symbol[4], s1[4])
    new_yh = max(symbol[5], s1[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "cos", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+1)

def updateCo_s(symbol,s1,symbol_list, im, i):
    new_x = min(symbol[2], s1[2])
    new_y = min(symbol[3], s1[3])
    new_xw = max(symbol[4], s1[4])
    new_yh = max(symbol[5], s1[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "cos", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+1)

def updateSin(symbol,s1,s2,symbol_list, im, i):
    new_x = min(symbol[2], s1[2], s2[2])
    new_y = min(symbol[3], s1[3], s2[3])
    new_xw = max(symbol[4], s1[4], s2[4])
    new_yh = max(symbol[5], s1[5], s2[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "sin", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+2)
    symbol_list.pop(i+1)

def updateI(symbol,s1,symbol_list, im, i):
    new_x = min(symbol[2], s1[2])
    new_y = min(symbol[3], s1[3])
    new_xw = max(symbol[4], s1[4])
    new_yh = max(symbol[5], s1[5])
    new_symbol = (im.crop((new_x, new_y, new_xw, new_yh)), "i", new_x, new_y, new_xw, new_yh)
    symbol_list[i] = new_symbol
    symbol_list.pop(i+1)
    
def updateBar(symbol,symbol_list, im, i):
    x, y, xw, yh = symbol[2:]
    new_symbol = (symbol[0], "bar", x, y, xw, yh)
    symbol_list[i] = new_symbol
    
def updateFrac(symbol,symbol_list, im, i):
    x, y, xw, yh = symbol[2:]
    new_symbol = (symbol[0], "frac", x, y, xw, yh)
    symbol_list[i] = new_symbol