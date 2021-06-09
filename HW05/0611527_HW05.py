"""
Machine_Learning_HW05
0611527_Tsao_Hsin-Yi
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

#----- Use array to draw the image of the question ---------------------------------------
image = np.array([
        [0,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,0,0,0,1,1],
        [0,0,1,1,1,0,1,1,1,1],
        [0,0,1,1,0,0,1,1,1,1],
        [0,1,1,1,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,1,0],
        [0,0,0,1,1,1,0,0,1,0],
        [0,0,0,1,1,1,1,0,0,0],
        [0,0,0,1,1,1,1,0,0,0],
        [0,0,0,0,0,0,1,0,0,0]
], dtype=np.uint8)

plt.imshow(image, cmap='gray_r') # show the image
plt.axis('off') # do not show the axis

#----- Save the image as 'image.png' -----------------------------------------------------

plt.savefig('image.png')

#----- Import the image  -----------------------------------------------------------------

img = cv2.imread('image.png', 0) 

print(img.shape) 
# shape(height, width, channel) =（288, 432, 3) 
# [channels]: 3(RGB); 1(cmap=gray)


#----- Define Point ----------------------------------------------------------------------

class Point(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y
 
    def getX(self):
        return self.x
    def getY(self):
        return self.y

#----- Calculate the value between two points  -------------------------------------------

def getGrayDiff(img,currentPoint,tmpPoint):
    return abs(int(img[currentPoint.x,currentPoint.y]) - int(img[tmpPoint.x,tmpPoint.y]))

#----- Define 8-connectivity/4-connectivity  ---------------------------------------------

def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [ Point(0, -1),  Point(1, 0),Point(0, 1), Point(-1, 0)]
    return connects

#----- Define region growing algorithm ---------------------------------------------------

def regionGrow(img,seeds,thresh,p = 1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while(len(seedList)>0):
        currentPoint = seedList.pop(0)
 
        seedMark[currentPoint.x,currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint,Point(tmpX,tmpY))
            if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
                seedMark[tmpX,tmpY] = label
                seedList.append(Point(tmpX,tmpY))
    return seedMark

#----- Region growing algorithm for segament the image into three islands ----------------

seeds = [Point(60,135)]
seed_grow_1 = regionGrow(img,seeds,10)
cv2.imwrite('region_growth_1.jpg', seed_grow_1*255)
cv2.imshow('region_growth_1',seed_grow_1)
cv2.waitKey(0)

seeds = [Point(60,293)]
seed_grow_2 = regionGrow(img,seeds,10)
cv2.imwrite('region_growth_2.jpg', seed_grow_2*255)
cv2.imshow('region_growth_2',seed_grow_2)
cv2.waitKey(0)

seeds = [Point(165,183)]
seed_grow_3 = regionGrow(img,seeds,10)
cv2.imwrite('region_growth_3.jpg', seed_grow_3*255)
cv2.imshow('region_growth_3',seed_grow_3)
cv2.waitKey(0)

#----- Calculate the size of the island --------------------------------------------------

# step1: calculate the # of pixels for one size

image = cv2.imread('pixel.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
pixels = cv2.countNonZero(thresh)
#print('# of pixles / one size:', pixels)

# step2: calculate the # of pixels for region_growth_x.jpg, respectively

image1 = cv2.imread('region_growth_1.jpg')
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
thresh1 = cv2.threshold(gray1,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
pixels1 = cv2.countNonZero(thresh1)
#print('# of pixles / region_growth_1:', pixels1)

image2 = cv2.imread('region_growth_2.jpg')
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
thresh2 = cv2.threshold(gray2,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
pixels2 = cv2.countNonZero(thresh2)
#print('# of pixles / region_growth_2:', pixels2)

image3 = cv2.imread('region_growth_3.jpg')
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
thresh3 = cv2.threshold(gray3,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
pixels3 = cv2.countNonZero(thresh3)
#print('# of pixles / region_growth_1:', pixels3)

# step3: calculate the size by divided the # of pixels for one size

size1 = round(pixels1 / pixels)
print('the size of region_growth_1:', size1)

size2 = round(pixels2 / pixels)
print('the size of region_growth_2:', size2)

size3 = round(pixels3 / pixels)
print('the size of region_growth_3:', size3)





