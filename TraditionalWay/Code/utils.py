import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
from skimage.feature import peak_local_max
import copy

# read the images
def readImages(filepath: str):
    print('Reading from: ', filepath)
    images = []
    files = os.listdir(filepath)
    sorted(files)
    for file in files:
        imagePath = os.path.join(filepath, file)
        image = cv2.imread(imagePath)
        if image is not None:
            images.append(image)
    print('Read ', len(images), ' images')
    return images

def makeImagesSizeSame(images:list[np.ndarray])->list[np.ndarray]:
    sizeList = []
    for image in images:
        x, y, ch = image.shape
        sizeList.append([x, y, ch])
    sizeArr = np.array(sizeList)
    r, c, ch = np.max(sizeArr, axis=0)
    newImages = []
    for i, image in enumerate(images):
        resized = np.zeros((r, c, sizeArr[i, 2]), np.uint8)
        resized[0:sizeArr[i, 0], 0:sizeArr[i, 1], 0:sizeArr[i, 2]] = image
        newImages.append(resized)
    return newImages

def displayImages(images:list[np.ndarray], fileName:str):
    newImages = makeImagesSizeSame(images)
    concatenated = newImages[0].copy()
    for i in range(1, len(newImages)):
        concatenated = np.concatenate((concatenated, newImages[i]), axis=1)
    cv2.imshow(fileName, concatenated)
    cv2.waitKey()
    cv2.destroyAllWindows()
    print('Writing file to ', fileName)
    cv2.imwrite(fileName, concatenated)

def showMatches(img1, img2, matched_pairs, filename):
    image1 = copy.deepcopy(img1)
    image2 = copy.deepcopy(img2)
    image1, image2 = makeImagesSizeSame([image1, image2])
    concat = np.concatenate((image1, image2), axis = 1)
    corners_1 = matched_pairs[:,0].copy()
    corners_2  = matched_pairs[:,1].copy()
    corners_2[:,0] += image1.shape[1]

    for (x1,y1) , (x2,y2) in zip(corners_1, corners_2):
        cv2.line(concat, (x1,y1), (x2,y2), (0, 0, 255), 1)
    cv2.imshow(filename, concat)
    cv2.waitKey() 
    cv2.destroyAllWindows()
    cv2.imwrite(filename, concat)
    return concat

def calcError(set1: np.ndarray, set2:np.ndarray):
    return np.linalg.norm(set1-set2, axis=1)

def cropImageRect(image):
    
    img = image.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # threshold to get the edges in image
    _,thresh = cv2.threshold(gray,5,255,cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    test = copy.deepcopy(image)
    # get the outer contour over the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(test, contours, -1, (0,255,0), 3)
    plt.imshow(test)

    x,y,w,h = cv2.boundingRect(contours[len(contours)-1])
    crop = img[y:y+h,x:x+w]

    return crop

def show(image):
    cv2.imshow('Image', image)
    cv2.waitKey()
    cv2.destroyAllWindows()