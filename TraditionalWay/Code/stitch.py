from ast import main, parse
from email import parser
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
from skimage.feature import peak_local_max
from detectCorner import *
from utils import *
from features import *
from stitchImage import *

def main(args):
    filepath = args["filepath"]
    imagesFolder = args["imagesFolder"]
    saveFolder = args["saveFolder"]
    showImages = args["showImages"]
    useHarris = args["useHarris"]
    choice = 1 if useHarris else 2 # shi-Tomashi , 1 for Harris corner

    images = readImages(os.path.join(filepath, imagesFolder))
    if showImages:
        displayImages(images, os.path.join(filepath, saveFolder, 'concatenated.png'))

    # total images
    N_images = len(images)
    # split in sets
    N_first_half = round(N_images/2)
    N_second_half = N_images - N_first_half
    print('Splitting total images in sets of', N_first_half, ',and', N_second_half)
    GoSequentially = True

    if not GoSequentially:
        while N_images is not 2:
            print("N = ", N_images, " N_half = ", N_first_half)
            merged_images = []
            for n in range(0, N_first_half, 2):
                if (n+1) <= N_first_half:
                    img_array = images[n:n+2]
                    print("combining: ", n, n+1)
                    I = joinImages(img_array, choice, os.path.join(filepath, saveFolder), n, showImages)
                    merged_images.append(I)
                else:
                    print("adding: ", n)
                    merged_images.append(images[n])

            for n in range(N_first_half, N_images, 2):
                if (n+1) < N_images:
                    img_array = images[n:n+2]
                    img_array.reverse()
                    print("combining: ", n+1, n)
                    I = joinImages(img_array, choice, os.path.join(filepath, saveFolder), n, showImages)
                    merged_images.append(I)
                else:
                    print("adding: ", n)
                    merged_images.append(images[n])
        
            images = merged_images
            N_images = len(images)
            N_first_half = round(N_images/2)
            N_second_half = N_images - N_first_half
        
        print("final merging")
        if N_images % 2 != 0:
            print("reversing")
            merged_images.reverse()
        final = joinImages(merged_images, choice, os.path.join(filepath, saveFolder), 100, showImages)
    else:
        Image0 = images[0]
        for n in range(N_images - 1):
            img_array = [Image0, images[n+1]]
            Image0 = joinImages(img_array, choice, os.path.join(filepath, saveFolder), n, showImages)

        if showImages:
            cv2.imshow(os.path.join(filepath, saveFolder) + "/pano" + str(n) + ".png", Image0)
            cv2.waitKey() 
            cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(filepath, saveFolder) + "/pano" + str(n) + ".png", Image0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-filepath", "--filepath", required=False,
            type=str,
            default=r"C:\Users\siddh\Documents\733\Panoroma_Stitching\TraditionalWay")
    parser.add_argument("-saveFolder", "--saveFolder", required=False,
            type=str,
            default=r"Results\Set4")
    parser.add_argument("-imagesFolder", "--imagesFolder", required=False,
            type=str,
            default=r"Data\Train\Set1")
    parser.add_argument("-showImages", "--showImages", required=False,
            type=bool, default=True)
    parser.add_argument("-useHarris", "--useHarris", required=False,
            type=bool, default=False)
    args = vars(parser.parse_args())
    main(args)

