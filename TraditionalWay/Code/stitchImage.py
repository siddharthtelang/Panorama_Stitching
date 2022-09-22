import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
from skimage.feature import peak_local_max
import copy
from utils import *
from detectCorner import *
from features import *

def stitchImagePairs(img0, img1, H):
    image0 = copy.deepcopy(img0)
    image1 = copy.deepcopy(img1)

    #stitch image 0 on image 1
    print("shapes")
    print(image0.shape)
    print(image1.shape)

    h0 ,w0 ,_ = image0.shape
    h1 ,w1 ,_ = image1.shape

    points_on_image0 = np.float32([[0, 0], [0, h0], [w0, h0], [w0, 0]]).reshape(-1,1,2)
    points_on_image0_transformed = cv2.perspectiveTransform(points_on_image0, H)
    print("transformed points = \n", points_on_image0_transformed)
    points_on_image1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1,1,2)
    points_on_merged_images = np.concatenate((points_on_image0_transformed, points_on_image1), axis = 0)
    points_on_merged_images_ = [p.ravel() for p in points_on_merged_images]
    # print('Points concatenated:\n', points_on_merged_images_)

    x_min, y_min = np.int0(np.min(points_on_merged_images_, axis = 0))
    x_max, y_max = np.int0(np.max(points_on_merged_images_, axis = 0))

    print('Min points: ', x_min, y_min)
    print('Max points: ', x_max, y_max)

    # translate to avoid negative values
    H_translate = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    H_final = np.dot(H_translate, H)

    # warp prespective
    image0_transformed_and_stitched = cv2.warpPerspective(image0, np.dot(H_translate, H), (x_max-x_min, y_max-y_min))

    plt.figure(figsize=(10,10))
    plt.imshow(image0_transformed_and_stitched)

    images_stitched = image0_transformed_and_stitched.copy()
    # add in second image
    images_stitched[-y_min:-y_min+h1, -x_min: -x_min+w1] = image1
    show(images_stitched)
    plt.figure(figsize=(10,10))
    plt.imshow(images_stitched)

    # fill in black spaces in second image (if any)
    indices = np.where(image1 == [0,0,0])
    # translate y, x from 2nd image to 1st image
    y = indices[0] + -y_min 
    x = indices[1] + -x_min 
    # fill in from the 1st image
    images_stitched[y,x] = image0_transformed_and_stitched[y,x]
    show(images_stitched)
    plt.figure(figsize=(10,10))
    plt.imshow(images_stitched)

    return images_stitched
    
def joinImages(img_array, choice, save_folder_name, n, show_steps = True):

    image_array = img_array.copy()
    N = len(image_array)
    image0 = image_array[0]
    j = 0
    for i in range(1, N):
        j = j + 1
        print("processing image ", i)
        image1 = image_array[i] 

        image_pair = [image0, image1]
        
        detected_corners, cmaps, corner_images = detectCorners(copy.deepcopy(image_pair), choice)
        if show_steps:
            displayImages(copy.deepcopy(corner_images), os.path.join(save_folder_name, "corners"+str(n)+str(j)+".png"))
        """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """
        if (choice == 1):
            print("Applying ALMS.")
            detected_corners, anms_image = AdaptiveNonMaximalSupression(copy.deepcopy(corner_images), cmaps, 300)
            if show_steps:
                displayImages(copy.deepcopy(anms_image), os.path.join(save_folder_name, "anms_output"+str(n)+str(j)+".png"))
        else:
            print("goodFeaturesToTrack is already using ALMS.") #review

        detected_corners0 = detected_corners[0]
        detected_corners1 = detected_corners[1]
                
        matched_pairs = getPairs(image0, image1, detected_corners0, detected_corners1, patch_size = 40)
        if show_steps:
            showMatches(image0, image1, matched_pairs, os.path.join(save_folder_name, "matched_pairs"+str(n)+str(j)+".png"))
        """
        Refine: RANSAC, Estimate Homography
        """
        H,filtered_matched_pairs = filterOutliers(matched_pairs, threshold=5)
        if show_steps:
            showMatches(image0, image1, filtered_matched_pairs, os.path.join(save_folder_name, "filtered_matched_pairs"+str(n)+str(j)+".png"))


        unique, counts = np.unique(filtered_matched_pairs[:,1,:], return_counts=True, axis = 0)
        unique_count = unique.shape[0]
        max_count = np.max(counts)

        stitching = True
        # print(unique_count, max_count)
        # if(unique_count < 7 and max_count > 8):
        #     print("Cannot match image")
        #     stitching = False
        """
        Image Warping + Blending
        Save Panorama output as mypano.png
        """
        if(stitching):
            stitched_image = stitchImagePairs(image0, image1, H)
            stitched_image = cropImageRect(stitched_image)
            if show_steps:
                cv2.imshow(os.path.join(save_folder_name, "pano"+str(n)+str(j)+".png"), stitched_image)
                cv2.waitKey() 
                cv2.destroyAllWindows()

            cv2.imwrite(os.path.join(save_folder_name, "pano"+str(n)+str(j)+".png"), stitched_image)
            image0 = stitched_image
        else:
            if show_steps:
                cv2.imshow(os.path.join(save_folder_name, "pano"+str(n)+str(j)+".png"), image0)
                cv2.waitKey() 
                cv2.destroyAllWindows()

            cv2.imwrite(os.path.join(save_folder_name, "pano"+str(n)+str(j)+".png"), image0)

    return image0