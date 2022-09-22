import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import os
from skimage.feature import peak_local_max
import copy
from utils import *

def getFeatureDescriptor(image, col, row, path_size=40):
    patch = image[col - path_size//2: col + path_size//2 , row - path_size//2: row + path_size//2]
    # gaussian blur
    patch = cv2.GaussianBlur(patch, (3,3), 0)
    # subsample to 20%
    patch = cv2.resize(patch, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    # flatten the patch
    feature = patch.reshape(-1)
    # zero mean and unit std
    feature = (feature - np.mean(feature)) / np.std(feature)
    return feature

def checkAndGetFeature(corners: list[np.ndarray], image:np.ndarray, patch_size=40):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width, height = image.shape
    features = []
    final_corners = []
    for corner in corners:
        x, y = corner.ravel()
        if (x - patch_size/2 > 0 and x + patch_size/2 < height
            and y - patch_size/2 > 0 and y + patch_size/2 < width):
            features.append(getFeatureDescriptor(image, y, x, patch_size))
            final_corners.append([x, y])
    return features, final_corners

def getPairs(img1, img2, corners1, corners2, patch_size=40):
    features = []
    corners = []
    
    feature, corner = checkAndGetFeature(corners1, img1, patch_size=patch_size)
    features.append(feature)
    corners.append(corner)

    feature, corner = checkAndGetFeature(corners2, img2, patch_size=patch_size)
    features.append(feature)
    corners.append(corner)

    matched_pairs = []
    for i, feature1 in enumerate(features[0]):
        ssd = []
        for j, feature2 in enumerate(features[1]):
            ssd.append(np.sum((feature1 - feature2)**2))
        best_match = np.argmin(ssd)
        matched_pairs.append([corners[0][i], corners[1][best_match]])
    print("Matched pairs between two images: ", len(matched_pairs))
    return np.array(matched_pairs)

def filterOutliers(matched_pairs:np.ndarray, threshold=5):
    set1 = matched_pairs[:, 0]
    set2 = matched_pairs[:, 1]
    N_best = -np.inf
    H_best = np.zeros((3,3))
    iterations = 5000
    threshold = 5
    best_matches = []
    for i in range(iterations):
        # randomly select 4 rows
        n_rows = set1.shape[0]
        random_idx = np.random.choice(n_rows, size=4)
        set1_random = set1[random_idx]
        set2_random = set2[random_idx]

        #compute Homography
        H = cv2.getPerspectiveTransform(np.float32(set1_random), np.float32(set2_random))
        
        # multiply with the source point to predict the target
        set1_dash = np.vstack((set1[:,0], set1[:,1], np.ones((1, n_rows))))
        set2_pred = np.dot(H, set1_dash)
        x_pred = set2_pred[0, :] / (set2_pred[2, :] + 1e-10)
        y_pred = set2_pred[1, :] / (set2_pred[2, :] + 1e-10)
        set2_pred = np.array([x_pred, y_pred]).T
        E = calcError(set2_pred, set2)
        E[E <= threshold] = 1
        E[E > threshold] = 0
        N = np.sum(E)
        if N > N_best:
            N_best = N
            H_best = H
            best_matches = np.where(E == 1)
        
    final_set1 = set1[best_matches]
    final_set2 = set2[best_matches]

    print("Final Good Matches: ", final_set1.shape[0])

    final_matches = np.zeros((final_set1.shape[0], 2, 2))
    # match from 1st image
    final_matches[:, 0, :] = final_set1
    # correspondence in 2nd image
    final_matches[:, 1, :] = final_set2
    final_matches = final_matches.astype(int)

    return H_best, final_matches

