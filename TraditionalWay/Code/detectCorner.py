import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import peak_local_max
import copy

def detectCorners(images:list[np.ndarray], choice:int):
    print('Detecting Corners...')
    detectedCorners = []
    cmaps = []
    corner_images = []
    for image in images:
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if choice == 1:
            print("Harris corner detection method.")
            corner_strength = cv2.cornerHarris(gray_image,2,3,0.001)
            corner_strength[corner_strength<0.01*corner_strength.max()] = 0
            cmaps.append(corner_strength)
            detected_corner = np.where(corner_strength>0.0001*corner_strength.max())
            detectedCorners.append(detected_corner)
            image[detected_corner] = [0,0,255]
            corner_images.append(image)
        else:
            print("Shi-Tomashi corner detection method.")
            dst = cv2.goodFeaturesToTrack(gray_image, 1000 ,0.01, 10)
            dst = np.int0(dst)
            detectedCorners.append(dst)
            for corner in dst:
                x, y = corner.ravel()
                cv2.circle(image, (x, y) ,3 ,(0, 0, 255), -1)
            corner_images.append(image)
            # cmap not used for shi-tomashi
            cmap = np.zeros(gray_image.shape)
            cmaps.append(cmap)
    return detectedCorners, cmaps, corner_images

def AdaptiveNonMaximalSupression(images:list[np.ndarray], Cmaps:list[np.ndarray], N_best:int):
    anms_images = []
    anms_corners = []
    for i, image in enumerate(images):
        cmap = Cmaps[i]
        local_max = peak_local_max(cmap, min_distance=15)
        n_strong = local_max.shape[0]
        r = [np.inf for i in range(n_strong)]
        dist = 0
        final = np.zeros((n_strong, 2), dtype=int)
        
        for i in range(n_strong):
            for j in range(n_strong):
                x_i = local_max[i][0]
                y_i = local_max[i][1]

                x_j = local_max[j][0]
                y_j = local_max[j][1]

                if (cmap[x_j,y_j] > cmap[x_i, y_i]):
                    dist = np.square(x_i-x_j) + np.square(y_i-y_j)
                if (dist < r[i]):
                    r[i] = dist
                    final[i, 0] = y_j
                    final[i, 1] = x_j
        
        if n_strong < N_best:
            N_best = n_strong
        index = np.argsort(r)
        index = np.flip(index)
        index = index[:N_best]
        final = final[index]
        for cx, cy in final:
            cv2.circle(image, (cx, cy), 4, (255, 0, 0), -1)
        anms_images.append(image)
        anms_corners.append(final)

    return anms_corners, anms_images
            
