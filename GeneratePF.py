import cv2
import os
import torch
import time
import numpy as np
import sys
import utils
import random
import numpy as np
from config import Config
import scipy.misc


pair1_path='trainPF1/pair1'
pair2_path='trainPF2/pair2'
train_dir='train_input'
dir = 'train2014/'
count = 0

def mold_image(images, config):
    return (images.astype(np.float32) - config.MEAN_PIXEL)


if __name__ == '__main__':
	output_PF=[]
	for root,dire,files in os.walk(dir):
	    for file in files:
	        # print(1)
	        srcImg = cv2.imread(root+str(file))

	        image = utils.resize_image(
	            srcImg,
	            min_dim=Config.IMAGE_MIN_DIM,
	            max_dim=Config.IMAGE_MAX_DIM,
	            padding=Config.IMAGE_PADDING)

	        (height, width,channel) = image.shape	        

	        marginal = Config.MARGINAL_PIXEL
	        patch_size = Config.PATCH_SIZE

	        # create random point P within appropriate bounds
	        y = random.randint(marginal, height - marginal - patch_size)
	        x = random.randint(marginal, width - marginal - patch_size)

	        # define corners of image patch
	        top_left_point = (x, y)
	        bottom_left_point = (x, patch_size + y)
	        bottom_right_point = (patch_size + x, patch_size + y)
	        top_right_point = (x + patch_size, y)

	        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
	        perturbed_four_points = []
	        for point in four_points:
	            perturbed_four_points.append((point[0] + random.randint(-marginal, marginal),
	                                          point[1] + random.randint(-marginal, marginal)))

	        y_grid, x_grid = np.mgrid[0:image.shape[0], 0:image.shape[1]]
	        point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()


	        H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
	        H_inverse = np.linalg.inv(H)
	        warped_image = cv2.warpPerspective(image, H_inverse, (image.shape[1], image.shape[0]))

	        img_patch_ori = image[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0]]
	        img_patch_pert = warped_image[top_left_point[1]:bottom_right_point[1],
	                                      top_left_point[0]:bottom_right_point[0]]

	        point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float32), H).squeeze()
	        diff_branch1 = point_transformed_branch1 - point
	        diff_x_branch1 = diff_branch1[:, 0]
	        diff_y_branch1 = diff_branch1[:, 1]

	        diff_x_branch1 = diff_x_branch1.reshape((image.shape[0], image.shape[1]))
	        diff_y_branch1 = diff_y_branch1.reshape((image.shape[0], image.shape[1]))

	        pf_patch_x_branch1 = diff_x_branch1[top_left_point[1]:bottom_right_point[1],
	                                            top_left_point[0]:bottom_right_point[0]]

	        pf_patch_y_branch1 = diff_y_branch1[top_left_point[1]:bottom_right_point[1],
	                                            top_left_point[0]:bottom_right_point[0]]

	        pf_patch = np.zeros((Config.PATCH_SIZE, Config.PATCH_SIZE, 2))
	        pf_patch[:, :, 0] = pf_patch_x_branch1
	        pf_patch[:, :, 1] = pf_patch_y_branch1
	        pf_patch=pf_patch.transpose(2,0,1)
	        output_PF.append(pf_patch)
	        
	        cv2.imwrite(pair1_path+"/"+str(file),image)
	        cv2.imwrite(pair2_path+"/_"+str(file),warped_image)
	        
	np.save("output_PF.npy",np.array(output_PF,dtype=np.float32))
        
