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
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return (images.astype(np.float32) - config.MEAN_PIXEL)


if __name__ == '__main__':
	output_PF=[]
	for root,dire,files in os.walk(dir):
	    for file in files:
	        # print(1)
	        srcImg = cv2.imread(root+str(file))
	        # srcImg = cv2.cvtColor(srcImg,cv2.COLOR_BGR2GRAY)

	        image = utils.resize_image(
	            srcImg,
	            min_dim=Config.IMAGE_MIN_DIM,
	            max_dim=Config.IMAGE_MAX_DIM,
	            padding=Config.IMAGE_PADDING)

	        (height, width,channel) = image.shape
	        # cv2.imshow('image',image)
	        # cv2.waitKey()
	        

	        marginal = Config.MARGINAL_PIXEL
	        patch_size = Config.PATCH_SIZE

	        # create random point P within appropriate bounds
	        # y = random.randint(marginal, height - marginal - patch_size)
	        # x = random.randint(marginal, width - marginal - patch_size)
	        # print(x,y)
	        y = marginal+50
	        x = marginal+50
	        # print(height,width)
	        # exit()

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

	        # Two branches. The CNN try to learn the H and inv(H) at the same time. So in the first branch, we just compute the
	        #  homography H from the original image to a perturbed image. In the second branch, we just compute the inv(H)
	        H = cv2.getPerspectiveTransform(np.float32(four_points), np.float32(perturbed_four_points))
	        H_inverse = np.linalg.inv(H)
	        warped_image = cv2.warpPerspective(image, H_inverse, (image.shape[1], image.shape[0]))
	        # cv2.imshow('image1',image)
	        # cv2.imshow('image2',warped_image)
	        # cv2.waitKey()
	        # exit()
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

	        # img_patch_ori = mold_image(img_patch_ori, Config)
	        # img_patch_pert = mold_image(img_patch_pert, Config)
	        # image_patch_pair = np.zeros((patch_size, patch_size, 2))
	        # image_patch_pair[:, :, 0] = img_patch_ori
	        # image_patch_pair[:, :, 1] = img_patch_pert
	        # image_patch_pair=image_patch_pair.transpose(2,0,1)
	        pf_patch=pf_patch.transpose(2,0,1)
	        # print(image_patch_pair.shape)
	        # scipy.misc.toimage(image_patch_pair).save(train_dir+"/"+str(file))
	        # cv2.imwrite(train_dir+"/"+str(file),image_patch_pair)

	        # input_image.append(image_patch_pair)
	        output_PF.append(pf_patch)
	        

	        # cv2.imshow('image',srcImg)
	        # # roiImg = srcImg[36:521, 180:745]
	        # print(img_patch_ori.shape)
	        # exit()
	        cv2.imwrite(pair1_path+"/"+str(file),image)
	        cv2.imwrite(pair2_path+"/_"+str(file),warped_image)
	        count +=1
	        if count == 100:
	        	break;
	np.save("output_PF.npy",np.array(output_PF,dtype=np.float32))
        # if count%400==0:
        #     print count
