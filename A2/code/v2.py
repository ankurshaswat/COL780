import argparse
import os
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt

print('Is CV3 Or Better = ', imutils.is_cv3(or_better=True))

FEATURE_EXTRACTOR = 'orb'  # 'sift'|'surf'|'brisk'|'orb'
GOOD_MATCH_PERCENT = 0.15


def define_args():
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument("-i", "--images", type=str, default="../InSample",
							help="path to directory containing image group directories")
	arg_parser.add_argument("-o", "--output", type=str, default="OutSample",
							help="path to directory to output combined images")
	return arg_parser


def load_images(folder_path):
	image_paths = os.listdir(folder_path)
	images = []
	for image_rel_path in image_paths:
		image_path = folder_path + '/' + image_rel_path
		image = cv2.imread(image_path)
		images.append(image)
	return images


def convert_to_grayscale(image):
	return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def get_descriptors(image):
	if FEATURE_EXTRACTOR == 'surf':
		descriptor = cv2.xfeatures2d.SURF_create()
	elif FEATURE_EXTRACTOR == 'sift':
		descriptor = cv2.xfeatures2d.SIFT_create()
	elif FEATURE_EXTRACTOR == 'brisk':
		descriptor = cv2.BRISK_create()
	elif FEATURE_EXTRACTOR == 'orb':
		descriptor = cv2.ORB_create()

	(keypoints, features) = descriptor.detectAndCompute(image, None)
	return (keypoints, features)


def create_matcher():
	matcher = cv2.DescriptorMatcher_create(
		cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
	return matcher

def Average(lst): 
	sum_ = 0
	for i in lst:
		sum_ += i.distance
	return sum_ / len(lst) 

def get_matches(kp1, dsc1, kp2, dsc2):
	matches = matcher.match(dsc1, dsc2, None)

	matches.sort(key=lambda x: x.distance, reverse=False)

	goodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
	matches = matches[:goodMatches]

	# imMatches = cv2.drawMatches(images[img1_ind], keypoints1, images[img2_ind], keypoints2, matches, None)
	# cv2.imshow("matches.jpg", imMatches)
	# cv2.waitKey(0)

	points1 = np.zeros((len(matches), 2), dtype=np.float32)
	points2 = np.zeros((len(matches), 2), dtype=np.float32)

	for i, match in enumerate(matches):
		points1[i, :] = kp1[match.queryIdx].pt
		points2[i, :] = kp2[match.trainIdx].pt

	return Average(matches), points1, points2,matches


args = vars(define_args().parse_args())

print("[INFO] loading images...")
image_grp_folders = sorted(os.listdir(args["images"]))
print('Found following image groups -', image_grp_folders)

matcher = create_matcher()

for image_grp in image_grp_folders:
	final_image_name = image_grp+'.jpg'
	print('[INFO] working on image - '+final_image_name)
	images = load_images(args['images']+'/'+image_grp)
	grayscale_imgs = [convert_to_grayscale(x) for x in images]
	described_imgs = [get_descriptors(x) for x in grayscale_imgs]

	pairwise_matches = []

	img1, img1_grayscale, img1_described = images[0], grayscale_imgs[0], described_imgs[0]
	(kp1,dsc1) = img1_described

	images.pop(0)
	grayscale_imgs.pop(0)
	described_imgs.pop(0)

	while(len(described_imgs) > 0):
		min_dist = 10000
		best_match = 0
		best_match_data = (0,0)
		for img2_ind in range(len(described_imgs)):
			(kp2,dsc2) = described_imgs[img2_ind]
			avg_distance, points1, points2,matches = get_matches(
				kp1, dsc1, kp2, dsc2)
			# print(avg_distance)
			if avg_distance < min_dist:
				best_match = img2_ind
				min_dist = avg_distance
				best_match_data = (points1,points2,matches)

		img2, img2_grayscale, img2_described = images[best_match], grayscale_imgs[best_match] ,described_imgs[best_match]
		(kp2,dsc2) = img2_described

		images.pop(best_match)
		grayscale_imgs.pop(best_match)
		described_imgs.pop(best_match)

		img3 = cv2.drawMatches(img1,kp1,img2,kp2,best_match_data[2],
						None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
		plt.imshow(img3)
		plt.show()

		h, mask = cv2.findHomography(best_match_data[0], best_match_data[1], cv2.RANSAC)
		height, width, channels = img2.shape

		# Apply panorama correction
		width = img1.shape[1] + img2.shape[1]
		height = img1.shape[0] + img2.shape[0]

		img1 = cv2.warpPerspective(img1, h, (width, height))
		img1[0:img2.shape[0], 0:img2.shape[1]] = img2

		gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		c = max(cnts, key=cv2.contourArea)

		(x, y, w, h) = cv2.boundingRect(c)

		img1 = img1[y:y + h, x:x + w]
		img1_grayscale = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		img1_described = get_descriptors(img1_grayscale)

		plt.imshow(img1)
		plt.show()



		
	# for img1_ind in range(len(described_imgs)):
	# 	for img2_ind in range(len(described_imgs)):
	# 		if(img1_ind <= img2_ind):
	# 			continue
	# 		else:
	# 			(keypoints1, descriptors1) = described_imgs[img1_ind]
	# 			(keypoints2, descriptors2) = described_imgs[img2_ind]

	# 			min_distance, points1, points2 = get_matches(
	# 				keypoints1, descriptors1, keypoints2, descriptors2)

	# 			h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

	# 			pairwise_matches.append(
	# 				(img1_ind, img2_ind, min_distance, h, mask))

# References
# https://colab.research.google.com/drive/11Md7HWh2ZV6_g3iCYSUw76VNr4HzxcX5#scrollTo=6eHgWAorE9gf
# https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/

#    # Use homography
#   height, width, channels = im2.shape
#   im1Reg = cv2.warpPerspective(im1, h, (width, height))
