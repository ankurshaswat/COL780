import argparse
import os
import cv2
import imutils

print('Is CV3 Or Better = ',imutils.is_cv3(or_better=True))

FEATURE_EXTRACTOR = 'orb' # 'sift'|'surf'|'brisk'|'orb'
FEATURE_MATCHING = 'bf'

def define_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--images", type=str, default="InSample",
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
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

def get_descriptors(image):
    if FEATURE_EXTRACTOR == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif FEATURE_EXTRACTOR == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif FEATURE_EXTRACTOR == 'brisk':
        descriptor = cv2.BRISK_create()
    elif FEATURE_EXTRACTOR == 'orb':
        descriptor = cv2.ORB_create()

    (keypoints,features) = descriptor.detectAndCompute(image,None)
    return (keypoints,features)

args = vars(define_args().parse_args())

print("[INFO] loading images...")
image_grp_folders = sorted(os.listdir(args["images"]))
print('Found following image groups -', image_grp_folders)

for image_grp in image_grp_folders:
    final_image_name = image_grp+'.jpg'
    print('[INFO] working on image - '+final_image_name)
    images = load_images(args['images']+'/'+image_grp)
    grayscale_imgs = [convert_to_grayscale(x) for x in images]
    described_imgs = [get_descriptors(x) for x in grayscale_imgs]