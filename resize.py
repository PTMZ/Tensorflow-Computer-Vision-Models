import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm, trange

img_size = 480

def resize_crop(img, img_size):
    w, h = img.shape[1], img.shape[0]
    if w > h:
        factor = h / img_size
        new_w = int(w / factor)
        img = cv2.resize(img, (new_w, img_size))
        left = new_w // 2 - img_size // 2
        right = new_w // 2 + img_size // 2
        img_cropped = img[:,left:right,:]
    elif w < h:
        factor = w / img_size
        new_h = int(h / factor)
        img = cv2.resize(img, (img_size, new_h))
        up = new_h // 2 - img_size // 2
        down = new_h // 2 + img_size // 2
        img_cropped = img[up:down,:,:]
    else:
        img_cropped = cv2.resize(img, (img_size, img_size))
    
    return img_cropped


def resize_test():
    # path to the folder containing images
    path = f'./test'

    # path to the folder where resized images will be stored
    path_to_save = f'./test_{img_size}'
    os.mkdir(path_to_save)

    # list of all images in the folder
    images = [x for x in os.listdir(path) if x[-3:] == 'png']

    # iterate over all images
    for image in tqdm(images):
        # read image
        img = cv2.imread(os.path.join(path, image))
        orig_shape = img.shape
        # resize image
        img = resize_crop(img, img_size)

        # save image
        cv2.imwrite(os.path.join(path_to_save, image), img)


def resize_train():
    for i in range(0, 75):
        print(f'Folder {i:02d} processing...')
        # path to the folder containing images
        path = f'./train/{i}'

        # path to the folder where resized images will be stored
        path_to_save = f'./train_{img_size}/{i:02d}'
        os.mkdir(path_to_save)

        # list of all images in the folder
        images = [x for x in os.listdir(path) if x[-3:] == 'png']

        # iterate over all images
        for image in tqdm(images):
            # read image
            img = cv2.imread(os.path.join(path, image))
            # resize image
            img = resize_crop(img, img_size)
            # save image
            cv2.imwrite(os.path.join(path_to_save, image), img)


resize_train()
resize_test()