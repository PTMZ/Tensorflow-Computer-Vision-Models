import os
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
import numpy as np
import math
from skimage.io import imread
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

class MySequence(Sequence):
    def __init__(self, batch_size, max_len):
        self.max_len = max_len
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.max_len / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx+1)* self.batch_size, self.max_len)
        return np.array([imread(f'test/{i}.png') for i in range(start, end)])

# Declaring Constants
IMAGE_PATH = "test/0.png"
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def preprocess_image(image_path):
	hr_image = tf.image.decode_image(tf.io.read_file(image_path))
	if hr_image.shape[-1] == 4:
		hr_image = hr_image[...,:-1]
	hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
	hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
	hr_image = tf.cast(hr_image, tf.float32)
	return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
	if not isinstance(image, Image.Image):
		image = tf.clip_by_value(image, 0, 255)
		image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
	image.save("%s.png" % filename)
	#print("Saved as %s.png" % filename)

def plot_image(image, title=""):
	image = np.asarray(image)
	image = tf.clip_by_value(image, 0, 255)
	image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
	plt.imshow(image)
	plt.axis("off")
	plt.title(title)
	plt.show()


batch_size = 32
max_len = 13000
test_img_seq = MySequence(batch_size, max_len)

model = hub.load(SAVED_MODEL_PATH)

cur_idx = 0
for data in tqdm(test_img_seq, total=len(test_img_seq)):
	tensor_data = tf.cast(data, tf.float32)
	fake_imgs = model(tensor_data)
	for img in fake_imgs:
		save_image(img, f'sr/{cur_idx}')
		cur_idx += 1

