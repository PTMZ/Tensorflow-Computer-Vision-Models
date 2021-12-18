import os
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np
import random
import math
from skimage.io import imread

def set_seeds(seed=123):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def unfreeze_layers(idx, base_model):
	base_model.trainable = True
	for layer in base_model.layers:
		if layer not in base_model.layers[idx:]:
			layer.trainable = False
		if layer.name.startswith('bn'):
			layer.call(layer.input, training=False)

def create_checkpoint_callback(checkpoint_name):
    checkpoint_path = f"checkpoints/{checkpoint_name}"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    
    return cp_callback

def create_earlystop_callback(patience=3):
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=patience, 
        verbose=0
        )
    return es_callback


class MySequence(Sequence):
    def __init__(self, batch_size, folder):
        self.batch_size = batch_size
        self.folder = folder
        self.filenames = os.listdir(folder)
        self.filenames.sort(key=lambda x: int(x.split('.')[0]))
        self.max_len = len(self.filenames)
    
    def __len__(self):
        return math.ceil(self.max_len / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx+1)* self.batch_size, self.max_len)
        return np.array([imread(f'{self.folder}/{self.filenames[i]}') for i in range(start, end)])
    

# For Stratified Sampling. Messes with labels and while it trained well, it did not perform well in the test set.
def generate_data_df_with_folds(kfold=10):
	files = pd.DataFrame()
	tmp = []
	labels_1 = []
	for class_id in os.listdir(f"train_{img_size}"):
		for img_id in os.listdir(f"train_{img_size}/{class_id}"):
			tmp.append(f"train_{img_size}/{class_id}/{img_id}")
			labels_1.append(int(class_id))


	files['filepaths'] = pd.Series(tmp)
	files['target'] = pd.Series(labels_1)

	folds = []
	for i in range(75):
		n = files[files['target']==i].shape[0]
		tmp = []
		for fold in range(kfold):
			if fold != kfold-1:
				tmp += [fold]*(n//kfold)
			else:
				tmp+= [fold]*(n-len(tmp))
		random.shuffle(tmp)
		folds+=tmp
	files['fold'] = folds
	files['target'] = files['target'].astype(str)
	return files
