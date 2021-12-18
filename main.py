
import os
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from models import create_swin, create_hybrid_swin, create_effnet_swin, create_vit
from utils import set_seeds, create_checkpoint_callback, create_earlystop_callback
from utils import generate_data_df_with_folds
import pandas as pd
import random
random.seed(123)

# Preprocessing function specific to base model
preproc_func_effnet = tf.keras.applications.efficientnet.preprocess_input # pass through layer
preproc_func = preproc_func_effnet

# Set seed for reproducibility
set_seeds(123)

# Create train / valid data generator
batch_size = 16
num_classes = 75
img_size = 480
train_dir = os.path.abspath(f"train_{img_size}")


datagen = ImageDataGenerator(horizontal_flip = True,
                            vertical_flip=True,
                            rotation_range = 20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
							validation_split=0.1, 
							preprocessing_function=preproc_func)

train_generator = datagen.flow_from_directory(train_dir, subset='training', batch_size = batch_size, 
											class_mode = 'categorical', target_size = (img_size, img_size))
valid_generator = datagen.flow_from_directory(train_dir, subset='validation', batch_size = batch_size, 
											class_mode = 'categorical', target_size = (img_size, img_size), shuffle=False)

# Stratified sampling
# files = generate_data_df_with_folds(kfold=10)
# train_generator = datagen.flow_from_dataframe(files[files['fold']>0],x_col='filepaths',y_col='target',
#                                 class_mode='sparse',batch_size=batch_size,target_size=(img_size,img_size))
# valid_generator = datagen.flow_from_dataframe(files[files['fold']==0],x_col='filepaths',y_col='target',
#                                 class_mode='sparse',batch_size=batch_size,target_size=(img_size,img_size),shuffle=False)

# Create Model

#model = create_swin(finetune=False)
#model = create_vit(num_classes=10)
model = create_effnet_swin()

#model.summary()
#load_model_ckpt = "checkpoints\hybridswin-v1-004.ckpt"
#model.load_weights(load_model_ckpt).expect_partial()

# Create callbacks
patience = 3
# checkpoint_name = "vitb8v1-cp-{epoch:04d}.ckpt"
checkpoint_name = "hybridswin-v1-{epoch:03d}.ckpt"
cp_callback = create_checkpoint_callback(checkpoint_name)
es_callback = create_earlystop_callback(patience=patience)

# Train the model
# model_history = model.fit(train_generator, validation_data=valid_generator, 
#                         	batch_size=batch_size, epochs=30, initial_epoch=0,
# 							callbacks=[cp_callback, es_callback])

# Stage 1: Warmup Swin
model.set_warmup(1)
model_history = model.fit(train_generator, validation_data=valid_generator, 
                        	batch_size=batch_size, epochs=1, initial_epoch=0,
							callbacks=[cp_callback, es_callback])

# Stage 2: Warmup Effnet
model.set_warmup(2)
model_history = model.fit(train_generator, validation_data=valid_generator, 
                        	batch_size=batch_size, epochs=1, initial_epoch=0,
							callbacks=[cp_callback, es_callback])

# Stage 3: Full training
model.set_warmup(3, lr=1e-4)
model_history = model.fit(train_generator, validation_data=valid_generator, 
                        	batch_size=batch_size, epochs=20, initial_epoch=0,
							callbacks=[cp_callback, es_callback])



