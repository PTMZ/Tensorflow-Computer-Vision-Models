import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import Sequence
from models import create_swin, create_effnet_swin
from utils import MySequence
import numpy as np


#checkpoint_file = 'checkpoints/swin-b384-ft001-v1-cp-012.ckpt' #3
checkpoint_file = 'checkpoints/hybridswin-v1-015.ckpt' #4
filename = "predictions4"
prediction_filename = f"{filename}.csv"
pred_t1_filename = f"{filename}_t1.csv"

img_size = 480
batch_size = 16
#test_dir = os.path.abspath("test_480")
filenames = os.listdir("test_480")

test_img_ds = MySequence(batch_size, "test_480")


# model_final = create_effnetv2(img_size = img_size)
# model = create_swin(img_size = img_size, compileloss=False)
model = create_effnet_swin(img_size=img_size)
model.set_warmup(3)

model.load_weights(checkpoint_file).expect_partial()

pred = model.predict(test_img_ds, verbose=1)


df = pd.DataFrame()
df['Id'] = [i for i in range(1, len(filenames)+1)]
top_5 = pred.argsort(axis=-1)[:,-5:]
df[['Top 5', 'Top 4', 'Top 3', 'Top 2', 'Top 1']] = top_5
df = df.reindex(columns=['Id', 'Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5'])
df.to_csv(prediction_filename, index=False)
df[['Id', 'Top 1']].to_csv(pred_t1_filename, index=False)