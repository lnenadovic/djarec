import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


def predicted_rating(user_id_input, item_id_input):
    avg_rating = 0.525751849506975
    model = tf.keras.models.load_model("saved_model.h5")
    prediction = model.predict([np.array([user_id_input]), np.array([item_id_input])]) + avg_rating
    return prediction
