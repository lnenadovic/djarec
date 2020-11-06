import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


df = pd.read_csv('ml-20m/ratings.csv')

df.userId = pd.Categorical(df.userId)
df['new_user_id'] = df.userId.cat.codes
df.movieId = pd.Categorical(df.movieId)
df['new_movie_id'] = df.movieId.cat.codes

user_ids = df['new_user_id'].values
movie_ids = df['new_movie_id'].values
ratings = df['rating'].values

N = len(set(user_ids))
M = len(set(movie_ids))
D = 10 # embedding dimension

user_input = Input(shape = (1, ))
movie_input = Input(shape = (1, ))
user_emb = Embedding(N, D)(user_input) # num samples x 1 x D
movie_emb = Embedding(M, D)(movie_input)
user_emb = Flatten()(user_emb) # num samples x D
movie_emb = Flatten()(movie_emb)
x = Concatenate()([user_emb, movie_emb]) # num samples x 2D
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.1)(x)
x = Dense(256, activation = 'relu')(x)
x = Dropout(0.1)(x)
x = Dense(1)(x)

model = Model(inputs = [user_input, movie_input],  outputs = x)
model.compile(loss = 'mse', optimizer = SGD(lr = 0.07, momentum = 0.9))

user_ids, movie_ids, ratings = shuffle(user_ids, movie_ids, ratings)
Ntrain = int(0.8 * len(ratings))
train_user = user_ids[:Ntrain]
train_movie = movie_ids[:Ntrain]
train_ratings = ratings[:Ntrain]
test_user = user_ids[Ntrain:]
test_movie = movie_ids[Ntrain:]
test_ratings = ratings[Ntrain:]

avg_rating = train_ratings.mean()
train_ratings = train_ratings - avg_rating
test_ratings = test_ratings - avg_rating

callback = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(x = [train_user, train_movie], y = train_ratings, epochs = 25,
              batch_size = 512, verbose = 2, callbacks = [callback],
              validation_data =([test_user, test_movie], test_ratings))

plt.plot(history.history['loss'], label = 'train_loss')
plt.plot(history.history['val_loss'], label = 'validation_loss')
plt.legend()
plt.show()

print(avg_rating)

model.save('saved_model.h5', overwrite = True)
#loaded_model = tf.keras.models.load_model('/tmp/model')
