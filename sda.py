import numpy as np

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import containers
from keras.layers.noise import GaussianNoise
from keras.layers.core import Dense, AutoEncoder
from keras.utils import np_utils
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score)
import MovieTasks as mt
import DataTasks as dt
training_data = 10000
np.random.seed(1337)

class_type="Genres"
input_size=200
class_path = None
class_names = None
class_by_class = None
vector_path = None


movie_names, movie_vectors, movie_labels = mt.getMovieData(class_type=class_type, input_size=input_size, class_path=class_path,
                                                                   class_names=class_names, class_by_class=class_by_class, vector_path=vector_path)

n_train, x_train, y_train, n_test, x_test, y_test = dt.splitData(training_data, movie_names, movie_vectors,  movie_labels)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
X_train_tmp = np.copy(x_train)
max_len = 800
max_words = 20000
batch_size = 64
nb_classes = 2
nb_epoch = 2
nb_hidden_layers = [200, 200, 200, 200]
nb_noise_layers = [0.6, 0.4, 0.3, ]

print('Train: {}'.format(x_train.shape))
print('Test: {}'.format(x_test.shape))

trained_encoders = []
for i, (n_in, n_out) in enumerate(
        zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):

    print('Pre-training the layer: Input {} -> Output {}'
          .format(n_in, n_out))

    ae = Sequential()
    encoder = containers.Sequential([
        GaussianNoise(nb_noise_layers[i - 1], input_shape=(n_in,)),
        Dense(input_dim=n_in, output_dim=n_out, activation='sigmoid'),
    ])
    decoder = Dense(input_dim=n_out, output_dim=n_in, activation='sigmoid')
    ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
                       output_reconstruction=False))
    ae.compile(loss='mean_squared_error', optimizer='rmsprop')
    ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch)
    trained_encoders.append(ae.layers[0].encoder)
    X_train_tmp = ae.predict(X_train_tmp)

model = Sequential()
for encoder in trained_encoders:
    model.add(encoder)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

y_pred = model.predict_classes(movie_vectors)
dt.write2dArray(y_pred, "newdata/spaces/sdapy_200.mds")
print model.predict_classes(movie_vectors)

