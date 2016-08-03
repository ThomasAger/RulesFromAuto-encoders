# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
from keras.layers.noise import GaussianNoise
import DataTasks as dt
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, l1
from keras.optimizers import SGD, Adagrad, Adadelta, Adam, RMSprop
from keras.models import Sequential
from keras.models import model_from_json
class NeuralNetwork:

    # The shared model
    end_space = None
    model = None

    def __init__(self,
                 epochs=1, learn_rate=0.01, loss="mse", batch_size=1, decay=1e-06,
                 hidden_activation="tanh", layer_init="glorot_uniform", output_activation="tanh",  hidden_layer_size=100,
                 file_name="unspecified_filename", vector_path=None, reg=0,
                 optimizer_name="rmsprop", class_names=None, noise=0, output_weights=None):

        # Initialize the model

        self.model = Sequential()

        # Import the numpy vectors
        try:
            movie_vectors = np.asarray(np.load(vector_path))
        except OSError:
            # If it fails, assume that it's in a standard format for vectors and then save it in numpy format
            movie_vectors = dt.importVectors(vector_path)
            movie_vectors = np.asarray(movie_vectors)
            np.save(file_name, movie_vectors)

        # Set the input and the output to be the same size, as this is an auto-encoder

        input_size = len(movie_vectors[0])
        output_size = len(movie_vectors[0])

        if noise > 0: # If using a noisy autoencoder, add GaussianNoise layers to the start of the encoder
            self.model.add(GaussianNoise(noise, input_shape=(input_size,)))
            self.model.add(Dense(output_dim=hidden_layer_size,  input_dim=input_size, init=layer_init, activation=hidden_activation,W_regularizer=l2(reg)))
        else:
            # Otherwise just add the hidden layer
            self.model.add(Dense(output_dim=hidden_layer_size,  input_dim=input_size, init=layer_init, activation=hidden_activation,W_regularizer=l2(reg)))

        # If using custom weights on the hidden layer to the output layer, apply those custom weights. Otherwise just add output layer.
        if output_weights == None:
            self.model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))
        else:
            self.model.add(Dense(output_dim=len(output_weights[0]), init=layer_init, activation=output_activation, weights=output_weights))

        # Compile the model and fit it to the data
        if optimizer_name == "sgd":
            optimizer = SGD(lr=learn_rate, decay=decay)
        elif optimizer_name == "rmsprop":
            optimizer = RMSprop(lr=learn_rate)
        self.model.compile(loss=loss, optimizer=optimizer)
        self.model.fit(movie_vectors, movie_vectors, nb_epoch=epochs, batch_size=batch_size, verbose=1)

        # Create a truncated model that has no output layer that has the same weights as the previous model and use it to obtain the hidden layer representation
        truncated_model = Sequential()
        total_file_name = "newdata/spaces/" + file_name +".mds"
        truncated_model.add(GaussianNoise(noise, input_shape=(input_size,)))
        truncated_model.add(Dense(output_dim=hidden_layer_size, input_dim=input_size, init=layer_init, activation=hidden_activation, W_regularizer=l2(reg)))
        truncated_model.compile(loss=loss, optimizer=optimizer)
        self.end_space = truncated_model.predict(movie_vectors)

        np.save(self.end_space, total_file_name)


    def getEndSpace(self):
        return self.end_space

    def getEncoder(self):
        return self.model.layers[1]

def main():

    #These are the parameter values
    hidden_layer_sizes = [200,100,50,25]
    noise = 0.6

    #Class and vector inputs
    vector_path = "filmdata/films200.mds/films200.mds"
    fn = "films200"


    #NN Setup
    hidden_activation="tanh"
    output_activation="tanh"
    optimizer_name = "rmsprop"
    learn_rate = 0.01
    decay = 1e-06

    # Create the stacked auto-encoders
    SDA_end_space = []
    SDA_encoders = []

    for s in range(0, len(hidden_layer_sizes)):
        if s >= 1:
            input_size = hidden_layer_sizes[s]
            output_size = hidden_layer_sizes[s]
            SDA = NeuralNetwork(
                                 hidden_layer_size=hidden_layer_sizes[s],
                                      noise=noise, vector_path=vector_path,
                                       hidden_activation=hidden_activation, output_activation=output_activation,
                                      file_name=fn+"N"+str(noise)+"H"+ str(hidden_layer_sizes[s]) +"L"+str(s+1))
        else:
            SDA = NeuralNetwork(     noise=noise, optimizer_name=optimizer_name,
                                 vector_path=vector_path,  hidden_layer_size=hidden_layer_sizes[s],
                                       hidden_activation=hidden_activation, output_activation=output_activation,
                                      file_name=fn+"N"+str(noise)+"H"+ str(hidden_layer_sizes[s]) +"L"+str(s+1))
        SDA_end_space = SDA.getEndSpace()
        SDA_encoders.append(SDA.getEncoder())



if  __name__ =='__main__':main()

