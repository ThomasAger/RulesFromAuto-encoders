# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
from keras.layers.noise import GaussianNoise
import DataTasks as dt
from keras.layers.core import Dense, Activation, Dropout, AutoEncoder
from keras.layers import containers
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, l1
from keras.optimizers import SGD, Adagrad, Adadelta, Adam
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.models import model_from_json
class NeuralNetwork:

    # The shared model
    end_space = None


    def __init__(self,  output_size=100, training_data=10000, class_path="",
                 epochs=1,  learn_rate=0.01, loss="mse", batch_size=1, decay=1e-06,
                 hidden_activation="tanh", layer_init="glorot_uniform", output_activation="tanh",
                  class_mode="categorical", autoencoder_space = None, encoders=None, hidden_layer_size=100,
                 file_name="unspecified_filename", vector_path=None, layers_to_cut_at=1, reg=0,
                 optimizer="rmsprop", class_names=None, noise=0):

        model = Sequential()

        movie_vectors = np.asarray(dt.importVectors(vector_path))

        input_size = len(movie_vectors[0])
        output_size = len(movie_vectors[0])

        if noise > 0: # If using a noisy autoencoder, add GaussianNoise layers to the start of the encoder
            self.encoder = containers.Sequential([
                GaussianNoise(noise, input_shape=(input_size,)),
                Dense(output_dim=hidden_layer_size,  input_dim=input_size, init=layer_init, activation=hidden_activation,W_regularizer=l2(reg)),])
        else:
            self.encoder = Dense(output_dim=hidden_layer_size,  input_dim=input_size, init=layer_init, activation=hidden_activation,W_regularizer=l2(reg))
        decoder = Dense(output_dim=output_size, init=layer_init, activation=output_activation)
        model.add(AutoEncoder(encoder=self.encoder, decoder=decoder,
                       output_reconstruction=False))

        model.compile(loss=loss, optimizer=optimizer, class_mode=class_mode)
        history = model.fit(movie_vectors, movie_vectors, nb_epoch=epochs, batch_size=batch_size, show_accuracy=True, verbose=1)

        truncated_model = Sequential()
        total_file_name = "newdata/spaces/" + file_name +".mds"
        truncated_model.add(self.encoder)
        truncated_model.compile(loss=loss, optimizer="sgd", class_mode=class_mode)
        self.end_space = truncated_model.predict(movie_vectors)
        dt.write2dArray(self.end_space, total_file_name)


    def getEndSpace(self):
        return self.end_space

    def getEncoder(self):
        return self.encoder

def main():

    #These are the parameter values
    hidden_layer_sizess = [75,50,25,50,75,100]
    noise = 0.6

    #Class and vector inputs
    vector_path = "filmdata/films100.mds/films100.mds"
    fn = "films100"

    #NN Setup
    hidden_activation="tanh"
    output_activation="tanh"
    reg = 0.0

    # Create the stacked auto-encoders
    SDA_end_space = []
    SDA_encoders = []
    for s in range(0, len(hidden_layer_sizess)):
        if s >= 1:
            input_size = hidden_layer_sizess[s]
            output_size = hidden_layer_sizess[s]
            SDA = NeuralNetwork( autoencoder_space=SDA_end_space, reg=reg,
                                 hidden_layer_size=hidden_layer_sizess[s],
                                      noise=noise, vector_path=vector_path,
                                       hidden_activation=hidden_activation, output_activation=output_activation,
                                      file_name=fn+"N"+str(noise)+"H"+ str(hidden_layer_sizess[s]) +"L"+str(s+1))
        else:
            SDA = NeuralNetwork( autoencoder_space= None,    noise=noise, reg=reg,
                                 vector_path=vector_path,  hidden_layer_size=hidden_layer_sizess[s],
                                       hidden_activation=hidden_activation, output_activation=output_activation,
                                      file_name=fn+"N"+str(noise)+"H"+ str(hidden_layer_sizess[s]) +"L"+str(s+1))
        SDA_end_space = SDA.getEndSpace()
        SDA_encoders.append(SDA.getEncoder())



if  __name__ =='__main__':main()

