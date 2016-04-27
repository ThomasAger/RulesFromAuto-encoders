# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
from keras.layers.noise import GaussianNoise
import DataTasks as dt
import MovieTasks
from keras.layers.core import Dense, Activation, Dropout, AutoEncoder
from keras.layers import containers
from keras.regularizers import activity_l2
from keras.optimizers import SGD, Adagrad, Adadelta, Adam
import theano
import os
from sklearn.metrics import f1_score
from keras.models import Sequential
import SVM
from keras.models import model_from_json
import json
import csv
import pandas

class NeuralNetwork:

    # The shared model
    model = None
    end_space = None


    # Shared variables for printing
    predicted_classes = None
    trainer = None
    compiled_model = None
    history = None
    objective_score = None
    f1_score = None
    encoder = None
    correct_classes = 0

    def __init__(self, input_size=200, hidden_size=[400], output_size=100, training_data=10000, class_type="Genres",
                 epochs=250,  learn_rate=0.01, loss="binary_crossentropy", batch_size=100, decay=1e-06,
                 hidden_activation="relu", layer_init="glorot_uniform", output_activation="softmax", dropout_chance=0.5,
                  class_mode="binary",save_space=False, autoencoder_space = None, encoders=None,
                 file_name="unspecified_filename", vector_path=None, layers_to_cut_at=None,
                 optimizer="adagrad", class_names=None, is_autoencoder=False, noise=0,  numpy_vector_path=None):

        self.model = Sequential()

        if autoencoder_space is not None:
            movie_vectors = autoencoder_space
        else:
            movie_vectors, movie_labels = MovieTasks.getMovieData(class_type=class_type, class_names=class_names,  input_size=input_size, vector_path=vector_path,  numpy_vector_path=numpy_vector_path)

        if is_autoencoder:
            movie_labels = movie_vectors

        input_size = len(movie_vectors[0])
        output_size = len(movie_labels[0])
        print input_size, output_size
        print movie_labels[0]

        x_train, y_train, x_test, y_test = dt.splitData(training_data, movie_vectors, movie_labels)
        movie_labels = np.asarray(movie_labels)
        movie_vectors = np.asarray(movie_vectors)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        if encoders is not None:
            for e in encoders:
                self.model.add(e)
            self.model.add(Dense(output_dim=output_size, input_dim=hidden_size[0], init=layer_init, activation=output_activation))
        elif is_autoencoder:
            if noise > 0:
                self.encoder = containers.Sequential([
                    GaussianNoise(noise, input_shape=(input_size,)),
                    Dense(output_dim=hidden_size[0],  input_dim=input_size, init=layer_init, activation=hidden_activation),])
            else:
                self.encoder = Dense(output_dim=hidden_size[0],  input_dim=input_size, init=layer_init, activation=hidden_activation)
            decoder = Dense(output_dim=output_size, init=layer_init)
            self.model.add(AutoEncoder(encoder=self.encoder, decoder=decoder,
                           output_reconstruction=False))
        else:
            hidden_amount = len(hidden_size)
            print hidden_amount
            if hidden_amount > 0:
                self.model.add(Dense(output_dim=hidden_size[0], input_dim=input_size, init=layer_init))
            else:
                self.model.add(Dense(output_dim=output_size, input_dim=input_size, init=layer_init))

            for x in range(0, hidden_amount):
                if x == hidden_amount-1:
                    self.model.add(Dense(activation=hidden_activation, output_dim=output_size))
                else:
                    self.model.add(Dense(activation=hidden_activation, output_dim=hidden_size[x]))
            self.model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))

        if optimizer is "adagrad":
            optimizer = Adagrad(lr=learn_rate, epsilon=decay)
        print input_size, output_size
        print x_train[0]
        print y_train[0]
        self.model.compile(loss=loss, optimizer=optimizer, class_mode=class_mode)
        self.history = self.model.fit(x_train, y_train, nb_epoch=epochs, batch_size=batch_size, show_accuracy=True, verbose=1)
        self.predicted_classes = self.model.predict_classes(x_test, batch_size=batch_size)

        if is_autoencoder is False:
            #self.f1_score = f1_score(y_test, self.predicted_classes, average='macro')
            self.objective_score = self.model.evaluate(x_test, y_test, batch_size=batch_size, show_accuracy=True, verbose=1)
            dt.write1dArray(["Objective and accuracy ", self.objective_score], "scores/"+file_name+"score.txt")
            print "F1 " + str(self.f1_score), "Correct " + str(self.correct_classes),  ["Objective and accuracy ", self.objective_score]

        if save_space is True:
            truncated_model = Sequential()
            if encoders is not None:
                total_file_name = "newdata/spaces/AUTOENCODER" + file_name +"FINETUNED.mds"
                for e in range(len(encoders)):
                    weights = self.model.layers[e].get_weights()
                    truncated_model.add(encoders[e])
                    truncated_model.layers[e].set_weights(weights)
                truncated_model.compile(loss=loss, optimizer="sgd", class_mode=class_mode)
            elif is_autoencoder:
                total_file_name = "newdata/spaces/AUTOENCODER" + file_name +".mds"
                truncated_model.add(self.encoder)
                truncated_model.compile(loss=loss, optimizer="sgd", class_mode=class_mode)
            else:
                for l in layers_to_cut_at:
                    total_file_name = "newdata/spaces/" + file_name + str(l) + ".mds"
                    hidden_amount = len(hidden_size)
                    print "Saving space for layer " + str(l)
                    truncated_model = Sequential()
                    if hidden_amount > 0:
                        truncated_model.add(Dense(output_dim=hidden_size[0], input_dim=input_size, weights=self.model.layers[0].get_weights()))
                    else:
                        truncated_model.add(Dense(output_dim=output_size, input_dim=input_size, weights=self.model.layers[0].get_weights()))
                    if l == 0:
                        truncated_model.compile(loss=loss, optimizer='sgd', class_mode=class_mode)
                        break
                    else:
                        for x in range(1, hidden_amount):
                            truncated_model.add(Dense(activation=hidden_activation, output_dim=hidden_size[x-1], weights=self.model.layers[x].get_weights()))
                            if l == x:
                                print "Saving at hidden layer " + str(l)
                                truncated_model.compile(loss=loss, optimizer='sgd', class_mode=class_mode)
                                break
                    truncated_model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))
                    trainer = Adagrad(lr=learn_rate, epsilon=decay)
                    truncated_model.compile(loss=loss, optimizer=trainer, class_mode=class_mode)
            self.end_space = truncated_model.predict(movie_vectors)
            dt.write2dArray(self.end_space, total_file_name)

    def getEndSpace(self):
        return self.end_space

    def getEncoder(self):
        return self.encoder

def main():

    # Next, try autoencoders of different sizes.
    # Once we find a good one, extrapolate.
    """
    class_type = "All"
    output_size = 125
    input_size=200
    epochs=20
    layers_to_cut_at=[1]
    save_space=True
    denoising = False

    hidden_activation="sigmoid"
    hidden_size = [100]
    dimension_100 = NeuralNetwork(input_size=input_size, epochs=epochs, hidden_size=hidden_size, denoising=denoising,
                                  output_size=output_size, class_type=class_type, vector_path="newdata/spaces/AUTOENCODERnoise020sigmoid.mds",
                                save_space=save_space, layers_to_cut_at=layers_to_cut_at, hidden_activation=hidden_activation,
                                  file_name="Autoencoder_AdjustedIS"+str(input_size)+"HS"+ str(hidden_size)+ "OS" + str(output_size) + class_type)
    """



    for amount_of_sda in range(1, 10):
        class_type = "All"
        loss = "mse"
        optimizer="rmsprop"
        epochs=2
        hidden_activation="sigmoid"
        output_activation="sigmoid"
        layers_to_cut_at=[0]
        save_space=True
        batch_size=64
        class_mode="categorical"
        is_autoencoder = True
        hidden_size = [1000]
        noise = 1
        numpy_vector_path="D:\Dropbox\PhD\My Work\Code\MSDA\Python\Data\IMDB\Transformed/" + "NORMALIZEDno_below200no_above0.25.npy"
        input_size=200
        output_size=25
        amount_of_sda =1
        SDA_end_space = []
        SDA_encoders = []
        for s in range(0, amount_of_sda):
            if s >= 1:
                epochs=8
                output_activation="linear"
                input_size = hidden_size[0]
                output_size = hidden_size[0]
                hidden_size = [int(hidden_size[0]-(hidden_size[0]/amount_of_sda+1))]
                print input_size, hidden_size, output_size
                SDA = NeuralNetwork( autoencoder_space=SDA_end_space,
                        epochs=epochs, is_autoencoder=is_autoencoder,  class_mode=class_mode,
                                           class_type=class_type, numpy_vector_path=numpy_vector_path, noise=noise,
                                        save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer,
                                           hidden_activation=hidden_activation, output_activation=output_activation, hidden_size=hidden_size,
                                          file_name=str(noise)+hidden_activation+output_activation + loss+ str(epochs) + hidden_activation +
                                                    str(hidden_size)  + str(amount_of_sda)+ "SDA" + str(s+1))
            else:
                output_activation="linear"
                SDA = NeuralNetwork( autoencoder_space= None, epochs=epochs, is_autoencoder=is_autoencoder,  class_mode=class_mode, input_size=input_size, output_size=output_size,
                                         class_type=class_type, numpy_vector_path=numpy_vector_path, noise=noise,
                                        save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer,
                                           hidden_activation=hidden_activation, output_activation=output_activation, hidden_size=hidden_size,
                                          file_name=str(noise)+hidden_activation+output_activation + loss+ str(epochs) + hidden_activation + str(hidden_size) + str(amount_of_sda) + "SDA" + str(s+1))

            SDA_end_space = SDA.getEndSpace()
            SDA_encoders.append(SDA.getEncoder())
        optimizer="adagrad"
        loss="binary_crossentropy"
        epochs=2
        is_autoencoder = False
        autoencoder_space = None
        layers_to_cut_at = [0,1,2,3]
        output_size=125
        input_size=200
        output_activation="softmax"
        print "FINE TUNING"
        class_mode="binary"
        SDA = NeuralNetwork( autoencoder_space= autoencoder_space, encoders=SDA_encoders,
                        epochs=epochs, is_autoencoder=is_autoencoder, input_size=input_size, class_mode=class_mode,
                                          output_size=output_size, class_type=class_type, numpy_vector_path=numpy_vector_path, noise=noise,
                                        save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer,
                                           hidden_activation=hidden_activation,  output_activation=output_activation, hidden_size=hidden_size,
                                          file_name=str(noise)+hidden_activation+output_activation + loss+ str(epochs) + hidden_activation + str(hidden_size) + str(amount_of_sda))


    """
    #WORKING @ HIGHEST ACCURACY, F1, ETC FOR MULTI-LABEL, FOLLOWING "REVISITING MULTI-LABEL NEURAL NETS" GUIDELINES
    class_type = "Genres"
    loss = "binary_crossentropy"
    optimizer="adagrad"
    epochs=300
    hidden_activation="relu"
    output_activation="sigmoid"
    layers_to_cut_at=[1,2]
    save_space = True
    denoising = False
    output_size = 25
    input_size = 200
    is_autoencoder = False
    hidden_size = [2000,50]

    numpy_vector_path=None
    class_type = "Genres"

    dimension_100 = NeuralNetwork( epochs=epochs, is_autoencoder=is_autoencoder, input_size=input_size, denoising=denoising,
                                  output_size=output_size, class_type=class_type, hidden_size=hidden_size, numpy_vector_path=numpy_vector_path,
                                save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer,
                                   hidden_activation=hidden_activation, output_activation=output_activation,
                                  file_name="GenresUVInputrelu,softmax,bc,adagrad" + str(epochs) + hidden_activation + str(hidden_size))

    class_type = "Keywords"

    dimension_100 = NeuralNetwork( epochs=epochs, is_autoencoder=is_autoencoder, input_size=input_size, denoising=denoising,
                                  output_size=output_size, class_type=class_type, hidden_size=hidden_size, numpy_vector_path=numpy_vector_path,
                                save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer,
                                   hidden_activation=hidden_activation, output_activation=output_activation,
                                  file_name="Keywordsrelu,softmax,bc,adagrad" + str(epochs) + hidden_activation + str(hidden_size))

    class_type = "All"

    dimension_100 = NeuralNetwork( epochs=epochs, is_autoencoder=is_autoencoder, input_size=input_size, denoising=denoising,
                                  output_size=output_size, class_type=class_type, hidden_size=hidden_size, numpy_vector_path=numpy_vector_path,
                                save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer,
                                   hidden_activation=hidden_activation, output_activation=output_activation,
                                  file_name="Allrelu,softmax,bc,adagrad" + str(epochs) + hidden_activation + str(hidden_size))
    """







if  __name__ =='__main__':main()

