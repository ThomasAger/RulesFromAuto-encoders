# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np

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
from keras.models import model_from_json
import json
import csv
import pandas

class NeuralNetwork:

    # The shared model
    model = None

    # Shared variables for printing
    predicted_classes = None
    trainer = None
    compiled_model = None
    history = None
    objective_score = None
    f1_score = None
    correct_classes = 0

    def __init__(self, input_size=200, hidden_size=[400], output_size=100, training_data=10000, class_type="Genres",
                 epochs=250,  learn_rate=0.01, loss="binary_crossentropy", batch_size=100, decay=1e-06,
                 hidden_activation="relu", layer_init="glorot_uniform", output_activation="softmax", dropout_chance=0.5,
                  class_mode="binary",save_space=False, noise=0.25,
                 file_name="unspecified_filename", vector_path=None, layers_to_cut_at=None,
                 optimizer="adagrad", class_names=None, is_autoencoder=False,
                   denoising=True,  numpy_vector_path=None):

        self.model = Sequential()

        movie_names, movie_vectors, movie_labels = MovieTasks.getMovieData(class_type=class_type, class_names=class_names,  input_size=input_size, vector_path=vector_path,  numpy_vector_path=numpy_vector_path)

        if numpy_vector_path is not None:
            input_size = len(movie_vectors[0])
        if is_autoencoder:
            movie_labels = movie_vectors
            movie_labels = np.asarray(movie_labels)
            if denoising is True:
                movie_vectors = MovieTasks.maskingNoise(movie_vectors, noise)
                dt.write2dArray(movie_vectors, "newdata/spaces/noisy_"+file_name+".mds")
            movie_vectors = np.asarray(movie_vectors)
            self.createModel(self.model,dropout_chance, hidden_activation, layer_init, hidden_size,
                    input_size, output_size, output_activation)
            self.predicted_classes = self.trainNetwork(self.model, movie_vectors, movie_labels, movie_vectors,
                                     movie_labels, epochs, batch_size, learn_rate, decay, loss, class_mode, optimizer)
        else:
            n_train, x_train, y_train, n_test, x_test, y_test = dt.splitData(training_data, movie_names, movie_vectors,
                                                                             movie_labels)
            n_train = np.asarray(n_train)
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            n_test = np.array(n_test)
            x_test = np.asarray(x_test)
            y_test = np.asarray(y_test)
            self.createModel(self.model,  dropout_chance, hidden_activation, layer_init, hidden_size,
                    input_size, output_size, output_activation)
            self.predicted_classes = self.trainNetwork(self.model, x_train, y_train, x_test, y_test, epochs, batch_size, learn_rate, decay, loss, class_mode, optimizer)
            self.f1_score = f1_score(y_test, self.predicted_classes, average='macro')
            self.objective_score = self.model.evaluate(x_test, y_test, batch_size=batch_size, show_accuracy=True, verbose=1)
            movie_vectors = np.asarray(movie_vectors)
            print "F1 " + str(self.f1_score), "Correct " + str(self.correct_classes), "Objective and accuracy ", self.objective_score

        if save_space is True:
            """
            if is_autoencoder:
                total_file_name = "newdata/spaces/AUTOENCODER" + file_name +".mds"
                end_space = self.predicted_classes
                dt.write2dArray(end_space, total_file_name)
            else:
            """
            for l in layers_to_cut_at:
                total_file_name = "newdata/spaces/" + file_name + str(l) + ".mds"
                end_space = self.saveSpace(self.model, l, movie_vectors, file_name, hidden_size, input_size,
                layer_init,  hidden_activation, output_size, output_activation, dropout_chance,
                learn_rate, decay, loss, class_mode)
                dt.write2dArray(end_space, total_file_name)


    def createAutoEncoder(self, model, dropout_chance, hidden_activation, layer_init, hidden_size,
                    input_size, output_size, output_activation):
        autoencoder_amt = len(hidden_size)
        if autoencoder_amt >= 2:
            temp_model = Sequential()
            temp_model.add(Dense(output_dim=hidden_size[0],  input_dim=input_size, init=layer_init))
            temp_model.add(Dense(activation=hidden_activation, output_dim=hidden_size[0]))
            temp_model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))

            """
            for x in range(0,autoencoder_amt,2):
                encoder = containers.Sequential([Dense(output_dim=hidden_size[x], input_dim=input_size), Dense(hidden_size[x+1])])
                decoder = containers.Sequential([Dense(output_dim=hidden_size[x], input_dim=hidden_size[x+1]), Dense(input_size)])
                model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))
            """
        elif autoencoder_amt == 1:
            """
            encoder = Dense(output_dim=hidden_size[0], input_dim=input_size, activation=hidden_activation)
            decoder = Dense(output_dim=output_size, input_dim=hidden_size[0], activation=hidden_activation)
            model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))
            """
            model.add(Dense(output_dim=hidden_size[0],  input_dim=input_size, init=layer_init))
            model.add(Dense(activation=hidden_activation, output_dim=hidden_size[0]))
            model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))


    def createModel(self, model,  dropout_chance, hidden_activation, layer_init, hidden_size,
                    input_size, output_size, output_activation):
        hidden_amount = len(hidden_size)
        print hidden_amount
        if hidden_amount > 0:
            print "Startup layer added, input " + str(input_size) + " output " + str(hidden_size[0])
            model.add(Dense(output_dim=hidden_size[0], input_dim=input_size, init=layer_init))
        else:
            print "Model has no hidden layers."
            model.add(Dense(output_dim=output_size, input_dim=input_size, init=layer_init))

        for x in range(0, hidden_amount):
            if x == hidden_amount-1:
                print "Final hidden layer added, output " + str(output_size)
                model.add(Dense(activation=hidden_activation, output_dim=output_size))
            else:
                print str(x+1) + " Hidden layer added, output " + str(hidden_size[x])
                model.add(Dense(activation=hidden_activation, output_dim=hidden_size[x]))
        print "Output layer added, input " + str(hidden_size[hidden_amount-1]) + " output " + str(output_size)
        model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))

    def trainNetwork(self, model, x_train, y_train, x_test, y_test, epochs, batch_size, learn_rate, decay, loss, class_mode, optimizer):
        if optimizer is "adagrad":
            optimizer = Adagrad(lr=learn_rate, epsilon=decay)
        model.compile(loss=loss, optimizer=optimizer, class_mode=class_mode)
        self.history = model.fit(x_train, y_train, nb_epoch=epochs, batch_size=batch_size, verbose=1)
        predicted_classes = model.predict_classes(x_test, batch_size=batch_size)
        return predicted_classes


    def saveSpace(self, model, layer_to_cut_at, movie_vectors, file_name, hidden_size, input_size,
                  layer_init,  hidden_activation, output_size, output_activation, dropout_chance,
                  learn_rate, decay, loss, class_mode):
        hidden_amount = len(hidden_size)
        print "Saving space for layer " + str(layer_to_cut_at)
        truncated_model = Sequential()
        if hidden_amount > 0:
            truncated_model.add(Dense(output_dim=hidden_size[0], input_dim=input_size, init=layer_init, weights=model.layers[0].get_weights()))
        else:
            truncated_model.add(Dense(output_dim=output_size, input_dim=input_size, init=layer_init, weights=model.layers[0].get_weights()))

        if layer_to_cut_at == 0:
            truncated_model.compile(loss=loss, optimizer='sgd', class_mode=class_mode)
            return truncated_model.predict(movie_vectors)
        else:
            for x in range(1, hidden_amount):
                truncated_model.add(Dense(activation=hidden_activation, output_dim=hidden_size[x-1], weights=model.layers[x].get_weights()))
                if layer_to_cut_at == x:
                    print "Saving at hidden layer " + str(layer_to_cut_at)
                    truncated_model.compile(loss=loss, optimizer='sgd', class_mode=class_mode)
                    prediction = truncated_model.predict(movie_vectors)
                    return prediction
        truncated_model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))
        trainer = Adagrad(lr=learn_rate, epsilon=decay)
        truncated_model.compile(loss=loss, optimizer=trainer, class_mode=class_mode)
        return truncated_model.predict(movie_vectors)




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


    class_type = "Genres"
    loss = "mse"
    optimizer="sgd"
    epochs=120
    hidden_activation="sigmoid"
    output_activation="sigmoid"
    layers_to_cut_at=[1]
    save_space=True
    denoising = True
    output_size = 200
    input_size = 200
    is_autoencoder = True
    hidden_size = [200]
    numpy_vector_path=None#"D:\Dropbox\PhD\My Work\Code\MSDA\Python\Data\IMDB\Transformed/" + "NORMALIZEDno_below200no_above0.25.npy"

    dimension_100 = NeuralNetwork( epochs=epochs, is_autoencoder=is_autoencoder, input_size=input_size, denoising=denoising,
                                  output_size=output_size, class_type=class_type, numpy_vector_path=numpy_vector_path, noise=0.25,
                                save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer,
                                   hidden_activation=hidden_activation, output_activation=output_activation, hidden_size=hidden_size,
                                  file_name=hidden_activation+output_activation + loss+ str(epochs) + hidden_activation + str(hidden_size))

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

