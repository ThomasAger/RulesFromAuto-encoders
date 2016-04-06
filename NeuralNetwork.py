# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np

import DataTasks as dt
import MovieTasks
from keras.layers.core import Dense, Activation, Dropout, AutoEncoder
from keras.layers import containers
from keras.optimizers import SGD, Adagrad, Adadelta, Adam
import theano
import os
from sklearn.metrics import f1_score, precision_score, recall_score
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
                 epochs=250,  learn_rate=0.01, loss="binary_crossentropy", hidden_amount=1, batch_size=100, decay=1e-06,
                 hidden_activation="tanh", layer_init="glorot_uniform", output_activation="sigmoid", dropout_chance=0.5,
                  class_mode="binary", save_weights=False, save_space=False, save_architecture=False,
                 file_name="unspecified_filename", vector_path=None, layers_to_cut_at=None, autoencoder_sizes=None,
                 autoencoder_loss='mse', autoencoder_optimizer="sgd", optimizer="adagrad", class_names=None,
                  is_autoencoder=False, denoising=True, output_reconstruction = False):

        hidden_amount = len(hidden_size)
        if autoencoder_sizes is None:
            autoencoder_amt = 0
        else:
            autoencoder_amt = len(autoencoder_sizes)

        movie_names, movie_vectors, movie_labels = MovieTasks.getMovieData(class_type=class_type, class_names=class_names,
                                                                           input_size=input_size, vector_path=vector_path)
        print len(movie_vectors), len(movie_vectors[0])
        print len(movie_labels), len(movie_labels[0])
        if autoencoder_amt > 0 or is_autoencoder is True:
            movie_labels = movie_vectors
            movie_labels = np.asarray(movie_labels)
            if denoising is True:
                movie_vectors = MovieTasks.makeSpaceNoisy(dt.getMovieVectors(vector_path=vector_path), 0.25)
                dt.write2dArray(movie_vectors, "newdata/spaces/"+file_name+".mds")
        else:
            n_train, x_train, y_train, n_test, x_test, y_test = dt.splitData(training_data, movie_names, movie_vectors,
                                                                             movie_labels)
            n_train = np.asarray(n_train)
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            n_test = np.array(n_test)
            x_test = np.asarray(x_test)
            y_test = np.asarray(y_test)

        self.model = Sequential()
        if autoencoder_amt == 0 and is_autoencoder is False:
            self.createModel(self.model, hidden_amount, dropout_chance, hidden_activation, layer_init, hidden_size,
                         input_size, output_size, output_activation)
            self.predicted_classes = self.trainNetwork(self.model, x_train, y_train, x_test, y_test, epochs, batch_size,
                                                   learn_rate, decay, loss, class_mode, optimizer, is_autoencoder, autoencoder_amt, movie_vectors, movie_labels)
            self.model_architecture = self.model.to_json()
            self.f1_score = f1_score(y_test, self.predicted_classes, average='macro')
            self.objective_score = self.model.evaluate(x_test, y_test, batch_size=batch_size, show_accuracy=True, verbose=1)
            movie_vectors = np.asarray(movie_vectors)
        else:
            movie_vectors = np.asarray(movie_vectors)
            if is_autoencoder:
                encoder = self.createModel(self.model, hidden_amount, dropout_chance, hidden_activation, layer_init, hidden_size,
                         input_size, output_size, output_activation)
            else:
                encoder = self.createAutoEncoder(self.model, hidden_amount, dropout_chance, hidden_activation, layer_init, hidden_size,
                    input_size, output_size, output_activation, autoencoder_sizes, autoencoder_amt, output_reconstruction)

            self.predicted_classes = self.trainNetwork(self.model, movie_vectors, movie_labels, movie_labels, movie_labels, epochs, batch_size,
                                                   learn_rate, decay, autoencoder_loss, class_mode, autoencoder_optimizer,is_autoencoder,  autoencoder_amt, movie_vectors, movie_labels)

        print self.f1_score, self.correct_classes, self.objective_score

        if save_architecture == True:
            with open("newdata/networks/" + file_name + ".json", 'w') as outfile:
                json.dump(self.model_architecture, outfile)

        if save_weights == True:
            self.model.save_weights("newdata/weights/" + file_name + ".h5", overwrite=True)

        if save_space is True:
            if autoencoder_amt > 0:
                total_file_name = "newdata/spaces/AUTOENCODER" + file_name +".mds"
                end_space = self.predicted_classes
            else:
                for l in layers_to_cut_at:
                    if is_autoencoder is True:
                        total_file_name = "newdata/spaces/AUTOENCODER" + file_name +".mds"
                    else:
                        total_file_name = "newdata/spaces/" + file_name + str(l) + ".mds"
                    end_space = self.saveSpace(self.model, l, movie_vectors, total_file_name, hidden_size, input_size,
                                               layer_init,  hidden_amount, hidden_activation, output_size,
                                               output_activation, dropout_chance, learn_rate, decay, loss, class_mode)
            dt.write2dArray(end_space, total_file_name)

    def createAutoEncoder(self, model,  hidden_amount, dropout_chance, hidden_activation, layer_init, hidden_size,
                    input_size, output_size, output_activation, autoencoder_sizes, autoencoder_amt, output_reconstruction):
        if autoencoder_amt >= 2:
            for x in range(0,autoencoder_amt,2):
                encoder = containers.Sequential([Dense(output_dim=autoencoder_sizes[x], input_dim=input_size), Dense(autoencoder_sizes[x+1])])
                decoder = containers.Sequential([Dense(output_dim=autoencoder_sizes[x], input_dim=autoencoder_sizes[x+1]), Dense(input_size)])
                model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=output_reconstruction))
        elif autoencoder_amt == 1:
            encoder = Dense(output_dim=autoencoder_sizes[0], input_dim=input_size, activation=hidden_activation)
            decoder = Dense(output_dim=input_size, input_dim=autoencoder_sizes[0], activation=hidden_activation)
            model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=output_reconstruction))
        return encoder


    def createModel(self, model,  hidden_amount, dropout_chance, hidden_activation, layer_init, hidden_size,
                    input_size, output_size, output_activation):

        if hidden_amount > 0:
            print 1
            model.add(Dense(output_dim=hidden_size[0],  input_dim=input_size, init=layer_init))
        else:

            model.add(Dense(output_dim=output_size, input_dim=input_size, init=layer_init))

        for x in range(0, hidden_amount):
            if x >= 1:
                model.add(Dense(activation=hidden_activation, output_dim=hidden_size[x]))
            else:
                print 2
                model.add(Dense(activation=hidden_activation, output_dim=output_size))
        print 3
        model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))

    def trainNetwork(self, model, x_train, y_train, x_test, y_test, epochs, batch_size, learn_rate, decay, loss, class_mode, optimizer, is_autoencoder, autoencoder_amt, movie_vectors, movie_labels):
        if optimizer is "adagrad":
            optimizer = Adagrad(lr=learn_rate, epsilon=decay)
        if autoencoder_amt > 0 or is_autoencoder is True:
            print 4
            model.compile(loss=loss, optimizer=optimizer)
        else:
            model.compile(loss=loss, optimizer=optimizer, class_mode=class_mode)
        if autoencoder_amt > 0:
            print movie_vectors[0], movie_labels[0]
            model.fit(movie_vectors, movie_labels, nb_epoch=epochs, batch_size=batch_size, verbose=1)
            model.output_reconstruction = False  # the autoencoder has to be recompiled after modifying this property
            model.compile(loss=loss, optimizer=optimizer)
            predicted_classes = model.predict(movie_labels)
        else:
            print 5
            self.history = model.fit(x_train, y_train, nb_epoch=epochs, batch_size=batch_size, verbose=1)
            predicted_classes = model.predict_classes(x_test, batch_size=batch_size)
        return predicted_classes

    def saveSpace(self, model, layer_to_cut_at, movie_vectors, file_name, hidden_size, input_size,
                  layer_init, hidden_amount, hidden_activation, output_size, output_activation, dropout_chance,
                  learn_rate, decay, loss, class_mode):
        truncated_model = Sequential()

        if hidden_amount > 0:
            truncated_model.add(Dense(output_dim=hidden_size[0], input_dim=input_size, init=layer_init, weights=model.layers[0].get_weights()))
        else:
            truncated_model.add(Dense(output_dim=output_size, input_dim=input_size, init=layer_init, weights=model.layers[0].get_weights()))

        if layer_to_cut_at == 0:
            truncated_model.compile(loss=loss, optimizer='sgd', class_mode=class_mode)
            return truncated_model.predict(movie_vectors)
        else:
            for x in range(hidden_amount):
                if x >= 1:
                    truncated_model.add(Dense(activation=hidden_activation, output_dim=hidden_size[x], weights=model.layers[x+1].get_weights()))
                else:
                    truncated_model.add(Dense(activation=hidden_activation, output_dim=output_size, weights=model.layers[x+1].get_weights()))
                if layer_to_cut_at == x+1:
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
    output_size = 200
    loss = "mse"
    trainer="sgd"
    input_size=200
    epochs=250
    hidden_activation="tanh"
    output_activation="tanh"
    layers_to_cut_at=[0]
    save_space=True
    denoising = True
    is_autoencoder = True
    autoencoder_sizes=[150]
    hidden_size = []
    #vector_path="newdata/spaces/" + "AUTOENCODER" + str(epochs) + hidden_activation + ".mds",
    for x in range(0, 10):
        if x == 0:
            dimension_100 = NeuralNetwork(input_size=input_size, epochs=epochs, autoencoder_sizes=autoencoder_sizes,
                                          output_size=output_size, class_type=class_type,
                                        save_space=save_space, layers_to_cut_at=layers_to_cut_at,  autoencoder_loss = loss,
                                          autoencoder_optimizer=trainer, hidden_activation=hidden_activation, output_activation=output_activation,
                                          file_name="" +str(x) + str(epochs) + hidden_activation + str(autoencoder_sizes[0]))
        else:
            new_as = [int(autoencoder_sizes[0] - (autoencoder_sizes[0] / 3))]
            dimension_100 = NeuralNetwork(input_size=autoencoder_sizes[0], epochs=epochs, autoencoder_sizes=new_as,
                                          output_size=autoencoder_sizes[0], class_type=class_type, vector_path="newdata/spaces/AUTOENCODER"+ str(x-1) + str(epochs) + hidden_activation + str(autoencoder_sizes[0])+".mds",
                                        save_space=save_space, layers_to_cut_at=layers_to_cut_at,  autoencoder_loss = loss,
                                          autoencoder_optimizer=trainer, hidden_activation=hidden_activation, output_activation=output_activation,
                                          file_name="" + str(x) + str(epochs) + hidden_activation + str(new_as[0]))
            autoencoder_sizes[0] = new_as[0]

    """
                                  file_name="A"+hidden_activation+"L"+loss+"T"+ trainer +"IS"+str(input_size)+"HS"+ str(hidden_size)+ "OS" + str(output_size) +
                                  "OA"+output_activation+class_type+str(autoencoder_sizes))
    """
    """

    """






if  __name__ =='__main__':main()

