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
                 hidden_layer_space=1, class_mode="binary", save_weights=False, save_space=False, save_architecture=False,
                 file_name="unspecified_filename", vector_path=None, layers_to_cut_at=None, autoencoder_sizes=None):

        hidden_amount = len(hidden_size)
        autoencoder_amt = len(autoencoder_sizes)
        movie_names, movie_vectors, movie_labels = MovieTasks.getMovieData(class_type=class_type, class_names=None,
                                                                           input_size=input_size, vector_path=vector_path)

        n_train, x_train, y_train, n_test, x_test, y_test = dt.splitData(training_data, movie_names, movie_vectors,
                                                                         movie_labels)
        n_train = np.asarray(n_train)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        n_test = np.array(n_test)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        self.model = Sequential()
        self.createModel(self.model, hidden_amount, dropout_chance, hidden_activation, layer_init, hidden_size,
                         input_size, output_size, output_activation, autoencoder_sizes, autoencoder_amt)
        self.predicted_classes = self.trainNetwork(self.model, x_train, y_train, x_test, y_test, epochs, batch_size,
                                                   learn_rate, decay, loss, class_mode)

        self.model_architecture = self.model.to_json()
        self.f1_score = f1_score(y_test, self.predicted_classes, average='macro')
        self.correct_classes = dt.printClasses(y_test, self.predicted_classes, n_test)
        print self.f1_score, self.correct_classes, self.objective_score

        if save_architecture == True:
            with open("newdata/networks/" + file_name + ".json", 'w') as outfile:
                json.dump(self.model_architecture, outfile)
        if save_weights == True:
            self.model.save_weights("newdata/weights/" + file_name + ".h5", overwrite=True)

        if save_space is True:
            for l in layers_to_cut_at:
                if autoencoder_amt > 1:
                    total_file_name = "newdata/spaces/AUTOENCODER" + file_name + str(l) + ".mds"
                else:
                    total_file_name = "newdata/spaces/" + file_name + str(l) + ".mds"

                movie_vectors = np.asarray(movie_vectors)
                new_space = self.saveSpace(self.model, l, movie_vectors, total_file_name, hidden_size, input_size, layer_init,
                               hidden_amount, hidden_activation, output_size, output_activation, dropout_chance,
                                learn_rate, decay, loss, class_mode, autoencoder_amt, autoencoder_sizes)
                dt.write2dArray(new_space, total_file_name)


    def createModel(self, model,  hidden_amount, dropout_chance, hidden_activation, layer_init, hidden_size,
                    input_size, output_size, output_activation, autoencoder_sizes, autoencoder_amt):
        if autoencoder_amt > 0:
            encoder = containers.Sequential([Dense(autoencoder_sizes[0], input_dim=input_size), Dense(autoencoder_sizes[0])])
            decoder = containers.Sequential([Dense(16, input_dim=8), Dense(32)])

            model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))
        if hidden_amount > 0:
            model.add(Dense(output_dim=hidden_size[0], input_dim=input_size, init=layer_init))
        else:
            model.add(Dense(output_dim=output_size, input_dim=input_size, init=layer_init))
        for x in range(hidden_amount):
            print x, 1
            if x != hidden_amount:
                model.add(Dense(activation=hidden_activation, output_dim=hidden_size[x]))
            else:
                model.add(Dense(activation=hidden_activation, output_dim=output_size))
            # Dropout resulted in a performance drop, so I've removed it.
            #model.add(Dropout(dropout_chance))

        model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))

    def trainNetwork(self, model, x_train, y_train, x_test, y_test, epochs, batch_size, learn_rate, decay, loss, class_mode):
        trainer = Adagrad(lr=learn_rate, epsilon=decay)
        model.compile(loss=loss, optimizer=trainer, class_mode=class_mode)
        self.history = model.fit(x_train, y_train, nb_epoch=epochs, batch_size=batch_size, verbose=1)
        self.objective_score = model.evaluate(x_test, y_test, batch_size=batch_size, show_accuracy=True, verbose=1)
        predicted = model.predict(x_train)
        predicted_classes = model.predict_classes(x_test, batch_size=batch_size)
        return predicted_classes

    def saveSpace(self, model, layer_to_cut_at, movie_vectors, file_name, hidden_size, input_size,
                  layer_init, hidden_amount, hidden_activation, output_size, output_activation, dropout_chance,
                  learn_rate, decay, loss, class_mode, autoencoder_sizes, autoencoder_amt):
        # we build a new model with the activations of the old model
        # this model is truncated after the first layer
        truncated_model = Sequential()
        if hidden_amount > 0:
            truncated_model.add(Dense(output_dim=hidden_size[0], input_dim=input_size, init=layer_init, weights=model.layers[0].get_weights()))
        else:
            truncated_model.add(Dense(output_dim=output_size, input_dim=input_size, init=layer_init, weights=model.layers[0].get_weights()))

        if layer_to_cut_at == 0:
            trainer = Adagrad(lr=learn_rate, epsilon=decay)
            truncated_model.compile(loss=loss, optimizer=trainer, class_mode=class_mode)
            return truncated_model.predict(movie_vectors)
        else:
            for x in range(hidden_amount):
                if x != hidden_amount:
                    truncated_model.add(Dense(activation=hidden_activation, output_dim=hidden_size[x], weights=model.layers[x+1].get_weights()))
                else:
                    truncated_model.add(Dense(activation=hidden_activation, output_dim=output_size, weights=model.layers[x+1].get_weights()))
                #truncated_model.add(Dropout(dropout_chance))
                if layer_to_cut_at == x+1:
                    trainer = Adagrad(lr=learn_rate, epsilon=decay)
                    truncated_model.compile(loss=loss, optimizer=trainer, class_mode=class_mode)
                    print "was x"
                    return truncated_model.predict(movie_vectors)
        print "not x"
        truncated_model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))
        trainer = Adagrad(lr=learn_rate, epsilon=decay)
        truncated_model.compile(loss=loss, optimizer=trainer, class_mode=class_mode)
        return truncated_model.predict(movie_vectors)




def main():
    """

    print "All"
    class_type = "Keywords"
    output_size = 100
    loss = "sparse_bias_binary_crossentropy"

    dimension_100 = NeuralNetwork(save_weights=True, input_size=200, epochs=250,
                                 hidden_size=[200], hidden_amount=1, output_size=output_size, class_type=class_type,
                                loss=loss, save_space=True, save_architecture=True, file_name="Keywords-1Autoencoder400,20,400-200-200",
                                layers_to_cut_at=[0], autoencoder_sizes=[400,20,400], autoencoders_amt=1)

    """
    """
    print "All"
    class_type = "All"
    output_size = 125
    loss = "sparse_bias_binary_crossentropy"

    dimension_100 = NeuralNetwork(save_weights=True, input_size=200, epochs=250, training_data=15000,
                                 hidden_size=[200], hidden_amount=1, output_size=output_size, class_type=class_type,
                                loss=loss, save_space=True, save_architecture=True, file_name="Keywords-1Autoencoder200,20,200-200-200",
                                layers_to_cut_at=[0], autoencoder_sizes=[200,20,200], autoencoders_amt=1)
    """
    print "All"
    class_type = "All"
    output_size = 125
    loss = "sparse_bias_binary_crossentropy"

    dimension_100 = NeuralNetwork(input_size=200, epochs=250,
                                 hidden_size=[100,20,200],  output_size=output_size, class_type=class_type,
                                loss=loss, save_space=True, layers_to_cut_at=[3],
                                  file_name="All-100,20,200x")




if  __name__ =='__main__':main()

