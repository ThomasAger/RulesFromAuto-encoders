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
                 file_name="unspecified_filename", vector_path=None, layers_to_cut_at=None, autoencoder_sizes=None, autoencoders_amt=0):


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
                         input_size, output_size, output_activation, autoencoders_amt, autoencoder_sizes)
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
        if save_space == True:
            for l in layers_to_cut_at:
                if autoencoders_amt > 1:
                    total_file_name = "newdata/spaces/AUTOENCODER" + file_name + str(l) + ".mds"
                else:
                    total_file_name = "newdata/spaces/" + file_name + str(l) + ".mds"
                self.saveSpace(self.model, l, movie_vectors, total_file_name)

    def createModel(self, model,  hidden_amount, dropout_chance, hidden_activation, layer_init, hidden_size,
                    input_size, output_size, output_activation, autoencoders_amt, autoencoder_sizes):
        if autoencoders_amt >= 1:
            model.add(Dense(output_dim=hidden_size[0], input_dim=input_size, init=layer_init))
            sizes_follower = 0
            for x in range(autoencoders_amt):
                encoder = containers.Sequential(layers=[Dense(output_dim=autoencoder_sizes[sizes_follower + 0],
                                                              input_dim=input_size), Dense(output_dim=autoencoder_sizes[sizes_follower + 1],
                                                                                           input_dim=autoencoder_sizes[sizes_follower + 0])])
                decoder = containers.Sequential(layers=[Dense(output_dim=autoencoder_sizes[sizes_follower + 2],
                                                              input_dim=autoencoder_sizes[sizes_follower + 1]),
                                                        Dense(output_dim=input_size, input_dim=autoencoder_sizes[sizes_follower + 2])])
                model.add(AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True))
                sizes_follower += 3
        else:
            model.add(Dense(output_dim=hidden_size[0], input_dim=input_size, init=layer_init))
        if hidden_amount >= 1:

            for x in range(hidden_amount):
                model.add(Activation(hidden_activation))
                model.add(Dropout(dropout_chance))
                print "added?"

            model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))

    def trainNetwork(self, model, x_train, y_train, x_test, y_test, epochs, batch_size, learn_rate, decay, loss, class_mode):
        trainer = Adagrad(lr=learn_rate, epsilon=decay)
        model.compile(loss=loss, optimizer=trainer, class_mode=class_mode)
        self.history = model.fit(x_train, y_train, nb_epoch=epochs, batch_size=batch_size, verbose=1)
        self.objective_score = model.evaluate(x_test, y_test, batch_size=batch_size, show_accuracy=True, verbose=1)
        predicted_classes = model.predict_classes(x_test, batch_size=batch_size)
        return predicted_classes


    def saveSpace(self, model, layer_to_cut_at, movie_vectors, file_name):
        print "savespace"
        print "model layer 0 input", [model.layers[layer_to_cut_at].input], "model layer 1 get output", model.layers[layer_to_cut_at+1].get_output(train=False)
        # When using a Dense as the hidden layer, this fails as it apparently has no attribute 'Output'
        hidden_layer = theano.function([model.layers[layer_to_cut_at].input], model.layers[layer_to_cut_at+1].get_output(train=False), allow_input_downcast=True)
        print hidden_layer
        transformed_space = hidden_layer(np.asarray(movie_vectors, dtype=np.float32))
        print transformed_space
        dt.write2dArray(transformed_space, file_name)



def main():


    print "Genres"
    class_type = "Genres"
    output_size = 25
    loss = "bias_binary_crossentropy"

    dimension_100 = NeuralNetwork(save_weights=True, input_size=200, epochs=250,
                                 hidden_size=[200], hidden_amount=1, output_size=output_size, class_type=class_type,
                                loss=loss, save_space=True, save_architecture=True, file_name="NewKeywords-200-200",
                                layers_to_cut_at=[4,5], autoencoder_sizes=[20,200,20], autoencoders_amt=1)

    """
    print "NewKeywords"
    class_type = "NewKeywords"
    output_size = 3000
    loss = "sparse_bias_binary_crossentropy"

    dimension_100 = NeuralNetwork(save_weights=True, input_size=200, epochs=250,
                                 hidden_size=[200], hidden_amount=1, output_size=output_size, class_type=class_type,
                                loss=loss, save_space=True, save_architecture=True, file_name="NewKeywords-200-200")

    dimension_100 = NeuralNetwork(save_weights=True, input_size=200, epochs=250,
                                 hidden_size=[20], hidden_amount=1, output_size=output_size, class_type=class_type,
                                loss=loss, save_space=True, save_architecture=True, file_name="NewKeywords-200-20")

    dimension_100 = NeuralNetwork(save_weights=True, input_size=200, epochs=250,
                                 hidden_size=[50], hidden_amount=1, output_size=output_size, class_type=class_type,
                                loss=loss, save_space=True, save_architecture=True, file_name="NewKeywords-200-50")

    dimension_100 = NeuralNetwork(save_weights=True, input_size=200, epochs=250,
                                 hidden_size=[100], hidden_amount=1, output_size=output_size, class_type=class_type,
                                loss=loss, save_space=True, save_architecture=True, file_name="NewKeywords-200-100")

    dimension_100 = NeuralNetwork(save_weights=True, input_size=200, epochs=250,
                                 hidden_size=[400], hidden_amount=1, output_size=output_size, class_type=class_type,
                                loss=loss, save_space=True, save_architecture=True, file_name="NewKeywords-200-400")

    dimension_100 = NeuralNetwork(save_weights=True, input_size=200, epochs=250,
                                 hidden_size=[600], hidden_amount=1, output_size=output_size, class_type=class_type,
                                loss=loss, save_space=True, save_architecture=True, file_name="NewKeywords-200-600")
    """



if  __name__ =='__main__':main()

