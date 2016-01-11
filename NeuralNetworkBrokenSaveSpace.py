# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np

import DataTasks as dt
import MovieTasks
from keras.layers.core import Dense, Activation, Dropout
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
                 hidden_activation="tanh", layer_init="glorot_uniform", output_activation="sigmoid", dropout_chance=0.5, hidden_layer_space=1,
                 class_mode="binary", save_weights=False, save_space=False, save_architecture=False, file_name="unspecified_filename", vector_path=None):


        movie_names, movie_vectors, movie_labels = MovieTasks.getMovieData(class_type=class_type, class_names=None, input_size=input_size, vector_path=vector_path)
        n_train, x_train, y_train, n_test, x_test, y_test = dt.splitData(training_data, movie_names, movie_vectors, movie_labels)
        n_train = np.asarray(n_train)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        n_test = np.array(n_test)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        self.model = Sequential()
        self.addLayers(self.model, hidden_amount, dropout_chance, hidden_activation, layer_init, hidden_size, input_size, output_size, output_activation)
        self.predicted_classes = self.trainNetwork(self.model, x_train, y_train, x_test, y_test, epochs, batch_size, learn_rate, decay, loss, class_mode)

        self.model_architecture = self.model.to_json()
        self.f1_score = f1_score(y_test, self.predicted_classes, average='macro')
        self.correct_classes = dt.printClasses(y_test, self.predicted_classes, n_test)


        if save_architecture == True:
            with open("newdata/networks/" + file_name + ".json", 'w') as outfile:
                json.dump(self.model_architecture, outfile)
        if save_weights == True:
            self.model.save_weights("newdata/weights/" + file_name + ".h5", overwrite=True)
        if save_space == True:
            if hidden_amount > 1:
                for i in range(hidden_amount - 1):
                    self.saveSpace(self.model, movie_vectors, i, "newdata/spaces/" + file_name + str(hidden_amount) +".mds")
            else:
                self.saveSpace(self.model, movie_vectors, hidden_amount, "newdata/spaces/" + file_name + ".mds")

    def addLayers(self, model,  hidden_amount, dropout_chance, hidden_activation, layer_init, hidden_size, input_size, output_size, output_activation):

        model.add(Dense(output_dim=hidden_size[0], input_dim=input_size, init=layer_init))

        for x in range(hidden_amount):
            model.add(Activation(hidden_activation))
            model.add(Dropout(dropout_chance))

        model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))


    def trainNetwork(self, model, x_train, y_train, x_test, y_test, epochs, batch_size, learn_rate, decay, loss, class_mode):
        trainer = Adagrad(lr=learn_rate, epsilon=decay)
        model.compile(loss=loss, optimizer=trainer, class_mode=class_mode)
        self.history = model.fit(x_train, y_train, nb_epoch=epochs, batch_size=batch_size, verbose=0)
        self.objective_score = model.evaluate(x_test, y_test, batch_size=batch_size, show_accuracy=True, verbose=0)
        predicted_classes = model.predict_classes(x_test, batch_size=batch_size)
        return predicted_classes


    def saveSpace(self, model, movie_vectors, layer, file_name):


        # When using a Dense as the hidden layer, this fails as it apparently has no attribute 'Output'
        hidden_layer = theano.function([model.layers[0].input], model.layers[1].get_output(train=False), allow_input_downcast=True)
        print hidden_layer
        transformed_space = hidden_layer(np.asarray(movie_vectors, dtype=np.float32))

        dt.write2dArray(transformed_space, file_name)



def main():

    print "ALL"
    class_type = "All"
    output_size = 125
    loss = "sparse_bias_binary_crossentropy"

    dimension_100 = NeuralNetwork(save_weights=True, epochs=250,
                                 hidden_size=[300], hidden_amount=1, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="All-Layer-300-L")

    print "GENRES"
    class_type = "Genres"
    output_size = 25
    loss = "bias_binary_crossentropy"

    dimension_100 = NeuralNetwork(save_weights=True, epochs=250,
                                 hidden_size=[300], hidden_amount=1, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="Genres-Layer-300-L")

    print "KEYWORDS"
    class_type = "Keywords"
    output_size = 100
    loss = "sparse_bias_binary_crossentropy"

    dimension_100 = NeuralNetwork(save_weights=True, epochs=250,
                                 hidden_size=[300], hidden_amount=1, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="Keywords-Layer-300-L")


if  __name__ =='__main__':main()

