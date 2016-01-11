# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import importdata
import outputdata
import numpy as np

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



    def __init__(self, input_size=200, hidden_size=400, output_size=100, training_data=10000, class_type="Genres",
                 epochs=250,  learn_rate=0.01, loss="binary_crossentropy", hidden_amount=1, batch_size=100, decay=1e-06,
                 hidden_activation="tanh", layer_init="glorot_uniform", output_activation="sigmoid", dropout_chance=0.5, hidden_layer_space=1,
                 class_mode="binary", save_weights=False, save_space=False, save_architecture=False, file_name="unspecified_filename", vector_path=None):


        movie_names, movie_vectors, movie_labels = self.importData(class_type, input_size, vector_path)
        n_train, x_train, y_train, n_test, x_test, y_test = self.splitData(training_data, movie_names, movie_vectors, movie_labels)

        self.model = Sequential()
        self.addLayers(self.model, hidden_amount, dropout_chance, hidden_activation, layer_init, hidden_size, input_size, output_size, output_activation)
        self.predicted_classes = self.trainNetwork(self.model, x_train, y_train, x_test, y_test, epochs, batch_size, learn_rate, decay, loss, class_mode)

        self.model_architecture = self.model.to_json()
        self.f1_score = f1_score(y_test, self.predicted_classes, average='macro')
        self.correct_classes = self.printClasses(y_test, self.predicted_classes, n_test)


        if save_architecture == True:
            with open("newdata/networks/" + file_name + ".json", 'w') as outfile:
                json.dump(self.model_architecture, outfile)
        if save_weights == True:
            self.model.save_weights("newdata/weights/" + file_name + ".h5", overwrite=True)
        if save_space == True:
            self.saveSpace(self.model, movie_vectors, hidden_layer_space, "newdata/spaces/" + file_name + ".mds")


    def importData(self, class_type="All", input_size=200, vector_path=None):
        movie_names = importdata.importString("filmdata/filmNames.txt")

        if class_type == "All":
            movie_labels = importdata.getAllLabels("Keywords")
            genre_labels = importdata.getAllLabels("Genres")
            for c in range(len(movie_labels)):
                movie_labels[c].extend(genre_labels[c])
        else: movie_labels = importdata.getAllLabels(class_type)

        if vector_path is None:
            movie_vectors = importdata.getMovieVectors(input_size)
        else:
            movie_vectors = importdata.getMovieVectors(input_size, vector_path)


        return movie_names, movie_vectors, movie_labels


    def splitData(self, training_data, movie_names, movie_vectors, movie_labels):
        n_train = movie_names[:training_data]
        x_train = np.asarray(movie_vectors[:training_data])
        y_train = np.asarray(movie_labels[:training_data])

        n_test = movie_names[training_data:]
        x_test = np.asarray(movie_vectors[training_data:])
        y_test = np.asarray(movie_labels[training_data:])
        return n_train, x_train, y_train, n_test, x_test, y_test

    def addLayers(self, model,  hidden_amount, dropout_chance, hidden_activation, layer_init, hidden_size, input_size, output_size, output_activation):

        model.add(Dense(output_dim=hidden_size, input_dim=input_size, init=layer_init))

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
        hidden_layer = theano.function([model.layers[0].input], model.layers[layer].get_output(train=False))
        transformed_space = hidden_layer(np.asarray(movie_vectors, dtype=np.float32))

        outputdata.write2dArray(transformed_space, file_name)

    # refactor this
    def printClasses(self, Y_test, Y_predicted, n_test):
        misclassified1 = 0
        classified1 = 0
        misclassified0 = 0
        classified0 = 0
        correctCount = 0
        amt_correct = 0
        films_classified_correctly = []
        for sets in range(len(Y_predicted)):
            correctCount = 0
            for c in range(len(Y_predicted[sets])):
                if Y_test[sets][c] == 1 and Y_predicted[sets][c] == 0:
                    misclassified1 = misclassified1 + 1
                elif Y_test[sets][c] == 1 and Y_predicted[sets][c] == 1:
                    classified1 = classified1 + 1
                    correctCount = correctCount + 1
                elif Y_test[sets][c] == 0 and Y_predicted[sets][c] == 1:
                    misclassified0 = misclassified0 + 1
                else:
                    classified0 = classified0 + 1
                    correctCount = correctCount + 1
            if correctCount >= len(Y_test[0]):
                amt_correct = amt_correct + 1

        return amt_correct

def main():

    print "#### layer sizes ####"
    """
    class_type = "All"
    output_size = 125
    loss = "sparse_bias_binary_crossentropy"

    dimension_200 = NeuralNetwork(save_weights=True, epochs=250,
                                 hidden_size=400, hidden_amount=2, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="All-Layer-standard" + str(2))
    dimension_100 = NeuralNetwork(save_weights=True, epochs=250,
                                 hidden_size=400, hidden_amount=4, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="All-Layer-standard" + str(4))
    dimension_50 = NeuralNetwork( save_weights=True, epochs=250,
                                 hidden_size=400, hidden_amount=8, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="All-Layer-standard" + str(8))
    dimension_20 = NeuralNetwork( save_weights=True, epochs=250,
                                 hidden_size=400, hidden_amount=16, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="All-Layer-standard" + str(16))

    print "#### ALL ####"

    print "standard:", dimension_200.objective_score, dimension_200.f1_score, dimension_200.correct_classes
    print "Neural Network 0.02:", dimension_100.objective_score, dimension_100.f1_score, dimension_100.correct_classes
    print "Neural Network 0.04:", dimension_50.objective_score, dimension_50.f1_score, dimension_50.correct_classes
    print "Neural Network 0.06:", dimension_20.objective_score, dimension_20.f1_score, dimension_20.correct_classes

    ifile = open('experiments/' + "All-Layer-standard" + '.csv', "w")
    writer = csv.writer(ifile, delimiter=b" ")
    writer.writerow("Hidden Size, Objective Score, Accuracy, F1, Correct Classes")
    writer.writerow("2,", str(dimension_200.objective_score[0])+","+str(dimension_200.objective_score[1])+","+str(dimension_200.f1_score)+","+str(dimension_200.correct_classes)+"")
    writer.writerow("4,"+str(dimension_100.objective_score[0])+","+str(dimension_100.objective_score[1])+","+str(dimension_100.f1_score)+","+str(dimension_100.correct_classes)+"")
    writer.writerow("8,"+str(dimension_50.objective_score[0])+","+str(dimension_50.objective_score[1])+","+str(dimension_50.f1_score)+","+str(dimension_50.correct_classes)+"")
    writer.writerow("16,"+str(dimension_20.objective_score[0])+","+str(dimension_20.objective_score[1])+","+str(dimension_20.f1_score)+","+str(dimension_20.correct_classes)+"")
    ifile.close()
    """
    class_type = "Genres"
    output_size = 25
    loss = "bias_binary_crossentropy"

    dimension_200 = NeuralNetwork(save_weights=True, epochs=250,
                                 hidden_size=400, hidden_amount=2, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="Genre-Layer-standard" + str(2))
    dimension_100 = NeuralNetwork(save_weights=True, epochs=250,
                                 hidden_size=400, hidden_amount=4, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="Genre-Layer-standard" + str(4))
    dimension_50 = NeuralNetwork( save_weights=True, epochs=250,
                                 hidden_size=400, hidden_amount=8, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="Genre-Layer-standard" + str(8))
    dimension_20 = NeuralNetwork( save_weights=True, epochs=250,
                                 hidden_size=400, hidden_amount=10, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="Genre-Layer-standard" + str(10))

    print "#### GENRES ####"
    print "Neural Network 0.005:", dimension_200.objective_score, dimension_200.f1_score, dimension_200.correct_classes
    print "Neural Network 0.02:", dimension_100.objective_score, dimension_100.f1_score, dimension_100.correct_classes
    print "Neural Network 0.04:", dimension_50.objective_score, dimension_50.f1_score, dimension_50.correct_classes
    print "Neural Network 0.06:", dimension_20.objective_score, dimension_20.f1_score, dimension_20.correct_classes


    columns = ["Hidden Size", "Objective Score", "Accuracy", "F1", "Correct Classes"]
    data = ["2", dimension_200.objective_score[0], dimension_200.objective_score[1], dimension_200.f1_score, dimension_200.correct_classes]
    df = pandas.DataFrame(data, columns=columns)
    df.tocsv("experiments/Genres-Layer-standard.csv")

    class_type = "Keywords"
    output_size = 100
    loss = "sparse_bias_binary_crossentropy"

    dimension_200 = NeuralNetwork(save_weights=True, epochs=250,
                                 hidden_size=400, hidden_amount=2, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="Keyword-Layer-standard" + str(2))
    dimension_100 = NeuralNetwork(save_weights=True, epochs=250,
                                 hidden_size=400, hidden_amount=4, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="Keyword-Layer-standard" + str(4))
    dimension_50 = NeuralNetwork( save_weights=True, epochs=250,
                                 hidden_size=400, hidden_amount=8, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="Keyword-Layer-standard" + str(8))
    dimension_20 = NeuralNetwork( save_weights=True, epochs=250,
                                 hidden_size=400, hidden_amount=10, output_size=output_size, class_type=class_type, loss=loss, save_space=True, save_architecture=True, file_name="Keyword-Layer-standard" + str(10))

    print "#### KEYWORDS ####"
    print "Neural Network 0.005:", dimension_200.objective_score, dimension_200.f1_score, dimension_200.correct_classes
    print "Neural Network 0.02:", dimension_100.objective_score, dimension_100.f1_score, dimension_100.correct_classes
    print "Neural Network 0.04:", dimension_50.objective_score, dimension_50.f1_score, dimension_50.correct_classes
    print "Neural Network 0.06:", dimension_20.objective_score, dimension_20.f1_score, dimension_20.correct_classes


    columns = ["Hidden Size", "Objective Score", "Accuracy", "F1", "Correct Classes"]
    data = ["2", dimension_200.objective_score[0], dimension_200.objective_score[1], dimension_200.f1_score, dimension_200.correct_classes]
    df = pandas.DataFrame(data, columns=columns)
    df.tocsv("experiments/Keywords-Layer-standard.csv")



if  __name__ =='__main__':main()

