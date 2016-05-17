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


    def __init__(self,  hidden_layer_sizes=[400], output_size=100, training_data=10000, class_path="",
                 epochs=250,  learn_rate=0.05, loss="binary_crossentropy", batch_size=100, decay=1e-06,
                 hidden_activation="relu", layer_init="glorot_uniform", output_activation="softmax", dropout_chance=0.5,
                  class_mode="binary",save_space=False, autoencoder_space = None, encoders=None, hidden_layer_size=100,
                 file_name="unspecified_filename", vector_path=None, layers_to_cut_at=None, reg=0,
                 optimizer="adagrad", class_names=None, is_autoencoder=False, noise=0,  numpy_vector_path=None):

        model = Sequential()

        if autoencoder_space is not None:
            movie_vectors = np.asarray(autoencoder_space)
        else:
            if numpy_vector_path is not None:
                movie_vectors = dt.importNumpyVectors(numpy_vector_path)
            else:
                movie_vectors = np.asarray(dt.importVectors(vector_path))
            movie_labels = np.asarray(dt.importLabels(class_path))

        if is_autoencoder:
            movie_labels = movie_vectors

        input_size = len(movie_vectors[0])
        output_size = len(movie_labels[0])

        print input_size, output_size
        print movie_labels[0]

        x_train, y_train, x_test, y_test = dt.splitData(training_data, movie_vectors, movie_labels)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        if encoders is not None:  # If encoders have been passed in, create a model using encoders from before
            for e in encoders:
                model.add(e)
            for h in range(len(hidden_layer_sizes)):
                model.add(Dense(output_dim=hidden_layer_sizes[h], init=layer_init, activation=hidden_activation))
            model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))
        elif is_autoencoder:  # If creating an autoencoder, use autoencoder specific methods.
            if noise > 0: # If using a noisy autoencoder, add GaussianNoise layers to the start of the encoder
                if hidden_activation=="LeakyReLU":
                    self.encoder = containers.Sequential([
                        GaussianNoise(noise, input_shape=(input_size,)),
                        Dense(output_dim=hidden_layer_size,  input_dim=input_size, init=layer_init, activation="linear",W_regularizer=l1(reg)),
                        LeakyReLU(alpha=.3)],)
                else:
                    self.encoder = containers.Sequential([
                        GaussianNoise(noise, input_shape=(input_size,)),
                        Dense(output_dim=hidden_layer_size,  input_dim=input_size, init=layer_init, activation=hidden_activation,W_regularizer=l1(reg)),])
            else:
                self.encoder = Dense(output_dim=hidden_layer_size,  input_dim=input_size, init=layer_init, activation=hidden_activation,W_regularizer=l1(reg))
            decoder = Dense(output_dim=output_size, init=layer_init, activation=output_activation)
            model.add(AutoEncoder(encoder=self.encoder, decoder=decoder,
                           output_reconstruction=False))
        else:  # Just create a model using normal layers
            model.add(Dense(output_dim=output_size, input_dim=input_size, init=layer_init))
            for x in range(0, len(hidden_layer_sizes)):
                if x == len(hidden_layer_sizes)-1:
                    model.add(Dense(activation=hidden_activation, output_dim=output_size))
                else:
                    model.add(Dense(activation=hidden_activation, output_dim=hidden_layer_sizes[x]))
            model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))

        if optimizer is "adagrad":
            optimizer = Adagrad(lr=learn_rate, epsilon=decay)

        print x_train[0]
        print y_train[0]

        model.compile(loss=loss, optimizer=optimizer, class_mode=class_mode)
        history = model.fit(x_train, y_train, nb_epoch=epochs, batch_size=batch_size, show_accuracy=True, verbose=1)
        predicted_classes = model.predict_classes(x_test, batch_size=batch_size)

        if is_autoencoder is False:
            f1 = f1_score(y_test, predicted_classes, average='macro')
            objective_score = model.evaluate(x_test, y_test, batch_size=batch_size, show_accuracy=True, verbose=1)
            dt.write1dArray(["F1 " + str(f1),  ["Objective and accuracy ", objective_score]], "scores/"+file_name+"score.txt")
            print "F1 " + str(f1), ["Objective and accuracy ", objective_score]

        if save_space is True:
            if encoders is not None:
                print "Saving encoder layer spaces", layers_to_cut_at

                for l in range(len(model.layers)):
                    truncated_model = Sequential()
                    total_file_name = "newdata/spaces/AUTOENCODER" + file_name + str(l) +"FINETUNED.mds"
                    for e in range(len(encoders)):
                        weights = model.layers[e].get_weights()
                        truncated_model.add(encoders[e])
                        truncated_model.layers[e].set_weights(weights)
                        if l == e:
                            break
                    if l >= len(encoders):
                        for h in range(len(hidden_layer_sizes)):
                            weights = model.layers[len(encoders) + h].get_weights()
                            truncated_model.add(Dense(output_dim=hidden_layer_sizes[h], init=layer_init, activation=output_activation))
                            truncated_model.layers[len(encoders)+h].set_weights(weights)
                            if l == h + len(encoders):
                                break
                    if l >= len(encoders) + len(hidden_layer_sizes):
                        weights = model.layers[l].get_weights()
                        truncated_model.add(Dense(output_dim=output_size, init=layer_init, activation=output_activation))
                        truncated_model.layers[l].set_weights(weights)
                    truncated_model.compile(loss=loss, optimizer="sgd", class_mode=class_mode)
                    self.end_space = truncated_model.predict(movie_vectors)
                    dt.write2dArray(self.end_space, total_file_name)
            elif is_autoencoder:
                print "Saving autoencoder spaces", layers_to_cut_at
                truncated_model = Sequential()
                total_file_name = "newdata/spaces/AUTOENCODER" + file_name +".mds"
                truncated_model.add(self.encoder)
                truncated_model.compile(loss=loss, optimizer="sgd", class_mode=class_mode)
                self.end_space = truncated_model.predict(movie_vectors)
                dt.write2dArray(self.end_space, total_file_name)
            else:
                print "Saving feedforward spaces", layers_to_cut_at

                for l in layers_to_cut_at:
                    total_file_name = "newdata/spaces/" + file_name + str(l) + ".mds"
                    hidden_amount = len(hidden_layer_sizes)
                    truncated_model = Sequential()
                    if hidden_amount > 0:
                        truncated_model.add(Dense(output_dim=hidden_layer_sizes[0], input_dim=input_size, weights=model.layers[0].get_weights()))
                    else:
                        truncated_model.add(Dense(output_dim=output_size, input_dim=input_size, weights=model.layers[0].get_weights()))
                    if l == 0:
                        truncated_model.compile(loss=loss, optimizer='sgd', class_mode=class_mode)
                        break
                    else:
                        for x in range(1, hidden_amount):
                            truncated_model.add(Dense(activation=hidden_activation, output_dim=hidden_layer_sizes[x-1], weights=model.layers[x].get_weights()))
                            if l == x:
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
    hidden_layer_sizes = [100]
    dimension_100 = NeuralNetwork(input_size=input_size, epochs=epochs, hidden_layer_sizes=hidden_layer_sizes, denoising=denoising,
                                  output_size=output_size, class_type=class_type, vector_path="newdata/spaces/AUTOENCODERnoise020sigmoid.mds",
                                save_space=save_space, layers_to_cut_at=layers_to_cut_at, hidden_activation=hidden_activation,
                                  file_name="Autoencoder_AdjustedIS"+str(input_size)+"HS"+ str(hidden_layer_sizes)+ "OS" + str(output_size) + class_type)
    """

    for n in range(0, 12, 2):
        class_path="filmdata/classesGenres/class-All"
        loss = "mse"
        optimizer="rmsprop"
        epochs=1
        hidden_activation="tanh"
        layers_to_cut_at=[1]
        save_space=True
        batch_size=1
        class_mode="categorical"
        is_autoencoder = True
        hidden_layer_sizess = [200,200,200,200]
        print n
        noise = n * 0.1
        print noise
        amount_of_sda=4
        numpy_vector_path=None#D:\Dropbox\PhD\My Work\Code\MSDA\Python\Data\IMDB\Transformed/" + "NORMALIZEDno_below200no_above0.25.npy"
        input_size=200
        output_size=200
        learn_rate = 0.01
        vector_path = "filmdata/films200.mds/films200.mds"
        reg = 0.0
        SDA_end_space = []
        SDA_encoders = []
        for s in range(0, len(hidden_layer_sizess)):
            if s >= 1:
                output_activation="tanh"
                input_size = hidden_layer_sizess[s]
                output_size = hidden_layer_sizess[s]
                #noise = noise + 0.1
                reg = 0.0
                noise += 0.2
                SDA = NeuralNetwork( autoencoder_space=SDA_end_space, learn_rate=learn_rate, batch_size=batch_size,
                        epochs=epochs, is_autoencoder=is_autoencoder,  class_mode=class_mode, reg=reg,
                                     hidden_layer_size=hidden_layer_sizess[s],
                                           numpy_vector_path=numpy_vector_path, noise=noise, class_path=class_path,
                                        save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer,
                                           hidden_activation=hidden_activation, output_activation=output_activation, hidden_layer_sizes=hidden_layer_sizess,
                                          file_name="N"+str(noise)+"R"+str(reg)+hidden_activation+output_activation + loss+ str(epochs) + hidden_activation +
                                                    str(hidden_layer_sizess[s])  + str(amount_of_sda)+ "SDA" + str(s+1))
            else:
                output_activation="tanh"
                SDA = NeuralNetwork( autoencoder_space= None, epochs=epochs, is_autoencoder=is_autoencoder,  class_mode=class_mode,  output_size=output_size,
                                         class_path=class_path, numpy_vector_path=numpy_vector_path, noise=noise,reg=reg, learn_rate=learn_rate, vector_path=vector_path,
                                        save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer, batch_size=batch_size,
                                     hidden_layer_size=hidden_layer_sizess[s],
                                           hidden_activation=hidden_activation, output_activation=output_activation, hidden_layer_sizes=hidden_layer_sizess,
                                          file_name="N"+str(noise)+"R"+str(reg)+hidden_activation+output_activation + loss+ str(epochs) + hidden_activation + str(hidden_layer_sizess[s]) + str(amount_of_sda) + "SDA" + str(s+1))
                noise = 0.4
            SDA_end_space = SDA.getEndSpace()
            SDA_encoders.append(SDA.getEncoder())
        """
        class_path="filmdata/classesGenres/class-All"
        loss="binary_crossentropy"
        optimizer="adagrad"
        epochs=200
        is_autoencoder = False
        autoencoder_space = None
        layers_to_cut_at = [0,1,2,3,4,5,6]
        output_size=25
        batch_size=100
        learn_rate = 0.05
        vector_path="filmdata/films200.mds/films200.mds"
        hidden_activation = "relu"
        hidden_layer_sizes=[]
        output_activation="softmax"
        print "FINE TUNING"
        class_mode="binary"
        SDA = NeuralNetwork( autoencoder_space= autoencoder_space, encoders=SDA_encoders, reg=reg, learn_rate=learn_rate,
                        epochs=epochs, is_autoencoder=is_autoencoder, class_mode=class_mode, batch_size=batch_size,
                                          output_size=output_size, class_path=class_path, numpy_vector_path=numpy_vector_path, noise=noise,
                                        save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer,
                                           hidden_activation=hidden_activation,  output_activation=output_activation, hidden_layer_sizes=hidden_layer_sizes,
                                          file_name=str(noise)+hidden_activation+output_activation + loss+ str(epochs) + hidden_activation + str(hidden_layer_sizes) + str(amount_of_sda))
        """
    print "dun"
    """
    #WORKING @ HIGHEST ACCURACY, F1, ETC FOR MULTI-LABEL, FOLLOWING "REVISITING MULTI-LABEL NEURAL NETS" GUIDELINES
    class_type = "Genres"
    loss = "binary_crossentropy"
    optimizer="adagrad"
    epochs=50
    hidden_activation="relu"
    output_activation="softmax"
    layers_to_cut_at=[1,2]
    save_space = True
    denoising = False
    output_size = 25
    input_size = 200
    class_mode = "binary"
    is_autoencoder = False
    hidden_layer_sizes = [200,200]

    numpy_vector_path=None
    path="newdata/spaces/"
    filenames = ["AUTOENCODER0.5tanhtanhmse60tanh[200]4SDA4"]
    """
    "AUTOENCODER0.8tanhtanhmse15tanh[1000]4SDA1","AUTOENCODER0.8tanhtanhmse60tanh[200]4SDA2","AUTOENCODER0.8tanhtanhmse30tanh[1000]4SDA3",
    "AUTOENCODER0.8tanhtanhmse60tanh[200]4SDA4"
    "AUTOENCODER0.2tanhtanhmse15tanh[1000]4SDA1","AUTOENCODER0.2tanhtanhmse60tanh[200]4SDA2","AUTOENCODER0.2tanhtanhmse30tanh[1000]4SDA3",
    "AUTOENCODER0.2tanhtanhmse60tanh[200]4SDA4"

    """
    dimension_100 = NeuralNetwork( epochs=epochs, is_autoencoder=is_autoencoder, input_size=input_size, class_mode=class_mode, vector_path=None,
                                      output_size=output_size, class_type=class_type, hidden_layer_sizes=hidden_layer_sizes, numpy_vector_path=numpy_vector_path,
                                    save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer,
                                       hidden_activation=hidden_activation, output_activation=output_activation,
                                      file_name="GenresUVInputrelu,softmax,bc,adagrad" + str(epochs) + hidden_activation + str(hidden_layer_sizes))
    for f in filenames:
        vector_path = path+f+".mds"
        class_type = "Genres"
        dimension_100 = NeuralNetwork( epochs=epochs, is_autoencoder=is_autoencoder, input_size=input_size, class_mode=class_mode, vector_path=vector_path,
                                      output_size=output_size, class_type=class_type, hidden_layer_sizes=hidden_layer_sizes, numpy_vector_path=numpy_vector_path,
                                    save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer,
                                       hidden_activation=hidden_activation, output_activation=output_activation,
                                      file_name=f+"GenresUVInputrelu,softmax,bc,adagrad" + str(epochs) + hidden_activation + str(hidden_layer_sizes))
    """
    """
    class_type = "Keywords"

    dimension_100 = NeuralNetwork( epochs=epochs, is_autoencoder=is_autoencoder, input_size=input_size, class_mode=class_mode,
                                  output_size=output_size, class_type=class_type, hidden_layer_sizes=hidden_layer_sizes, numpy_vector_path=numpy_vector_path,
                                save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer,
                                   hidden_activation=hidden_activation, output_activation=output_activation,
                                  file_name="Keywordsrelu,softmax,bc,adagrad" + str(epochs) + hidden_activation + str(hidden_layer_sizes))

    class_type = "All"

    dimension_100 = NeuralNetwork( epochs=epochs, is_autoencoder=is_autoencoder, input_size=input_size, class_mode=class_mode,
                                  output_size=output_size, class_type=class_type, hidden_layer_sizes=hidden_layer_sizes, numpy_vector_path=numpy_vector_path,
                                save_space=save_space, layers_to_cut_at=layers_to_cut_at, loss=loss, optimizer=optimizer,
                                   hidden_activation=hidden_activation, output_activation=output_activation,
                                  file_name="Allrelu,softmax,bc,adagrad" + str(epochs) + hidden_activation + str(hidden_layer_sizes))

    print "dun"
    """





if  __name__ =='__main__':main()

