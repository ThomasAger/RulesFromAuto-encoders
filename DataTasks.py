import os

"""

DATA IMPORTING TASKS

"""



def getMovieVectors(input_size=200, vector_path=None):
    if vector_path is None:
        return convertToFloat(importString("filmdata/films" + str(input_size) + ".mds/films" + str(input_size) + ".mds"))
    else:
        return convertToFloat(importString(vector_path))

def getAllLabels(class_type):
    return convertToInt(importString("filmdata/classes" + str(class_type) + "/class-All"), True)

def getClassByClass(class_type):
    all_classes = convertToInt(importString("filmdata/classes" + str(class_type) + "/class-All"), True)
    class_by_class = []
    for y in range(len(all_classes[0])):
        building_array = []
        for x in range(len(all_classes)):
            building_array.append(all_classes[x][y])
        class_by_class.append(building_array)
    return class_by_class

def getLabels(class_type, class_names):
    labels = []
    for class_name in class_names:
        labels.append(convertToInt(importString("filmdata/classes"+class_type+"/class-" + class_name + ""), False))
    return labels

def importString(file_name):
    file = open(file_name, "r")
    temp_strings = []
    with file as s:
        temp_strings.extend(s)
    return temp_strings

def getAllFileNames(folder_path):
    file_names = []
    for i in os.listdir(folder_path):
        if i.endswith("All") == False and i.endswith("all") == False:
            file_names.append(i)
    return file_names

def importFile(file):
    temp_strings = []
    with file as s:
        temp_strings.extend(s)
    return temp_strings

def importClasses(root_dir):
    classes = []
    for subdir, dirs, files in os.walk(root_dir):
        for directory_file in files:
            classes.append(convertToInt(importString(root_dir + directory_file), False))
    return classes

"""

DATA EDITING TASKS

"""

def convertTo2d(movie_labels):
    movie_labels_2d = []
    for l in range(len(movie_labels)):
        if movie_labels[l] == 0:
            movie_labels_2d.append([1, 0])
        else:
            movie_labels_2d.append([0, 1])
    return movie_labels_2d

def splitData(training_data, movie_names, movie_vectors, movie_labels):
    n_train = movie_names[:training_data]
    x_train = movie_vectors[:training_data]
    y_train = movie_labels[:training_data]

    n_test = movie_names[training_data:]
    x_test = movie_vectors[training_data:]
    y_test = movie_labels[training_data:]
    return n_train, x_train, y_train, n_test, x_test, y_test

def convertTo2d(movie_labels):
    movie_labels_2d = []
    for l in range(len(movie_labels)):
        if movie_labels[l] == 0:
            movie_labels_2d.append([1, 0])
        else:
            movie_labels_2d.append([0, 1])
    return movie_labels_2d

def sampleData(X_data, Y_data, sparsity):
    Y_arranged_data = []
    X_arranged_data = []
    active_count = 0
    for d in range(len(Y_data)):
        if Y_data[d] == 1:
            active_count = active_count + 1
            Y_arranged_data.insert(0, Y_data[d])
            X_arranged_data.insert(0, X_data[d])
        else:
            Y_arranged_data.append(Y_data[d])
            X_arranged_data.append(X_data[d])
    amountOf0 = active_count + active_count*sparsity
    Y_sampled_data = Y_arranged_data[:amountOf0]
    X_sampled_data = X_arranged_data[:amountOf0]
    return X_sampled_data, Y_sampled_data


def countHitsAndMisses(our_classes, correct_classes):
    failures1 = 0
    failures0 = 0
    total1 = 0
    total0 = 0
    for c in range(len(our_classes)):
        if our_classes[c] != correct_classes[c]:
            if correct_classes[c] == 1:
                failures1 = failures1 + 1
            else:
                failures0 = failures0 + 1
    return failures1, failures0


def countValues(count_array):
    valueType1 = 0
    valueType0 = 0
    for c in count_array:
        if c == 1:
            valueType1 = valueType1 + 1
        else:
            valueType0 = valueType0 + 1
    return valueType1, valueType0

def convertToFloat(string_array):
    temp_floats = []
    for string in string_array:
        float_strings = string.split()
        i = 0
        for float_string in float_strings:
            float_strings[i] = float(float_string)
            i = i + 1
        temp_floats.append(float_strings)
    return temp_floats


def convertToInt(string_array, vectors):
    temp_ints = []
    for string in string_array:
        int_strings = string.split()

        i = 0
        for int_string in int_strings:
            int_strings[i] = int(int_string)
            i = i + 1
        if(vectors):
            temp_ints.append(int_strings)
        else:
            temp_ints.extend(int_strings)
    return temp_ints

    # refactor this
def printClasses(Y_test, Y_predicted, n_test):
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

"""

OUTPUT TASKS

"""

def write2dArray(array, name):
    file = open(name, "w")

    for i in xrange(len(array)):
        for n in xrange(len(array[i])):
            file.write(str(array[i][n]) + " ")
        file.write("\n")
    file.close()

def write1dArray(array, name):
    file = open(name, "w")
    for i in xrange(len(array)):
        file.write(str(array[i]) + " ")
        file.write("\n")
    file.close()

def rewriteToAll(array_of_all, place_to_write):
    movie = []
    len_all = len(array_of_all[0])
    len_element = len(array_of_all)
    file = open(place_to_write, "w")
    for i in xrange(len_all):
        for j in xrange(len_element):
            file.write(str(array_of_all[j][i]) + " ")
        file.write("\n")
    file.close()
