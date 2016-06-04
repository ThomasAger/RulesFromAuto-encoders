import os
import shutil
import numpy as np
"""

DATA IMPORTING TASKS

"""


def importNumpyVectors(numpy_vector_path=None):
    movie_vectors = np.load(numpy_vector_path)
    movie_vectors = np.ndarray.tolist(movie_vectors)
    movie_vectors = list(reversed(zip(*movie_vectors)))
    movie_vectors = np.asarray(movie_vectors)
    return movie_vectors

def importString(file_name):
    with open(file_name, "r") as infile:
        string_array = [line.strip() for line in infile]
    return string_array

def importLabels(file_name):
    with open(file_name, "r") as infile:
        int_array = [map(int, line.strip().split()) for line in infile]
    return int_array

def importVectors(file_name):
    with open(file_name, "r") as infile:
        int_array = [map(float, line.strip().split()) for line in infile]
    return int_array

def importDiscreteVectors(file_name):
    with open(file_name, "r") as infile:
        int_array = [line.strip().split() for line in infile]
    return int_array


def getFns(folder_path):
    print folder_path
    file_names = []
    print len(os.listdir(folder_path))
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    for i in onlyfiles:
        if i != "class-all" and i != "nonbinary" and i != "low_keywords" and i != "class-All" and i != "archive":
            file_names.append(i)
    return file_names


"""

DATA EDITING TASKS

"""


def splitData(training_data, movie_vectors, movie_labels):
    x_train = np.asarray(movie_vectors[:training_data])
    y_train = np.asarray(movie_labels[:training_data])
    x_test = np.asarray(movie_vectors[training_data:])
    y_test = np.asarray(movie_labels[training_data:])
    return  x_train, y_train,  x_test, y_test

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
        file.write(str(array[i]) + "\n")
    file.close()

def rewriteToAll(array_of_all, place_to_write):
    movie = []
    len_element = len(array_of_all[0])
    len_all = len(array_of_all)
    print len_element
    print len_all
    file = open(place_to_write, "w")

    for j in xrange(len_element):
        for i in xrange(len_all):
            file.write(str(array_of_all[i][j]) + " ")
        file.write("\n")
    file.close()

def writeAllPhrasesToSingleFile(files_folder, phrases):
    file_names = getFns(files_folder)
    matched_file_names = []
    all_names = []

    for p in phrases:
        #print p
        for f in file_names:
            #print f
            if "class-" + p.strip() == f.strip():
                print "matched:", f
                matched_file_names.append(f)
                break

    for f in matched_file_names:
        if f != "archive" and f != "nonbinary" and f != "class-all" and f != "class-low-all"\
                and f != "low_keywords":
            file = open(files_folder + "/" + f, "r")
            lines = file.readlines()
            current_file = []
            for line in lines:
                current_file.append(line.strip())
            all_names.append(current_file)

            file.close()
    print "File about to be written."
    rewriteToAll(all_names, files_folder + "\class-all")

#writeAllPhrasesToSingleFile("filmdata/classesPhrases", importString("filmdata/uniquePhrases.txt"))
"""

Note: This method is currently missing 12 files.

"""

def writeAllKeywordsToSingleFile(files_folder):
    file_names = getFns(files_folder)
    all_names = []
    for f in file_names:
        if f != "all":
            file = open(files_folder + "/" + f, "r")
            lines = file.readlines()
            current_file = []
            for line in lines:
                current_file.append(line.strip())
            all_names.append(current_file)

            file.close()
    print "File about to be written."
    rewriteToAll(all_names, files_folder + "\class-all")
#writeAllKeywordsToSingleFile("F:\Dropbox\PhD\My Work\Code\MSDA\Python\Data\unprocessed.tar\sorted_data\dvd\one_hot")

def mean_of_array(array):
    total = array[0]
    for a in range(1, len(array)):
        for v in range(0, len(array[a])):
            total[v] = total[v] + array[a][v]
    for v in range(len(total)):
        total[v] = total[v] / len(array)
    return total


def checkIfInArray(array, thing):
    for t in array:
        if thing == t:
            return True
    return False

def getIndexInArray(array, thing):
    for t in range(len(array)):
        if thing == array[t]:
            return t
    return None

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def sortByArray(array_to_sort, array_to_sort_by):
    array_to_sort_by = np.asarray(array_to_sort_by)
    array_to_sort = np.asarray(array_to_sort)
    sorting_indices = array_to_sort_by.argsort()
    sorted_array = array_to_sort[sorting_indices[::-1]]
    return sorted_array.tolist()
"""
sortAndOutput("filmdata/KeywordData/most_common_keywords.txt", "filmdata/KeywordData/most_common_keywords_values.txt",
              "filmdata/KeywordData/most_common_keywordsSORTED.txt", "filmdata/KeywordData/most_common_keyword_valuesSORTED.txt")
"""
