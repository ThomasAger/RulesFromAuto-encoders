
import DataTasks as dt
import re
from fuzzywuzzy import fuzz, process
import numpy as np
import string
from collections import defaultdict

"""

Get the top movies from the ratings list

"""

def outputTopByVotes(amount_of_votes):
    file = open("filmdata/ratings.list/ratings.list", "r")
    lines = file.readlines()
    top_movies = []
    top_ratings = []
    for line in lines:
        top_ratings.append(int(line.split()[1]))
    top_ratings = np.asarray(top_ratings)
    indices = np.argpartition(top_ratings, -amount_of_votes)[-amount_of_votes:]
    for i in indices:
        just_movie = lines[i].split()[3:]
        just_movie = " ".join(just_movie)
        just_movie = just_movie.split('{')
        just_movie = just_movie[0]
        just_movie = just_movie.split('(')
        try:
            if not re.findall(r'\d+', just_movie[2])[0]:
                del just_movie[2]
            else:
                just_movie[0] = just_movie[0] + "(" + just_movie[1]
                del just_movie[1]
        except IndexError:
            print
        try:
            year = re.findall(r'\d+', just_movie[1])[0]
        except IndexError:
            print "FALED", just_movie
        if just_movie[0].endswith(' '):
            just_movie[0] = just_movie[0][:-1]
        if just_movie[0].startswith('"') and just_movie[0].endswith('"'):
            just_movie[0] = just_movie[0][1:-1]
        just_movie = just_movie[0] + " " + str(year)
        print just_movie
        top_movies.append(just_movie)
    dt.write1dArray(top_movies, "filmdata/top50000moviesbyvotes.txt")

"""

Get the top movies by the ratings list, but don't format it.

"""

def getUnformattedTopByVotes(amount_of_votes):
    file = open("filmdata/ratings.list/ratings.list", "r")
    lines = file.readlines()
    top_movies = []
    top_ratings = []
    for line in lines:
        top_ratings.append(int(line.split()[1]))
    top_ratings = np.asarray(top_ratings)
    indices = np.argpartition(top_ratings, -amount_of_votes)[-amount_of_votes:]
    for i in indices:
        just_movie = lines[i].split()[3:]
        just_movie = " ".join(just_movie)
        print just_movie
        top_movies.append(just_movie)
    dt.write1dArray(top_movies, "filmdata/imdb_formatted_top50000.txt")

"""

From the main keywords list, get the n most common keywords.

"""

def getCommonIMDBKeywords(top_value):
    file = open("filmdata\keywords.list\keywords.list", "r")
    lines = file.readlines()
    keywords_list = lines[62:79460]
    keyword_amounts = []
    keywords = []
    common_keywords = []
    for line in keywords_list:
        split_line = line.split()
        for i in range(len(split_line)):
            if split_line[i].startswith('('):
                keyword_amounts.append(int(re.findall(r'\d+', split_line[i])[0]))
                split_line[i] = "()"
        split_line = " ".join(split_line)
        split_line = split_line.split("()")

        for blob in split_line:
            if not blob:
                break
            blob = blob.strip()
            keywords.append(blob)
    keyword_amounts = np.asarray(keyword_amounts)
    indices = np.argpartition(keyword_amounts, -top_value)[-top_value:]
    lowest_value = 2147000000
    for i in indices:
        common_keywords.append(keywords[i])
        if keyword_amounts[i] < lowest_value:
            lowest_value = keyword_amounts[i]
    print "These keywords have appeared in at least", lowest_value, "Films"
    return common_keywords


        # if the title and the year match
        # save the entire line to a new array

"""

Given a set of movie names, get the IMDB movie lines for them

"""

def getIMDBKeywordsForMovieNames(movie_names):
    stripped_movie_names = []
    for movie in movie_names:
        stripped_movie_names.append(movie.replace('\n', ''))
    stripped_movie_names = sorted(stripped_movie_names)
    split_names = []
    split_years = []
    for stripped_movie_name in stripped_movie_names:
        split = stripped_movie_name.split()
        split_year = split[len(split)-1]
        split_years.append(split_year)
        split_names.append(stripped_movie_name[:-len(split_year)-1])

    file = open("filmdata\keywords.list\keywords.list", "r")
    lines = file.readlines()
    keywords_list = lines[79748:]
    matched_lines = []
    x = 0
    last_line = keywords_list[0]
    matched = False
    while x < 50000:
        for line in keywords_list:
            split_line = line.rsplit('(', 2)
            movie_name = split_line[0].rstrip()

            if movie_name.startswith('"') and movie_name.endswith('"'):
                movie_name = movie_name[1:-1]
            try:
                movie_year = str(re.findall(r'\d+', split_line[1])[0])
            except IndexError:
                movie_year = "NULL"
            if not movie_name:
                movie_name = "'NULL"
            formatted_line = movie_name.rstrip() + " " + str(movie_year).rstrip()

            if matched is True and formatted_line == last_line:
                matched_lines.append(line)
                print split_names[x], line
            elif matched is False and similar(movie_name.strip().upper(), split_names[x].strip().upper()) and movie_year == split_years[x]:
                matched = True
                matched_lines.append(line)
                print split_names[x], line
            elif matched is True and formatted_line != last_line:
                matched = False
                x = x + 1
            last_line = formatted_line
        print "cycled through"



    print "Found:", x
    dt.write1dArray(matched_lines, "filmdata/imdb_movie_keywords.txt")
        # if the title and the year match
        # save the entire line to a new array

"""

Get 50k films. Pretty sure this method sucks.

"""

def get50k(movie_names):
    stripped_movie_names = []
    for movie in movie_names:
        stripped_movie_names.append(movie.replace('\n', ''))
    stripped_movie_names = sorted(stripped_movie_names)
    file = open("filmdata\keywords.list\keywords.list", "r")
    lines = file.readlines()
    lines = sorted(lines)
    keywords_list = lines[79748:]
    matched_lines = []
    x = 0
    back_to_start = False
    while x < 50000:
        for line in keywords_list:
            if "{" in stripped_movie_names[x]:
                x = x + 1
                break
            split_line = re.split(r'\t+', line)
            if similar(split_line[0], stripped_movie_names[x]):
                matched_lines.append(line)
                x = x + 1
                print "MATCHED"
            if ord(split_line[0][1]) > ord(stripped_movie_names[x][1]):
                x = x + 1
                break
        print "cycled through, but couldn't find" + stripped_movie_names[x]
        if back_to_start == True:
            x = x + 1
            back_to_start = False
        else:
            back_to_start == True


    print "Found:", x
    dt.write1dArray(matched_lines, "filmdata/top50000_keywords_list.txt")
        # if the title and the year match
        # save the entire line to a new array

def getMovieData(class_type="Keywords", class_names=None, vector_path=None, input_size=200, class_by_class=False):
    movie_names = dt.importString("filmdata/filmNames.txt")
    if class_type == "All":
        movie_labels = dt.getAllLabels("Keywords")
        genre_labels = dt.getAllLabels("Genres")
        for c in range(len(movie_labels)):
            movie_labels[c].extend(genre_labels[c])
    else:
        if class_names is None:
            if class_by_class:
                movie_labels = dt.getClassByClass(class_type)
            else:
                movie_labels = dt.getAllLabels(class_type)
        else:
            movie_labels = dt.getLabels(class_type, class_names)
    if vector_path is None:
        movie_vectors = dt.getMovieVectors(input_size=input_size)
    else:
        movie_vectors = dt.getMovieVectors(input_size=input_size, vector_path=vector_path)
    return movie_names, movie_vectors, movie_labels


def similar(seq1, seq2):
    if seq1 == seq2:
        return True
    else:
        return False

"""

Remove all of the stuff left over when the movie data was gathered prior

"""

def trimMovieData(file_string):
    new_movie_data = []
    movie_data_file = open(file_string)
    with movie_data_file as myFile:
        for num, line in enumerate(myFile, 1):
            if "{" not in line and "(V)" not in line and "(TV)" not in line and "(VG)" not in line and len(line) > 2 and line.startswith(" ") is False and line.startswith("\t") is False and line.startswith("\n") is False:
                new_movie_data.append(line[:-1])
    print "dun"
    dt.write1dArray(new_movie_data, file_string + "trimmed")

"""

Get all lines that match a set of movie names / years

"""

def getMovieDataFromIMDB(movie_strings):
    movie_data = []
    write_line_file = open("filmdata/IMDB_movie_data_15000FRPDURING.txt", "w")
    failed_movies = []
    names = []
    years = []
    movie_strings = movie_strings[9536:]
    for movie_string in movie_strings:
        names.append(movie_string[:-6])
        years.append(movie_string[-5:].strip())

    #movie_data_file =, "r")
    for n in range(len(names)):
        found = False
        last_name = ""
        old_num = 1636185
        with open("filmdata/IMDB_movie_data.txttrimmed") as myFile:
            for num, line in enumerate(myFile, 1):
                num = old_num
                split_line = re.split(r'\t+', line)
                movie_string = split_line[0]
                #print line
                movie_name = movie_string.split()
                #print movie_name
                del movie_name[len(movie_name)-1]
                movie_name = " ".join(movie_name)
                if found is True and last_name != movie_name:
                    break
                movie_year = int(re.findall(r'\d+', movie_string.split()[len(movie_string.split())-1])[0])
                #print names[n], movie_name
                #print years[n], movie_year
                if similar(names[n].upper().strip(), movie_name.upper().strip()) and int(years[n]) == int(movie_year):
                #if names[n] in line and years[n] in line and "(TV)" not in line and "(V)" not in line:
                    movie_data.append(line)
                    write_line_file.write(line)
                    found = True
                    old_num = num
                last_name = movie_name
        if found is False:
            failed_movies.append(names[n])
            print "FAILED:", names[n]
    print "Total failed", len(failed_movies)
    write_line_file.close()
    dt.write1dArray(failed_movies, "filmdata/failed_movies_final_push.txt")
    return movie_data

"""

Out of a list of movie_string : keyword, get the top_value most common keywords
from that list.

"""

def getMostCommonKeywords(top_value, file_name):
    common_keywords = []
    file = open(file_name, "r")
    lines = file.readlines()
    keywords = []
    keyword_amounts = []
    for line in lines:
        keyword = line.split()[len(line.split)-1]
        for i in range(len(keywords)):
            if keywords[i] == keyword:
                keyword_amounts[i] += 1
                continue
        keywords.append(keyword)
        keyword_amounts.append(1)
    keyword_amounts = np.asarray(keyword_amounts)
    indices = np.argpartition(keyword_amounts, -top_value)[-top_value:]
    for i in indices:
        common_keywords.append(keywords[i])
    return common_keywords

"""

Given a list of keywords to match on, create a [0 ... len(keywords)] vector
For every movie in the text corpus, output as a single array of vectors.

"""

def getKeywordVectors(keywords, movie_names, file_name):
    vector = []
    vectors = []
    for n in range(len(keywords)):
        with open(file_name) as myFile:
            for num, line in enumerate(myFile, 1):
                value = 0
                if keywords[n] in line:
                    value = 1
                split_line = line.split()
                del split_line[len(split_line)-1]
                line = " ".join(line)
                if line != last and num != 0 or num == len(myFile):
                    vector.append(value)
                last = line
        print len(vector), vector
        vectors.append(vector)
    return vectors

"""

Return a list of IDs ordered by the movie_strings

"""

def getIDs(movie_strings):
    ordered_IDs = []
    movie_names = []
    for name in movie_strings:
        movie_names.append(name[:-5])
    id_mappings = open("filmdata/films-ids.txt", "r")
    id_mappings_lines = id_mappings.readlines()
    found_name = False
    failed_names = []
    x = 0
    for name in movie_names:
        for line in id_mappings_lines:
            mapping_id = line.split()[0]
            mapping_name = re.split(r'\t+', line)[2]
            if similar(name.upper().strip(), mapping_name.upper().strip()):
                ordered_IDs.append(mapping_id)
                found_name = True
                break
        if found_name is True:
            found_name = False
        else:
            failed_names.append(name)
            ordered_IDs.append(-1)
        x += 1
        print x
    print failed_names
    return ordered_IDs

"""

Given a list of ordered IDs for the 15,0000 films.
And get all of the phrases mentioned in those IDs and return that list.

Then, given a list of phrases mentioned, create a vector for every movie and return
as an array of those vectors. Information is found within the film files.

"""

def getUniquePhrases(ordered_IDs, count_min, maintain_quantities):
    #movie_files = []
    phrase_to_count = {}
    # First collect together all possible vectors
    for ID in ordered_IDs:
        ID = ID.strip()
        if ID != "-1":
            open_ID = open("filmdata/Tokens/" + ID + ".film", "r")
            #movie_files.append(open_ID)
            lines = open_ID.readlines()[1:]
            for line in lines:
                split_line = line.split()
                try:
                    phrase_to_count[split_line[0]] += 1
                except KeyError:
                    phrase_to_count[split_line[0]] = 0
    common_phrases = []
    for key, value in phrase_to_count.iteritems():
        if value >= 100:
            common_phrases.append(key)

    unique_phrases = np.unique(common_phrases)
    unique_phrases = unique_phrases.tolist()
    return unique_phrases

def getVectors(ordered_IDs, unique_phrases):
    vectors = [[0 for x in range(len(ordered_IDs))] for x in range(len(unique_phrases))]
    vectors_maintained = [[0 for x in range(len(ordered_IDs))] for x in range(len(unique_phrases))]
    print len(vectors), len(vectors[0])
    multi_dictionary = {}
    dict_mapping = {}
    for i in range(len(ordered_IDs)):
        ordered_IDs[i] = str(ordered_IDs[i])
    for i in range(len(ordered_IDs)):
        ordered_IDs[i] = ordered_IDs[i].strip()
        dict_mapping[ordered_IDs[i]] = i
        if ordered_IDs[i] != "-1":
            file = open("filmdata/Tokens/" + ordered_IDs[i] + ".film", "r")
            lines = file.readlines()[1:]
            for line in lines:
                split_line = line.split()
                multi_dictionary[(ordered_IDs[i], split_line[0])] = int(split_line[1])
            file.close()
        else:
            multi_dictionary[(ordered_IDs[i], split_line[0])] = 0
    #print multi_dictionary
    for up in range(len(unique_phrases)):
        unique_phrases[up] = unique_phrases[up].strip()

    print len(multi_dictionary)
    for p in range(len(unique_phrases)):
        for key, value in multi_dictionary.iteritems():
            if key[1] == unique_phrases[p]:
                #print p, dict_mapping[key[0].strip()], key[0]
                vectors_maintained[p][dict_mapping[key[0].strip()]] = value
                vectors[p][dict_mapping[key[0]]] = 1
        print unique_phrases[p]
        dt.write1dArray(vectors_maintained[p], "filmdata/classesPhrases/nonbinary/class-" + unique_phrases[p])
        dt.write1dArray(vectors[p], "filmdata/classesPhrases/class-" + unique_phrases[p])

    return vectors_maintained, vectors

def outputPhrases():
    IDs = dt.importString("filmdata/filmIDs.txt")
    unique_phrases = dt.importString("filmdata/uniquePhrases.txt")
    vectors_maintained, vectors = getVectors(IDs, unique_phrases)
    dt.write2dArray(vectors, "filmdata/classesPhrases/class-all")
    dt.write2dArray(vectors_maintained, "filmdata/classesPhrases/nonbinary/class-all")

def outputKeywords():
    movie_strings = dt.importString("filmdata/filmNames.txt")
    movie_data = getMovieDataFromIMDB(movie_strings)
    #dt.write1dArray(movie_data, "IMDB_movie_data_15000FRP.txt")
    commonality = 0
    common_keywords = getMostCommonKeywords(0, "filmdata/IMDB_movie_data.txt")
    dt.write1dArray(common_keywords, "filmdata/common_keywords_15k_commanility_" + str(commonality))
    vectors = getKeywordVectors(common_keywords, movie_strings, "")
    dt.write2dArray(vectors, "filmdata/classesKeywords/class-extra-all-commonality-" + str(commonality))
    #common_keywords = getCommonIMDBKeywords(20000)
    #print common_keywords
    #print len(common_keywords)

#outputPhrases()
#outputKeywords()
