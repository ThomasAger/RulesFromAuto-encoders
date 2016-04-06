
import DataTasks as dt
import re
import numpy as np
import string
from collections import defaultdict
import random
import theano
from theano.tensor.shared_randomstreams import RandomStreams
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

def getMovieData(class_type="Keywords", class_names=None, vector_path=None, class_path=None, input_size=200, class_by_class=False):
    movie_names = ""
    if class_path is not None:
        movie_labels = dt.convertToInt(dt.importString(class_path), True)
        movie_labels = zip(*movie_labels)
        movie_labels = list(movie_labels)
        for l in range(len(movie_labels)):
            movie_labels[l] = list(movie_labels[l])
        print type(movie_labels), len(movie_labels), len(movie_labels[0])
    else:
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
                if len(class_names) == 1:
                    movie_labels = dt.getLabel(class_type, class_names)
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
    write_line_file = open("filmdata/Found_Missing.txt", "w")
    failed_movies = []
    names = []
    years = []
    for movie_string in movie_strings:
        names.append(movie_string[:-6])
        years.append(movie_string[-5:].strip())

    for n in range(len(names)):
        found = False
        last_name = ""
        old_num = 0
        with open("filmdata/IMDB_movie_data.txttrimmed") as myFile:
            for num, line in enumerate(myFile, 1):
                num = old_num
                split_line = re.split(r'\t+', line)
                movie_string = split_line[0]
                movie_name = movie_string.split()
                del movie_name[len(movie_name)-1]
                movie_name = " ".join(movie_name)
                if found is True and last_name != movie_name:
                    break
                movie_year = int(re.findall(r'\d+', movie_string.split()[len(movie_string.split())-1])[0])
                if similar(names[n].upper().strip(), movie_name.upper().strip()) and int(years[n]) == int(movie_year):
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

""" Make whitespace the only split, remove punctuation """

def makeConsistentKeywords(file_name, new_file_name):
    new_file = []
    with open(file_name) as my_file:
        for num, line in enumerate(my_file, 1):
            if "{" in line or "?" in line:
                continue
            split_line = re.split(r'\t+', line)
            split_on_bracket = split_line[0].split(" (")
            if split_on_bracket[1].startswith("1") == False and split_on_bracket[1].startswith("2") == False:
                year = split_on_bracket[2][:4]
                name = "".join([split_on_bracket[0], split_on_bracket[1]])
            else:
                year = split_on_bracket[1][:4]
                name = split_on_bracket[0]
            name = name.translate(None, string.punctuation)
            year = year.translate(None, string.punctuation)
            keyword = re.sub(r'\s+', '', split_line[1]).translate(None, string.punctuation)
            name_and_year = "\t".join([name, year])
            new_line = "\t".join([name_and_year, keyword])
            new_file.append(new_line)
            print new_line

    dt.write1dArray(new_file, new_file_name)

""" Make whitespace the only split, remove punctuation """

def makeConsistent(file_name, new_file_name):
    new_file = []
    with open(file_name) as my_file:
        for num, line in enumerate(my_file, 1):
            line = line.strip()
            name = line[:-4]
            year = line[len(line)-4:]

            name = name.translate(None, string.punctuation)
            year = year.translate(None, string.punctuation)
            new_line = "\t".join([name, year])
            new_file.append(new_line)
            print new_line

    dt.write1dArray(new_file, new_file_name)

def divideSpace(space, amount_to_divide_by):
    for s in space:
        for i in range(len(s)):
            s[i] = s[i] / amount_to_divide_by

    return space

#dt.write2dArray(divideSpace(dt.getMovieVectors(vector_path="newdata/spaces/AUTOENCODERnoise0100.mds"), 6), "newdata/spaces/AUTOENCODERnoise0100clean.mds")

def removeGaps(file_name, new_file_name):
    new_file = []
    with open(file_name) as my_file:
        for num, line in enumerate(my_file, 1):
            new_file.append(line.strip())
    dt.write1dArray(new_file, new_file_name)

#removeGaps("filmdata/KeywordData/matched_lines_NEW.txt", "filmdata/KeywordData/matched_lines_consistent.txt")

""" Gather which films are missing from our collected list """

def getMissing(file_name, complete_list):
    missing_list = []
    name_list = []
    for i in complete_list:
        name_list.append(i.strip()[:-4])

    compare_list = []
    with open(file_name) as my_file:
        for num, line in enumerate(my_file, 1):
            split_line = re.split(r'\t+', line)
            stripped_name = split_line[0].strip()[:-6]
            compare_list.append(stripped_name)

    compare_set = set(compare_list)
    unique_list = list(compare_set)

    for i in range(len(name_list)):
        found = False
        for l in compare_set:
            if l == name_list[i]:
                print "Found", name_list[i]
                found = True
                break
        if found == False:
            print "Didn't find", name_list[i]
            missing_list.append(complete_list[i].strip())
    return missing_list

""" Using two consistent files, find lines that match between them """

def getMatchedLines(file_name, lines_to_match):
    matched_lines = []
    failed_lines = []
    match_names = []
    match_years = []
    for line in lines_to_match:
        match_names.append(re.split(r'\t+', line)[0])
        match_years.append(re.split(r'\t+', line)[1])
    file = open(file_name, "r")
    lines = file.readlines()
    for i in range(len(lines_to_match)):
        matched = False
        last_movie = ""
        for l in range(len(lines)):
            if matched is True and re.split(r'\t+', lines[l])[0] != last_movie:
                break
            split_line = re.split(r'\t+', lines[l])
            split_line[0] = re.sub(r'\s+', '', split_line[0].translate(None, string.punctuation).lower())
            match_names[i] = re.sub(r'\s+', '', match_names[i].translate(None, string.punctuation).lower())
            if split_line[0] == match_names[i]:
                matched_lines.append(lines[l])
                matched = True
                last_movie = re.split(r'\t+', lines[l])[0]
                print "Found a line for " + last_movie
                continue
        if matched:
            print "Matched", lines_to_match[i]
        else:
            failed_lines.append(lines_to_match[i])
            print "Failed", lines_to_match[i]
    dt.write1dArray(failed_lines, "filmdata/KeywordData/failed_second_match.txt")
    dt.write1dArray(matched_lines, "filmdata/KeywordData/matched_lines_NEW.txt")


"""

Out of a list of movie_string : keyword, get the top_value most common keywords
from that list.

"""

def getMostCommonKeywords(top_value, file_name, keyword_file, value_file):
    common_keywords = []
    file = open(file_name, "r")
    lines = file.readlines()
    keywords = defaultdict(int)
    for line in lines:
        if len(line.split()) > 0:
            line_split = line.split()
            keyword = line_split[len(line_split)-1]
            keywords[keyword] += 1
        print line
    sorted_dict = sorted(keywords.iteritems(), key=lambda x:-x[1])[:top_value]
    print sorted_dict
    keys = []
    values = []
    for key, value in sorted_dict:
        keys.append(key)
        values.append(value)
    dt.write1dArray(keys, keyword_file)
    dt.write1dArray(values, value_file)

#getMostCommonKeywords(3000, "filmdata\KeywordData\Matched_Films.txt", "filmdata/KeywordData/most_common_keywords.txt", "filmdata/KeywordData/most_common_keywords_values.txt")


def makeKeywordPPMIVectors(file_name, common_keywords):
    print "?"
    file = open(file_name, "r")
    lines = file.readlines()
    last_film = ""
    movie_strings = dt.importString("filmdata/filmNames.txt")
    standard_strings = []
    for m in movie_strings:
        m = m[:-5]
        standard_strings.append(m.translate(None, string.punctuation).replace(" ", "").strip().upper())
    for line in lines:
        film_vectors = []
        line = line.strip()
        if len(line) > 2:
            line_split = re.split(r'\t+', line)
            line_split[0] = line_split[0].translate(None, string.punctuation).replace(" ", "").strip().upper()
            file_save = ""
            for m in range(len(standard_strings)):
                if standard_strings[m] == line_split[0]:
                    file_save = str(m)
                    break
            if file_save != "":
                file = open("filmdata\KeywordData\Movie_Most_Common_Keyword_Mapping\\" + file_save, "a")
                for keyword in common_keywords:
                    if line_split[2] == keyword.strip():
                        film_vectors.append("line")
                        file.write(keyword)
                        break
                if last_film.strip() != line_split[0].strip() and last_film is not None:
                    print "Succeeded", line_split[0]
                    file.close()

                last_film = line_split[0]
            else:
                print "Failed", line_split[0]
#makeKeywordPPMIVectors("filmdata\KeywordData\Matched_Films_Norm_Spaces.txt", dt.importString("filmdata/KeywordData/most_common_keywords.txt"))

def findMissingKeywords(file_name, common_keywords):
    print "?"
    file = open(file_name, "r")
    lines = file.readlines()
    last_film = ""
    movie_strings = dt.importString("filmdata/filmNames.txt")
    standard_strings = []
    indexes = []
    for m in movie_strings:
        m = m[:-5]
        standard_strings.append(m.translate(None, string.punctuation).replace(" ", "").strip().upper())
    for line in lines:
        film_vectors = []
        line = line.strip()
        if len(line) > 2:
            line_split = re.split(r'\t+', line)
            line_split[0] = line_split[0].translate(None, string.punctuation).replace(" ", "").strip().upper()
            file_save = ""
            for m in range(len(standard_strings)):
                if standard_strings[m] == line_split[0]:
                    print "matched", m, standard_strings[m], line_split[0]
                    file_save = str(m)
                    break
            if file_save != "":
                if last_film.strip() != line_split[0].strip() and last_film is not None:
                    print "Succeeded", line_split[0]
                    for m in range(len(standard_strings)):
                        if standard_strings[m] == last_film:
                            indexes.append(m)
                            break
                last_film = line_split[0]
            else:
                print "Failed", line_split[0],
    dt.write1dArray(indexes, "filmdata/MISSING_FROM_MOVIEDATA.txt")

#findMissingKeywords("filmdata\KeywordData\Matched_Films_Norm_Spaces.txt", dt.importString("filmdata/KeywordData/most_common_keywords.txt"))

"""

Compare a list of index-files to the list of 15,000 long movies and
write the ones that are missing.

"""


def writeMissing(folder_name):
    print "?"
    file_names = dt.getAllFileNames(folder_name)
    standard = range(15000)
    missing = []
    for i in standard:
        found = False
        for f in file_names:
            if int(f) == int(i):
                found = True
                break
        if found:
            print "found", i
        else:
            missing.append(i)
            print "no found", i
    dt.write1dArray(missing, "filmdata/MISSING_KEYWORD_ITEMS.txt")

#writeMissing("filmdata\KeywordData\Movie_Most_Common_Keyword_Mapping")



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
    id_mappings = open("filmdata/KeywordData/Movie_Most_Common_Keyword_Mapping/films-ids.txt", "r")
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
    dt.write1dArray(failed_names, "filmdata/KeywordData/NAMES_THAT_FAILED_IDS.txt")
    dt.write1dArray(ordered_IDs, "filmdata/KeywordData/IDsByOriginalOrdering.txt")


def getScoreDifferences(name_word_file1, name_score_file1, name_word_file2, name_score_file2, name):
    word_file1 = open(name_word_file1, "r")
    score_file1 = open(name_score_file1, "r")
    word_lines1 = word_file1.readlines()
    score_lines1 = score_file1.readlines()
    scores1 = []
    words1 = []
    for s in score_lines1:
        scores1.append(float(s.strip()))
    for w in word_lines1:
        words1.append(w.strip())
    word_file2 = open(name_word_file2, "r")
    score_file2 = open(name_score_file2, "r")
    word_lines2 = word_file2.readlines()
    score_lines2 = score_file2.readlines()
    scores2 = []
    words2 = []
    for s in score_lines2:
        scores2.append(float(s))
    for w in word_lines2:
        words2.append(w.strip())
    differences_list = []
    for i in range(len(score_lines1)):
        differences_list.append(scores1[i] - scores2[i])
    most_different_words = [x for (y,x) in sorted(zip(differences_list,words1))]
    differences_list = sorted(differences_list)
    dt.write1dArray(most_different_words, "filmdata/SVM/most_different_words_" + name + ".txt")
    dt.write1dArray(differences_list, "filmdata/SVM/most_different_values_" + name + ".txt")


"""

getScoreDifferences("filmdata/KeywordData/SVM/ALL_NAMES_NewKeywords_BCsgdAllAutoencoder-200-50xON200-200Hidden1.txt", "filmdata/KeywordData/SVM/ALL_SCORES_NewKeywords_BCsgdAllAutoencoder-200-50xON200-200Hidden1.txt",
                    "filmdata/KeywordData/SVM/ALL_NAMES_NewKeywords_normal200.txt", "filmdata/KeywordData/SVM/ALL_SCORES_NewKeywords_normal200.txt",
                    "200-150ae200-200nnL1toNormal200")


getScoreDifferences("filmdata/SVM/ALL_NAMES_Phrases_BCsgdAllAutoencoder-200-50xON200-200Hidden1.txt", "filmdata/SVM/ALL_SCORES_Phrases_BCsgdAllAutoencoder-200-50xON200-200Hidden1.txt",
                    "filmdata/SVM/ALL_NAMES_Phrases_normal200.txt", "filmdata/SVM/ALL_SCORES_Phrases_normal200.txt",
                    "BCsgdAllAutoencoder-200-50xON200-200toNormal200")

"""

def makeSpaceNoisy(input, amount_of_corruption):
    amount_of_corruption = len(input[0]) * amount_of_corruption
    print amount_of_corruption
    for x in range(int(amount_of_corruption)):
        for i in input:
            r = random.randint(0, len(i)-1)
            while i[r] == 0:
                r = random.randint(0, len(i)-1)
            i[r] = 0
    return input
"""
input_size = 200
dt.write2dArray(makeSpaceNoisy(dt.getMovieVectors(input_size=input_size), 0.25), "newdata/spaces/noise"+str(input_size)+".mds")
"""
"""
Given a list of ordered IDs for the 15,0000 films.
And get all of the phrases mentioned in those IDs and return that list.

Then, given a list of phrases mentioned, create a vector for every movie and return
as an array of those vectors. Information is found within the film files.

"""

def getUniquePhrases(ordered_IDs, count_min, maintain_quantities):
    phrase_to_count = {}
    for ID in ordered_IDs:
        ID = ID.strip()
        if ID != "-1":
            open_ID = open("filmdata/Tokens/" + ID + ".film", "r")
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
    multi_dictionary = {}
    dict_mapping = {}
    print "Mapping to memory."
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
    for up in range(len(unique_phrases)):
        unique_phrases[up] = unique_phrases[up].strip()

    print len("Iterating over memory.")
    for p in range(13177, 25842, 1):
        for key, value in multi_dictionary.iteritems():
            if key[1] == unique_phrases[p]:
                vectors_maintained[p][dict_mapping[key[0].strip()]] = value
                vectors[p][dict_mapping[key[0]]] = 1
        print unique_phrases[p]
        dt.write1dArray(vectors_maintained[p], "filmdata/classesPhrases/nonbinary/class-" + unique_phrases[p])
        dt.write1dArray(vectors[p], "filmdata/classesPhrases/class-" + unique_phrases[p])

    return vectors_maintained, vectors


def getVectorsKeywords(movie_strings, keywords):
    multi_dictionary = {}
    dict_mapping = {}
    movie_names = []
    file_names = dt.getAllFileNames("filmdata\KeywordData\Movie_Most_Common_Keyword_Mapping")
    for i in movie_strings:
        movie_names.append(i.strip()[:-5])
        print i
    print "Mapping to memory."
    for i in file_names:
        try:
            file = open("filmdata/KeywordData/Movie_Most_Common_Keyword_Mapping/" +i, "r")
            lines = file.readlines()
            dict_mapping[movie_strings[int(i)]] = i
            for line in lines:
                line = line.strip()
                multi_dictionary[(movie_strings[int(i)], line)] = 1
            file.close()
        except IOError:
            print movie_names[i]

    for up in range(len(keywords)):
        keywords[up] = keywords[up].strip()

    print len("Iterating over memory.")
    for p in range(len(keywords)):
        vector = [0 for x in range(len(movie_strings))]
        print len(vector)
        for key, value in multi_dictionary.iteritems():
            if key[1] == keywords[p]:
                #print int(dict_mapping[key[0]])
                vector[int(dict_mapping[key[0]])] = 1
        print keywords[p]
        dt.write1dArray(vector, "filmdata/classesKeywords/NewData/class-" + keywords[p])


#getVectorsKeywords(dt.importString("filmdata/filmNames.txt"), dt.importString("filmdata/KeywordData/most_common_keywords.txt"))
#dt.writeAllKeywordsToSingleFile("filmdata/classesNewKeywords")

def getMissingIndexes(index_list, length):
    full_index = range(length)
    for i in index_list:
        i = int(i)
        full_index[i] = -1
    missing_indexes = []
    for i in full_index:
        if i > -1:
            missing_indexes.append(i)
    dt.write1dArray(missing_indexes, "filmdata/missing_indexes_keywords.txt")

#getMissingIndexes(dt.importString("filmdata/MISSING_FROM_MOVIEDATA.txt"), 15000)

def getVectorsIO(ordered_IDs, unique_phrases):
    vectors = [[0 for x in range(len(ordered_IDs))] for x in range(len(unique_phrases))]
    vectors_maintained = [[0 for x in range(len(ordered_IDs))] for x in range(len(unique_phrases))]

    for p in range(11212, 25842, 1):
        unique_phrases[p] = unique_phrases[p].strip()
        for i in range(len(ordered_IDs)):
            ordered_IDs[i] = ordered_IDs[i].strip()
            if ordered_IDs[i] != "-1":
                file = open("filmdata/Tokens/" + ordered_IDs[i] + ".film", "r")
                lines = file.readlines()[1:]
                for line in lines:
                    split_line = line.split()
                    split_line[1] = split_line[1].strip()
                    if split_line[0] == p:
                        vectors_maintained[p][i] = split_line[1]
                        vectors[p][i] = 1
                file.close()
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
    commonality = 0
    common_keywords = getMostCommonKeywords(0, "filmdata/IMDB_movie_data.txt")
    dt.write1dArray(common_keywords, "filmdata/common_keywords_15k_commanility_" + str(commonality))
    vectors = getKeywordVectors(common_keywords, movie_strings, "")
    dt.write2dArray(vectors, "filmdata/classesKeywords/class-extra-all-commonality-" + str(commonality))

#makeConsistent("filmdata/KeywordData/Matched_Films.txt", "filmdata/KeywordData/Matched_Films_Normalised.txt")

#getMatchedLines("filmdata/KeywordData/All_Films_Norm_Spaces.txt", dt.importString("filmdata/KeywordData/Missing_Films_Normalised.txt"))
"""

movie_strings = dt.importString("filmdata/filmNames.txt")
missing_items = getMissing("filmdata/IMDB Keywords Movie Data/Matched_Films.txt", movie_strings)
dt.write1dArray(missing_items, "filmdata/missing_films.txt")

"""
#outputPhrases()
#outputKeywords()
