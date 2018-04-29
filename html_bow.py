from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from urllib.request import urlopen 
import re
import os
import numpy as np
from random import shuffle 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import sent_tokenize
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.corpus import stopwords
from nltk.tag.stanford import StanfordNERTagger
from sklearn.decomposition import TruncatedSVD
 

def term_frequency_matrix(documents, terms):
    # Module to create the term frequency matrix
    
    td_matrix = []
    for itr in documents:
        doc_terms = [itr[0].count(t) for t in terms]
        td_matrix.append(doc_terms)

    return np.array(td_matrix)


def main():

    #Main module to call subroutines

    text_container = [] # for storing the entire string of a webpage 
    unique_words = []   # stores the number of unique words in all the samples
    path = '/home/raja/Raja/Sem2/SMAI/Project/course-cotrain-data/fulltext/course/'
    stemmer = PorterStemmer()  # used for stemming
    tokenizer = RegexpTokenizer(r'\w+')  # for Regular expression 
    class_label = []   # holds the class labels
    for filename in os.listdir(path):
        filename = 'file:///home/raja/Raja/Sem2/SMAI/Project/course-cotrain-data/fulltext/course/' + filename
        sock = urlopen(filename) 
        htmlSource = sock.read() 
        htmlSource = htmlSource.decode("windows-1252")  # utf-8 could be used instead of "windows-1252"                           
        sock.close()
        class_label.append(0)  # appends the class label  

        # for obtaining text inside <> tags                     
        cleanr = re.compile('<.*?>')
        htmlSource = re.sub(cleanr, '', htmlSource)

        word_tokens = tokenizer.tokenize(htmlSource.lower())   # Changes to lower case
        word_list = [stemmer.stem(line) for line in word_tokens if line not in '']  # stemming is being done
        stop_words = set(stopwords.words('english'))  # for stop word removal
        word_tokens = [w for w in word_list if not w in stop_words]
        unique_words += list(set(word_tokens))
        unique_words = list(set(unique_words))  # updates unique word list
    

        dummy_str = ""
        for i in word_tokens:
            dummy_str += i + " "

        dummy_list = [dummy_str]
        text_container.append(dummy_list)  # Appends the entire text of a webpage into text_container

    class_one_samples_count = len(class_label)
    path = '/home/raja/Raja/Sem2/SMAI/Project/course-cotrain-data/fulltext/non-course/'
    for filename in os.listdir(path):
        filename = 'file:///home/raja/Raja/Sem2/SMAI/Project/course-cotrain-data/fulltext/non-course/' + filename
        sock = urlopen(filename) 
        htmlSource = sock.read() 
        htmlSource = htmlSource.decode("windows-1252")                           
        sock.close()    
        class_label.append(1)

        # for obtaining text inside <> tags                     
        cleanr = re.compile('<.*?>')
        htmlSource = re.sub(cleanr, '', htmlSource)

        # basic preprocessing
        word_tokens = word_tokenize(htmlSource)
        word_list = [stemmer.stem(line) for line in word_tokens if line not in '']
        stop_words = set(stopwords.words('english'))
        word_tokens = [w for w in word_list if not w in stop_words]
        unique_words += list(set(word_tokens))
        unique_words = list(set(unique_words))

        dummy_str = ""
        for i in word_tokens:
            dummy_str += i + " "

        dummy_list = [dummy_str]
        text_container.append(dummy_list)

    class_two_samples_count = len(class_label) - class_one_samples_count
    class_label = np.asarray(class_label)
    class_label = class_label.reshape(class_label.shape[0], 1)
    # print(class_label.shape)

    tf_matrix = term_frequency_matrix(text_container, unique_words)    
    tf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    tf_idf_matrix = tf.fit_transform(tf_matrix).todense() 
    
    # size = int(tf_idf_matrix.shape[0] * 0.7)
    # print(tf_idf_matrix)
    svd = TruncatedSVD(n_components=1050, random_state=42)
    tf_idf_matrix_SVD = svd.fit_transform(tf_idf_matrix)


    tf_idf_matrix_with_labels = np.concatenate((tf_idf_matrix_SVD, class_label), axis=1)
    class_one_test_samples = (class_one_samples_count  * 0.3)
    test_tf_idf_matrix = tf_idf_matrix_with_labels[0:class_one_test_samples,:]
    class_two_test_samples = (class_two_samples_count  * 0.3)
    temp_test_tf_idf_matrix = tf_idf_matrix_with_labels[class_one_samples_count:(class_one_samples_count+class_two_samples_count),:]
    test_tf_idf_matrix = np.concatenate((test_tf_idf_matrix,temp_test_tf_idf_matrix),axis = 0)
    temp_matrix1 = tf_idf_matrix_with_labels[class_one_test_samples:class_one_samples_count,:]
    temp_matrix2 = tf_idf_matrix_with_labels[(class_one_samples_count+class_two_samples_count):,:]
    train_tf_idf_matrix = np.concatenate((temp_matrix1, temp_matrix2), axis=0)

    # np.random.shuffle(tf_idf_matrix)

    # test_tf_idf_matrix = tf_idf_matrix_with_labels[size:, :]
    # train_tf_idf_matrix = tf_idf_matrix_with_labels[:size, :]

    fp = open('tfidf_matrix_fulltext_train.txt', 'w')
    for i in range(train_tf_idf_matrix.shape[0]):
        for j in range(train_tf_idf_matrix.shape[1]):
            fp.write(str(train_tf_idf_matrix[i][j]) + " ")
            
        fp.write("\n")    
    fp.close()

    fp = open('tfidf_matrix_fulltext_test.txt', 'w')
    for i in range(test_tf_idf_matrix.shape[0]):
        for j in range(test_tf_idf_matrix.shape[1]):
            fp.write(str(test_tf_idf_matrix[i][j]) + " ")

        fp.write("\n")
    fp.close()






    svd = TruncatedSVD(n_components=50, random_state=42)
    tf_idf_matrix_SVD = svd.fit_transform(tf_idf_matrix)


    tf_idf_matrix_with_labels = np.concatenate((tf_idf_matrix_SVD, class_label), axis=1)
    class_one_test_samples = (class_one_samples_count  * 0.3)
    test_tf_idf_matrix = tf_idf_matrix_with_labels[0:class_one_test_samples,:]
    class_two_test_samples = (class_two_samples_count  * 0.3)
    temp_test_tf_idf_matrix = tf_idf_matrix_with_labels[class_one_samples_count:(class_one_samples_count+class_two_samples_count),:]
    test_tf_idf_matrix = np.concatenate((test_tf_idf_matrix,temp_test_tf_idf_matrix),axis = 0)
    temp_matrix1 = tf_idf_matrix_with_labels[class_one_test_samples:class_one_samples_count,:]
    temp_matrix2 = tf_idf_matrix_with_labels[(class_one_samples_count+class_two_samples_count):,:]
    train_tf_idf_matrix = np.concatenate((temp_matrix1, temp_matrix2), axis=0)

    # np.random.shuffle(tf_idf_matrix)

    # test_tf_idf_matrix = tf_idf_matrix_with_labels[size:, :]
    # train_tf_idf_matrix = tf_idf_matrix_with_labels[:size, :]

    fp = open('tfidf_matrix_fulltext_train_small.txt', 'w')
    for i in range(train_tf_idf_matrix.shape[0]):
        for j in range(train_tf_idf_matrix.shape[1]):
            fp.write(str(train_tf_idf_matrix[i][j]) + " ")
            
        fp.write("\n")    
    fp.close()

    fp = open('tfidf_matrix_fulltext_test_small.txt', 'w')
    for i in range(test_tf_idf_matrix.shape[0]):
        for j in range(test_tf_idf_matrix.shape[1]):
            fp.write(str(test_tf_idf_matrix[i][j]) + " ")

        fp.write("\n")
    fp.close()








    # for inlinks view
    text_container = []
    unique_words = []
    path = '/home/raja/Raja/Sem2/SMAI/Project/course-cotrain-data/inlinks/course/'
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    class_label = []
    for filename in os.listdir(path):
        filename = 'file:///home/raja/Raja/Sem2/SMAI/Project/course-cotrain-data/inlinks/course/' + filename
        sock = urlopen(filename)
        htmlSource = sock.read()
        htmlSource = htmlSource.decode("windows-1252")
        sock.close()
        class_label.append(0)

        # for obtaining text inside <> tags
        cleanr = re.compile('<.*?>')
        htmlSource = re.sub(cleanr, '', htmlSource)

        word_tokens = tokenizer.tokenize(htmlSource.lower())
        word_list = [stemmer.stem(line)
                     for line in word_tokens if line not in '']
        stop_words = set(stopwords.words('english'))
        word_tokens = [w for w in word_list if not w in stop_words]
        unique_words += list(set(word_tokens))
        unique_words = list(set(unique_words))

        dummy_str = ""
        for i in word_tokens:
            dummy_str += i + " "

        dummy_list = [dummy_str]
        text_container.append(dummy_list)

    # print(unique_words)

    path = '/home/raja/Raja/Sem2/SMAI/Project/course-cotrain-data/inlinks/non-course/'
    for filename in os.listdir(path):
        filename = 'file:///home/raja/Raja/Sem2/SMAI/Project/course-cotrain-data/inlinks/non-course/' + filename
        sock = urlopen(filename)
        htmlSource = sock.read()
        htmlSource = htmlSource.decode("windows-1252")
        sock.close()
        class_label.append(1)

        # for obtaining text inside <> tags
        cleanr = re.compile('<.*?>')
        htmlSource = re.sub(cleanr, '', htmlSource)

        # basic preprocessing
        word_tokens = word_tokenize(htmlSource)
        word_list = [stemmer.stem(line)
                     for line in word_tokens if line not in '']
        stop_words = set(stopwords.words('english'))
        word_tokens = [w for w in word_list if not w in stop_words]
        unique_words += list(set(word_tokens))
        unique_words = list(set(unique_words))

        dummy_str = ""
        for i in word_tokens:
            dummy_str += i + " "

        dummy_list = [dummy_str]
        text_container.append(dummy_list)

    class_label = np.asarray(class_label)
    class_label = class_label.reshape(class_label.shape[0], 1)
    # print(class_label.shape)

    tf_matrix = term_frequency_matrix(text_container, unique_words)
    tf = TfidfTransformer(norm='l2', use_idf=True,smooth_idf=True, sublinear_tf=False)
    tf_idf_matrix = tf.fit_transform(tf_matrix).todense()

    size = int(tf_idf_matrix.shape[0] * 0.7)
    # print(tf_idf_matrix)

    svd = TruncatedSVD(n_components=1050, random_state=42)
    tf_idf_matrix_SVD = svd.fit_transform(tf_idf_matrix)

    tf_idf_matrix_with_labels = np.concatenate((tf_idf_matrix_SVD, class_label), axis=1)
    class_one_test_samples = (class_one_samples_count  * 0.3)
    test_tf_idf_matrix = tf_idf_matrix_with_labels[0:class_one_test_samples,:]
    class_two_test_samples = (class_two_samples_count  * 0.3)
    temp_test_tf_idf_matrix = tf_idf_matrix_with_labels[class_one_samples_count:(class_one_samples_count+class_two_samples_count),:]
    test_tf_idf_matrix = np.concatenate((test_tf_idf_matrix,temp_test_tf_idf_matrix),axis = 0)
    temp_matrix1 = tf_idf_matrix_with_labels[class_one_test_samples:class_one_samples_count,:]
    temp_matrix2 = tf_idf_matrix_with_labels[(class_one_samples_count+class_two_samples_count):,:]
    train_tf_idf_matrix = np.concatenate((temp_matrix1, temp_matrix2), axis=0)

    fp = open('tfidf_matrix_inlinks_train.txt', 'w')
    for i in range(train_tf_idf_matrix.shape[0]):
        for j in range(train_tf_idf_matrix.shape[1]):
            fp.write(str(train_tf_idf_matrix[i][j]) + " ")

        fp.write("\n")
    fp.close()

    fp = open('tfidf_matrix_inlinks_test.txt', 'w')
    for i in range(test_tf_idf_matrix.shape[0]):
        for j in range(test_tf_idf_matrix.shape[1]):
            fp.write(str(test_tf_idf_matrix[i][j]) + " ")

        fp.write("\n")
    fp.close()

    


    svd = TruncatedSVD(n_components=50, random_state=42)
    tf_idf_matrix_SVD = svd.fit_transform(tf_idf_matrix)


    tf_idf_matrix_with_labels = np.concatenate((tf_idf_matrix_SVD, class_label), axis=1)
    class_one_test_samples = (class_one_samples_count  * 0.3)
    test_tf_idf_matrix = tf_idf_matrix_with_labels[0:class_one_test_samples,:]
    class_two_test_samples = (class_two_samples_count  * 0.3)
    temp_test_tf_idf_matrix = tf_idf_matrix_with_labels[class_one_samples_count:(class_one_samples_count+class_two_samples_count),:]
    test_tf_idf_matrix = np.concatenate((test_tf_idf_matrix,temp_test_tf_idf_matrix),axis = 0)
    temp_matrix1 = tf_idf_matrix_with_labels[class_one_test_samples:class_one_samples_count,:]
    temp_matrix2 = tf_idf_matrix_with_labels[(class_one_samples_count+class_two_samples_count):,:]
    train_tf_idf_matrix = np.concatenate((temp_matrix1, temp_matrix2), axis=0)

    # np.random.shuffle(tf_idf_matrix)

    # test_tf_idf_matrix = tf_idf_matrix_with_labels[size:, :]
    # train_tf_idf_matrix = tf_idf_matrix_with_labels[:size, :]

    fp = open('tfidf_matrix_inlinks_train_small.txt', 'w')
    for i in range(train_tf_idf_matrix.shape[0]):
        for j in range(train_tf_idf_matrix.shape[1]):
            fp.write(str(train_tf_idf_matrix[i][j]) + " ")
            
        fp.write("\n")    
    fp.close()

    fp = open('tfidf_matrix_inlinks_test_small.txt', 'w')
    for i in range(test_tf_idf_matrix.shape[0]):
        for j in range(test_tf_idf_matrix.shape[1]):
            fp.write(str(test_tf_idf_matrix[i][j]) + " ")

        fp.write("\n")
    fp.close()
main()

