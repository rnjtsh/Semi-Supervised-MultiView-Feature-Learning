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


def kfold_tfidf_creation_module(root_pathname1, v1_pathname, root_pathname2, v2_pathname, principal_components, train_file_name, test_file_name):

    # module to create kfold suite of train and test tfidf matrices and writes into file

    text_container = [] # for storing the entire string of a webpage 
    unique_words = []   # stores the number of unique words in all the samples
    path = root_pathname1   #'/home/sanjoy/Desktop/course-cotrain-data/fulltext/course/'
    stemmer = PorterStemmer()  # used for stemming
    tokenizer = RegexpTokenizer(r'\w+')  # for Regular expression 
    class_label = []   # holds the class labels
    for filename in os.listdir(path):
        filename = v1_pathname + filename
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

    class_one_samples_count = len(class_label)  # stores count of class one samples
    path = root_pathname2
    for filename in os.listdir(path):
        # for every sample reads the data
        print(filename)
        filename = v2_pathname + filename
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
        #stop_words = set(stopwords.words('english'))
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

    tf_matrix = term_frequency_matrix(text_container, unique_words)    
    tf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    tf_idf_matrix = tf.fit_transform(tf_matrix).todense() 
    
    #Performs PCA to get top k principal components
    svd = TruncatedSVD(n_components=principal_components, random_state=42)
    tf_idf_matrix_SVD = svd.fit_transform(tf_idf_matrix)

    tf_idf_matrix_with_labels = np.concatenate((tf_idf_matrix_SVD, class_label), axis=1)
    class_one_test_samples = int(class_one_samples_count  * 0.3)  # Takes out 30% of samples for testing
    test_tf_idf_matrix = tf_idf_matrix_with_labels[0:class_one_test_samples,:]
    class_two_test_samples = int(class_two_samples_count  * 0.3)  # Takes out 30% of samples for testing
    temp_test_tf_idf_matrix = tf_idf_matrix_with_labels[class_one_samples_count:(class_one_samples_count+class_two_test_samples),:]
    test_tf_idf_matrix = np.concatenate((test_tf_idf_matrix,temp_test_tf_idf_matrix),axis = 0)
    temp_matrix1 = tf_idf_matrix_with_labels[class_one_test_samples:class_one_samples_count,:]
    temp_matrix2 = tf_idf_matrix_with_labels[(class_one_samples_count+class_two_test_samples):,:]
    train_tf_idf_matrix = np.concatenate((temp_matrix1, temp_matrix2), axis=0)

    # writes training tfidf into file
    fp = open(train_file_name, 'w')
    for i in range(train_tf_idf_matrix.shape[0]):
        for j in range(train_tf_idf_matrix.shape[1]):
            fp.write(str(train_tf_idf_matrix[i][j]) + " ")
            
        fp.write("\n")    
    fp.close()

    # writes test tfidf into file
    fp = open(test_file_name, 'w')
    for i in range(test_tf_idf_matrix.shape[0]):
        for j in range(test_tf_idf_matrix.shape[1]):
            fp.write(str(test_tf_idf_matrix[i][j]) + " ")

        fp.write("\n")
    fp.close()




def tfidf_creation_module(root_pathname1, v1_pathname, root_pathname2, v2_pathname, principal_components, train_file_name, test_file_name):
    
    # module to create train and test tfidf matrices and writes into file

    text_container = [] # for storing the entire string of a webpage 
    unique_words = []   # stores the number of unique words in all the samples
    path = root_pathname1   #'/home/sanjoy/Desktop/course-cotrain-data/fulltext/course/'
    stemmer = PorterStemmer()  # used for stemming
    tokenizer = RegexpTokenizer(r'\w+')  # for Regular expression 
    class_label = []   # holds the class labels
    for filename in os.listdir(path):
        filename = v1_pathname + filename
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

    class_one_samples_count = len(class_label)  # stores count of class one samples
    path = root_pathname2
    for filename in os.listdir(path):
        # for every sample reads the data
        print(filename)
        filename = v2_pathname + filename
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
        #stop_words = set(stopwords.words('english'))
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

    tf_matrix = term_frequency_matrix(text_container, unique_words)    
    tf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    tf_idf_matrix = tf.fit_transform(tf_matrix).todense() 
    
    #Performs PCA to get top k principal components
    svd = TruncatedSVD(n_components=principal_components, random_state=42)
    tf_idf_matrix_SVD = svd.fit_transform(tf_idf_matrix)
    tf_idf_matrix_with_labels = np.concatenate((tf_idf_matrix_SVD, class_label), axis=1)
    
    # train test splitting for first fold
    class_one_test_samples = int(class_one_samples_count  * 0.3)  # Takes out 30% of samples for testing
    test_tf_idf_matrix = tf_idf_matrix_with_labels[0:class_one_test_samples,:]
    class_two_test_samples = int(class_two_samples_count  * 0.3)  # Takes out 30% of samples for testing
    temp_test_tf_idf_matrix = tf_idf_matrix_with_labels[class_one_samples_count:(class_one_samples_count+class_two_test_samples),:]
    test_tf_idf_matrix = np.concatenate((test_tf_idf_matrix,temp_test_tf_idf_matrix),axis = 0)
    # two temporary matrices used to get a combined train matrix
    temp_matrix1 = tf_idf_matrix_with_labels[class_one_test_samples:class_one_samples_count,:]
    temp_matrix2 = tf_idf_matrix_with_labels[(class_one_samples_count+class_two_test_samples):,:]
    train_tf_idf_matrix = np.concatenate((temp_matrix1, temp_matrix2), axis=0)

    # writes training tfidf into file
    temp_test_file_name = train_file_name[:-4] + '_fold1.txt'
    fp = open(train_file_name, 'w')
    for i in range(train_tf_idf_matrix.shape[0]):
        for j in range(train_tf_idf_matrix.shape[1]):
            fp.write(str(train_tf_idf_matrix[i][j]) + " ")
            
        fp.write("\n")    
    fp.close()

    # writes test tfidf into file
    temp_test_file_name = test_file_name[:-4] + '_fold1.txt'
    fp = open(test_file_name, 'w')
    for i in range(test_tf_idf_matrix.shape[0]):
        for j in range(test_tf_idf_matrix.shape[1]):
            fp.write(str(test_tf_idf_matrix[i][j]) + " ")

        fp.write("\n")
    fp.close()



    # train test splitting for second fold
    test_tf_idf_matrix = tf_idf_matrix_with_labels[class_one_test_samples:(2*class_one_test_samples),:]
    temp_test_tf_idf_matrix = tf_idf_matrix_with_labels[(class_one_samples_count+class_one_test_samples):(class_one_samples_count+(2*class_one_test_samples)),:]
    test_tf_idf_matrix = np.concatenate((test_tf_idf_matrix,temp_test_tf_idf_matrix),axis = 0)
    temp_matrix1 = tf_idf_matrix_with_labels[0:class_one_test_samples,:]
    t1 = tf_idf_matrix_with_labels[(2*class_one_test_samples):(class_one_samples_count+class_two_test_samples),:]
    temp_matrix2 = tf_idf_matrix_with_labels[(class_one_samples_count+(2*class_two_test_samples)):,:]
    train_tf_idf_matrix = np.concatenate((temp_matrix1, t1), axis=0)
    train_tf_idf_matrix = np.concatenate((train_tf_idf_matrix, temp_matrix2), axis=0)

    # writes training tfidf into file
    temp_test_file_name = train_file_name[:-4] + '_fold2.txt'
    fp = open(train_file_name, 'w')
    for i in range(train_tf_idf_matrix.shape[0]):
        for j in range(train_tf_idf_matrix.shape[1]):
            fp.write(str(train_tf_idf_matrix[i][j]) + " ")
            
        fp.write("\n")    
    fp.close()

    # writes test tfidf into file
    temp_test_file_name = test_file_name[:-4] + '_fold2.txt'
    fp = open(test_file_name, 'w')
    for i in range(test_tf_idf_matrix.shape[0]):
        for j in range(test_tf_idf_matrix.shape[1]):
            fp.write(str(test_tf_idf_matrix[i][j]) + " ")

        fp.write("\n")
    fp.close()




    # train test splitting for third fold
    test_tf_idf_matrix = tf_idf_matrix_with_labels[(2*class_one_test_samples):class_one_samples_count,:]
    temp_test_tf_idf_matrix = tf_idf_matrix_with_labels[(class_one_samples_count+(2*class_one_test_samples)):,:]
    test_tf_idf_matrix = np.concatenate((test_tf_idf_matrix,temp_test_tf_idf_matrix),axis = 0)
    
    temp_matrix1 = tf_idf_matrix_with_labels[0:(2*class_one_test_samples),:]
    temp_matrix2 = tf_idf_matrix_with_labels[class_one_samples_count:(class_one_samples_count+(2*class_two_test_samples)),:]
    train_tf_idf_matrix = np.concatenate((temp_matrix1, temp_matrix2), axis=0)

    # writes training tfidf into file
    temp_test_file_name = train_file_name[:-4] + '_fold3.txt'
    fp = open(train_file_name, 'w')
    for i in range(train_tf_idf_matrix.shape[0]):
        for j in range(train_tf_idf_matrix.shape[1]):
            fp.write(str(train_tf_idf_matrix[i][j]) + " ")
            
        fp.write("\n")    
    fp.close()

    # writes test tfidf into file
    temp_test_file_name = test_file_name[:-4] + '_fold3.txt'
    fp = open(test_file_name, 'w')
    for i in range(test_tf_idf_matrix.shape[0]):
        for j in range(test_tf_idf_matrix.shape[1]):
            fp.write(str(test_tf_idf_matrix[i][j]) + " ")

        fp.write("\n")
    fp.close()



def main():

    #Main module to call subroutines

    # for 1050 principal components
    # root_name1 = '/home/sanjoy/Desktop/course-cotrain-data/fulltext/course/'
    # view_name1 = 'file:///home/sanjoy/Desktop/course-cotrain-data/fulltext/course/'
    # root_name2 = '/home/sanjoy/Desktop/course-cotrain-data/fulltext/non-course/'
    # view_name2 = 'file:///home/sanjoy/Desktop/course-cotrain-data/fulltext/non-course/'
    # tfidf_creation_module(root_name1,view_name1,root_name2,view_name2, 1050, 'tfidf_matrix_fulltext_train_large.txt', 'tfidf_matrix_fulltext_test_large.txt')

    # root_name1 = '/home/sanjoy/Desktop/course-cotrain-data/inlinks/course/'
    # view_name1 = 'file:///home/sanjoy/Desktop/course-cotrain-data/inlinks/course/'
    # root_name2 = '/home/sanjoy/Desktop/course-cotrain-data/inlinks/non-course/'
    # view_name2 = 'file:///home/sanjoy/Desktop/course-cotrain-data/inlinks/non-course/'
    # tfidf_creation_module(root_name1,view_name1,root_name2,view_name2, 1050, 'tfidf_matrix_inlinks_train_large.txt', 'tfidf_matrix_inlinks_test_large.txt')

    # for 100 principal components
    root_name1 = '/home/sanjoy/Desktop/course-cotrain-data/fulltext/course/'
    view_name1 = 'file:///home/sanjoy/Desktop/course-cotrain-data/fulltext/course/'
    root_name2 = '/home/sanjoy/Desktop/course-cotrain-data/fulltext/non-course/'
    view_name2 = 'file:///home/sanjoy/Desktop/course-cotrain-data/fulltext/non-course/'
    tfidf_creation_module(root_name1,view_name1,root_name2,view_name2, 100, 'tfidf_matrix_fulltext_train_small.txt', 'tfidf_matrix_fulltext_test_small.txt')

    root_name1 = '/home/sanjoy/Desktop/course-cotrain-data/inlinks/course/'
    view_name1 = 'file:///home/sanjoy/Desktop/course-cotrain-data/inlinks/course/'
    root_name2 = '/home/sanjoy/Desktop/course-cotrain-data/inlinks/non-course/'
    view_name2 = 'file:///home/sanjoy/Desktop/course-cotrain-data/inlinks/non-course/'
    tfidf_creation_module(root_name1,view_name1,root_name2,view_name2, 100, 'tfidf_matrix_inlinks_train_small.txt', 'tfidf_matrix_inlinks_test_small.txt')

main()

