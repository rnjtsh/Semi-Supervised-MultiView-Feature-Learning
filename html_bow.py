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


# def extract_phone_numbers(string):
#     r = re.compile(
#         r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
#     phone_numbers = r.findall(string)
#     return [re.sub(r'\D', '', number) for number in phone_numbers]


# def extract_email_addresses(string):
#     r = re.compile(r'[\w\.-]+@[\w\.-]+')
#     return r.findall(string)


# def ie_preprocess(document):
#     stop = stopwords.words('english')
#     document = ' '.join([i for i in document.split() if i not in stop])
#     sentences = sent_tokenize(document)
#     sentences = [word_tokenize(sent) for sent in sentences]
#     sentences = [pos_tag(sent) for sent in sentences]
#     return sentences


# def extract_names(document):
#     names = []
#     sentences = ie_preprocess(document)
#     for tagged_sentence in sentences:
#         for chunk in ne_chunk(tagged_sentence):
#             if type(chunk) == Tree:
#                 if chunk.label() == 'PERSON':
#                     names.append(' '.join([c[0] for c in chunk]))
#     return names



def main():

    # url_link = "file:///home/sanjoy/Desktop/course-cotrain-data/fulltext/course/http:%5E%5Ecs.cornell.edu%5EInfo%5ECourses%5ECurrent%5ECS415%5ECS414.html"                                    
    
    text_container = []
    unique_words = []
    path = '/home/raja/Raja/Sem2/SMAI/Project/course-cotrain-data/fulltext/course/'
    stemmer = PorterStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    class_label = []
    for filename in os.listdir(path):
        filename = 'file:///home/raja/Raja/Sem2/SMAI/Project/course-cotrain-data/fulltext/course/' + filename
        sock = urlopen(filename) 
        htmlSource = sock.read() 
        htmlSource = htmlSource.decode("windows-1252")                           
        sock.close()
        class_label.append(0)    

        # for obtaining text inside <> tags                     
        cleanr = re.compile('<.*?>')
        htmlSource = re.sub(cleanr, '', htmlSource)

        # basic preprocessing
        # word_tokens = word_tokenize(htmlSource)
        # numbers = extract_phone_numbers(htmlSource)
        # emails = extract_email_addresses(htmlSource)
        # names = extract_names(htmlSource)

        # text_content = ""
        # for i in numbers:
        #     if(i not in numbers):
        #         text_content += i
        # for i in emails:
        #     if(i not in emails):
        #         text_content += i
        # for i in names:
        #     if(i not in names):
        #         text_content += i
        # htmlSource = text_content

        # print (ne_chunk(pos_tag(word_tokenize(htmlSource))))
        word_tokens = tokenizer.tokenize(htmlSource.lower())
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
    
    # print(unique_words)

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

    class_label = np.asarray(class_label)
    class_label = class_label.reshape(class_label.shape[0], 1)
    # print(class_label.shape)

    tf_matrix = term_frequency_matrix(text_container, unique_words)    
    tf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    tf_idf_matrix = tf.fit_transform(tf_matrix).todense() 

    b = np.zeros((tf_idf_matrix.shape[0], tf_idf_matrix.shape[1]+1))
    b[:, :-1] = tf_idf_matrix
    for i in range(class_label.shape[0]):
        b[i][-1] = class_label[i][0]
    
    tf_idf_matrix = b
    np.random.shuffle(tf_idf_matrix)
    size = int(tf_idf_matrix.shape[0] * 0.7)
    train_tfidf_matrix = tf_idf_matrix[0:size,:]
    print(train_tfidf_matrix)
    test_tfidf_matrix = tf_idf_matrix[size:, :]
    svd = TruncatedSVD(n_components=1050, random_state=42)
    train_tfidf_matrix = svd.fit_transform(train_tfidf_matrix[:,:-1])
    # print(train_tfidf_matrix.shape)
    # print(train_tfidf_matrix)
    fp = open('tfidf_matrix_fulltext.txt', 'w')
    for i in range(train_tfidf_matrix.shape[0]):
        for j in range(train_tfidf_matrix.shape[1]):
            fp.write(str(train_tfidf_matrix[i][j]) + " ")
            
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
    tf_matrix = term_frequency_matrix(text_container, unique_words)
    class_label = np.asarray(class_label)
    class_label = class_label.reshape(class_label.shape[0], 1)
    tf = TfidfTransformer(norm='l2', use_idf=True,smooth_idf=True, sublinear_tf=False)
    tf_idf_matrix = tf.fit_transform(tf_matrix).todense()

    b = np.zeros((tf_idf_matrix.shape[0], tf_idf_matrix.shape[1]+1))
    b[:, :-1] = tf_idf_matrix
    for i in range(class_label.shape[0]):
        b[i][-1] = class_label[i][0]
    tf_idf_matrix = b
    np.random.shuffle(tf_idf_matrix)
    size = int(tf_idf_matrix.shape[0] * 0.7)
    train_tfidf_matrix = tf_idf_matrix[0:size, :]
    test_tfidf_matrix = tf_idf_matrix[size:, :]
    svd = TruncatedSVD(n_components=1050, random_state=42)
    train_tfidf_matrix = svd.fit_transform(train_tfidf_matrix[:,:-1])
    # print(train_tfidf_matrix.shape)
    # print(train_tfidf_matrix)
    fp = open('tfidf_matrix_inlinks.txt', 'w')
    for i in range(train_tfidf_matrix.shape[0]):
        for j in range(train_tfidf_matrix.shape[1]):
            fp.write(str(train_tfidf_matrix[i][j]) + " ")
        fp.write("\n")
    fp.close()

main()
