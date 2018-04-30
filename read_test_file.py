import numpy as np
import random
import sklearn.preprocessing

def generate_test_data() :
    full_data_file="tfidf_matrix_fulltext_test_small.txt"
    data_full_raw=np.genfromtxt(full_data_file,dtype=None,delimiter=" ")
    inlink_data_file="tfidf_matrix_inlinks_test_small.txt"
    data_inlink_raw=np.genfromtxt(full_data_file,dtype=None,delimiter=" ")
    print(data_full_raw.shape)
    print(data_full_raw)
    #extracting Features and DataLabel from the Full data information
    data_full=data_full_raw[:,:-1]
    label_full=data_full_raw[:,-1]
    #extracting Features and DataLabel from the inlinks Data information
    data_inlink=data_inlink_raw[:,:-1]
    label_inlink=data_inlink_raw[:,-1]
    print(data_full)
    #converting the Labels from Float to integer
    label_full=label_full.astype(int)
    label_inlink=label_inlink.astype(int)
    print(label_full)
    print(label_full.shape)
    # data_full=sklearn.preprocessing.normalize(data_full,axis=0)
    # data_inlink=sklearn.preprocessing.normalize(data_inlink,axis=0)
    data_full=sklearn.preprocessing.scale(data_full)
    data_inlink=sklearn.preprocessing.scale(data_inlink)
    number_of_samples=label_full.shape[0]
    print("-------------------------------------")

    label_full=label_full.reshape(label_full.shape[0], 1)
    label_inlink=label_inlink.reshape(label_inlink.shape[0], 1)
    # print(label_full.shape)
    # print(label_inlink.shape)
    # print(data_full.shape)

    data_full_with_label = np.concatenate((data_full, label_full), axis=1)
    data_inlink_with_label = np.concatenate((data_inlink, label_inlink), axis=1)

    class_0_full=data_full_with_label[data_full_with_label[:,-1]==0][:,:-1]
    class_0_inlink=data_inlink_with_label[data_inlink_with_label[:,-1]==0][:,:-1]
    class_1_full=data_full_with_label[data_full_with_label[:,-1]==1][:,:-1]
    class_1_inlink=data_inlink_with_label[data_inlink_with_label[:,-1]==1][:,:-1]
    # print("*******************************")
    # print(class_0_full.shape, class_0_inlink.shape)
    # print(class_1_full.shape, class_1_inlink.shape)
    # print(unlabelled_full.shape, unlabelled_inlink.shape)

    class_0 = np.stack((class_0_full, class_0_inlink), axis=0)
    class_1 = np.stack((class_1_full, class_1_inlink), axis=0)
    print(class_0.shape, class_1.shape)
    #putting all the 3 classes in a list
    class_view=[class_0,class_1]
    print(class_view[0].shape,class_view[1].shape)
    return class_view

#generate_test_data()
