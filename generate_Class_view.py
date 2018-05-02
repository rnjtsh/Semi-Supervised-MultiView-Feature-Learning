import numpy as np
import random
import sklearn.preprocessing

def generate_data() :
    full_data_file="tfidf_matrix_fulltext_train_100_fold3.txt"
    data_full_raw=np.genfromtxt(full_data_file,dtype=None,delimiter=" ")
    inlink_data_file="tfidf_matrix_inlinks_train_100_fold3.txt"
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
    #data_full=sklearn.preprocessing.normalize(data_full,axis=0)
    #data_inlink=sklearn.preprocessing.normalize(data_inlink,axis=0)
    data_full=sklearn.preprocessing.scale(data_full)
    data_inlink=sklearn.preprocessing.scale(data_inlink)
    number_of_samples=label_full.shape[0]
    #in the following loop put some data unlabelled, the unlabelled data will have class label -1
    for i in range(number_of_samples):
        #generate a random number p between 0 and 1
        p=random.uniform(0,1)
        if(p<0.7):
            label_full[i]=-1
            label_inlink[i]=-1

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
    unlabelled_full=data_full_with_label[data_full_with_label[:,-1]==-1][:,:-1]
    unlabelled_inlink=data_inlink_with_label[data_inlink_with_label[:,-1]==-1][:,:-1]
    # print("*******************************")
    # print(class_0_full.shape, class_0_inlink.shape)
    # print(class_1_full.shape, class_1_inlink.shape)
    # print(unlabelled_full.shape, unlabelled_inlink.shape)

    class_0 = np.stack((class_0_full, class_0_inlink), axis=0)
    class_1 = np.stack((class_1_full, class_1_inlink), axis=0)
    unlabelled = np.stack((unlabelled_full, unlabelled_inlink), axis=0)

    print(class_0.shape, class_1.shape, unlabelled.shape)
    #putting all the 3 classes in a list
    class_view=[class_0,class_1,unlabelled]
    print(class_view[0].shape,class_view[1].shape,class_view[2].shape)
    return class_view

#generate_data()
'''
class_0_full=np.zeros((1,data_full.shape[1]))
class_0_inlink=np.zeros((1,data_full.shape[1]))
class_1_full=np.zeros((1,data_full.shape[1]))
class_1_inlink=np.zeros((1,data_full.shape[1]))
unlabelled_full=np.zeros((1,data_full.shape[1]))
unlabelled_inlink=np.zeros((1,data_full.shape[1]))
for i in range(number_of_samples):
    if(label_full[i]==0):
        class_0_full=np.append(class_0_full,data_full[i],axis=0)
        class_0_inlink=np.append(class_0_inlink,data_inlink[i],axis=0)
    elif(label_full[i]==1):
        class_1_full=np.append(class_1_full,data_full[i],axis=0)
        class_1_inlink=np.append(class_1_inlink,data_inlink[i],axis=0)
    elif(label_full[i]==-1):
        unlabelled_full=np.append(unlabelled_full,data_full[i],axis=0)
        unlabelled_inlink=np.append(unlabelled_inlink,data_inlink[i],axis=0)

class_0=np.append(class_0_full,class_0_inlink,axis=0)
class_1=np.append(class_1_full,class_1_inlink,axis=0)
unlabelled=np.append(unlabelled_full,unlabelled_inlink,axis=0)

print(class_0.shape)
print(class_1.shape)
print(unlabelled.shape)
'''
