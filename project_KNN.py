import numpy as np
import sklearn.metrics
from generate_Class_view import generate_data
from read_test_file import generate_test_data
def check_KNN(W_matrix,class_view) :
    #print("YOLO")
    #class_view=generate_data()
    #print("CAT")
    #W_matrix=np.random.rand(1050,900)
    n_class=len(class_view)
    number_of_views=class_view[1].shape[0]
    #Taking the projections of the samples on the Weight Matrix
    all_sample_projection=np.zeros((1,number_of_views,W_matrix.shape[1]))
    flattened_train_sample_projection=np.zeros((1,number_of_views*W_matrix.shape[1]))
    #Declaring a matrix containing the labels
    all_sample_labels=np.zeros(1)
    for k in range(n_class-1):
        store_class_array=class_view[k]
        print(store_class_array.shape)
        n_samples=store_class_array.shape[1]
        n_view=store_class_array.shape[0]
        #For every sample
        for i in range(n_samples):
            #For each sample will take data from all the Views
            sample_projection=np.zeros((1,W_matrix.shape[1]))
            for j in range(n_view):
                temp_projection_view_each=np.dot(W_matrix.T,store_class_array[j][i].T)
                print(temp_projection_view_each.shape)
                temp_projection_view_each=temp_projection_view_each.reshape(temp_projection_view_each.shape[0],1)
                print(temp_projection_view_each.T.shape)
                print(sample_projection.shape)
                sample_projection=np.concatenate((sample_projection,temp_projection_view_each.T),axis=0)
            sample_projection=sample_projection[1:,:]
            print("CAT")
            #Flatten the samples for storing in flat view format
            train_flat_projection=np.hstack(sample_projection)
            train_flat_projection=train_flat_projection.reshape(1,train_flat_projection.shape[0])
            print(sample_projection.shape)
            print(all_sample_projection.shape)
            flattened_train_sample_projection=np.concatenate((flattened_train_sample_projection,train_flat_projection),axis=0)
            sample_projection=sample_projection.reshape(1,sample_projection.shape[0],sample_projection.shape[1])
            all_sample_projection=np.concatenate((all_sample_projection,sample_projection),axis=0)
            all_sample_labels=np.append(all_sample_labels,k)
    all_sample_projection=all_sample_projection[1:,:,:]
    all_sample_labels=all_sample_labels[1:]
    flattened_train_sample_projection=flattened_train_sample_projection[1:]
    print(all_sample_labels.shape)
    print(all_sample_projection.shape)
    print(flattened_train_sample_projection.shape)

    #-----------------------------------------------------------------------------
    #Processing of Train Data done
    #Processing of Test Data Begining
    #-----------------------------------------------------------------------------

    test_view=generate_test_data()
    #print("CAT")
    #W_matrix=np.random.rand(1050,900)
    n_class=len(test_view)
    number_of_views=test_view[1].shape[0]
    #Taking the projections of the samples on the Weight Matrix
    test_all_sample_projection=np.zeros((1,number_of_views,W_matrix.shape[1]))
    flattened_sample_projection=np.zeros((1,number_of_views*W_matrix.shape[1]))
    #Declaring a matrix containing the labels
    test_all_sample_labels=np.zeros(1)
    for k in range(n_class):
        store_test_class_array=test_view[k]
        print(store_test_class_array.shape)
        n_samples=store_test_class_array.shape[1]
        n_view=store_test_class_array.shape[0]
        for i in range(n_samples):
            #For each sample will take data from all the Views
            test_sample_projection=np.zeros((1,W_matrix.shape[1]))
            for j in range(n_view):
                temp_projection_view_each=np.dot(W_matrix.T,store_test_class_array[j][i].T)
                print(temp_projection_view_each.shape)
                temp_projection_view_each=temp_projection_view_each.reshape(temp_projection_view_each.shape[0],1)
                print(temp_projection_view_each.T.shape)
                print(test_sample_projection.shape)
                test_sample_projection=np.concatenate((test_sample_projection,temp_projection_view_each.T),axis=0)
            test_sample_projection=test_sample_projection[1:,:]
            print("CAT")
            #storing the flattened projection sample
            flat_projection=np.hstack(test_sample_projection)
            flat_projection=flat_projection.reshape(1,flat_projection.shape[0])
            print(flat_projection.shape)
            print(flattened_sample_projection.shape)
            flattened_sample_projection=np.concatenate((flattened_sample_projection,flat_projection),axis=0)
            #storing the regular projection
            print(test_sample_projection.shape)
            print(test_all_sample_projection.shape)
            test_sample_projection=test_sample_projection.reshape(1,test_sample_projection.shape[0],test_sample_projection.shape[1])
            test_all_sample_projection=np.concatenate((test_all_sample_projection,test_sample_projection),axis=0)
            test_all_sample_labels=np.append(test_all_sample_labels,k)
    test_all_sample_projection=test_all_sample_projection[1:,:,:]
    test_all_sample_labels=test_all_sample_labels[1:]
    flattened_sample_projection=flattened_sample_projection[1:,:]
    print(test_all_sample_labels.shape)
    print(test_all_sample_projection.shape)
    print(flattened_sample_projection.shape)
    #----------------------------------------------
    #--------Data Projection on W Done-------------
    #--------Cosine Similarity and KNN-------------
    #----------------------------------------------
    #print(all_sample_labels)
    cosine_similarity = sklearn.metrics.pairwise.cosine_similarity(flattened_train_sample_projection, flattened_sample_projection)
    print(cosine_similarity.shape)
    #Finding the K nearest Neighbour Based on Cosine cosine_similarity
    total=cosine_similarity.shape[1]
    correct=0
    for x in range(cosine_similarity.shape[1]) :
        augmented_similarity_matrix = np.concatenate((cosine_similarity[:, x].reshape(cosine_similarity[:, 0].shape[0], 1), all_sample_labels.reshape(all_sample_labels.shape[0], 1)), axis = 1)
        sorted_similarity_matrix = augmented_similarity_matrix[augmented_similarity_matrix[:, 0].argsort()]
        #print(sorted_similarity_matrix)
        similar_results = sorted_similarity_matrix[-11:, -1]
        similar_results = similar_results.astype(int)
        print(similar_results)
        class_labels = np.unique(similar_results)
        counts = np.bincount(similar_results)
        prediction = np.argmax(counts)
        #print("Predicted class label: ", prediction)
        if(prediction==test_all_sample_labels[x]):
            correct=correct+1
    acc=(correct/total)*100
    print(acc)





#check_KNN()
