import numpy as np

def generate_R_value(class_view):
    n_class=len(class_view)

    number_of_features=class_view[0].shape[2]
    S_W=np.zeros((number_of_features,number_of_features))

    #-----------------------------------------------------------------------------------------
    #-----------------------Calculation of S_W part of equation for review--------------------
    #-----------------------------------------------------------------------------------------
    # (-1) Since S_W calculation would not involve C+1th class of unlabelled Data
    for i in range(n_class-1) :

        ar_class=class_view[i]
        # print(i)
        # print(ar_class)
        S_W_temp_class=np.zeros((number_of_features,number_of_features))
        n_views,n_docs,n_features=ar_class.shape
        #print(n_views,n_docs,n_features)

        # Outer Loop for Each view
        for s in range(n_views) :
            # S(th) view in class i
            ar_view_s=ar_class[s]
            #Inner Loop For Views
            for t in range(n_views) :
                #t(th) view in class i
                ar_view_t=ar_class[t]
                doc_len_1=ar_view_s.shape[0]
                # for each document in s(th) view
                for p in range(doc_len_1) :
                    doc_len_2=ar_view_t.shape[0]
                    # for each document in s(th) view take a document in t(th) view
                    for q in range(doc_len_2) :
                        doc_1=ar_view_s[p]
                        doc_2=ar_view_t[q]
                        #The Feature Vectors are in 1D matrix of length(Number Of Features)
                        #Converting the to 2*D matrix of Dimnesion (1 X Number Of Features)
                        doc_1=doc_1.reshape(ar_view_s.shape[1],1)
                        doc_2=doc_2.reshape(ar_view_t.shape[1],1)
                        temp_prod=np.dot(doc_1,doc_2.T)
                        #print (temp_prod)
                        #print(temp_prod.shape)
                        #Keeping Summation in S_W
                        S_W_temp_class=S_W_temp_class+temp_prod
        #l(i) which represents the number of documents in class i
        l_i=n_views*n_docs
        S_W_temp_class=S_W_temp_class/(l_i*l_i)
        #print (S_W_temp_class.shape)
        S_W=S_W+S_W_temp_class

    #Dividing the S_W with number of classes

    S_W=S_W/(n_class-1)
    #print("final S_W")
    #print(S_W)

    #-----------------------------------------------------------------------------------------
    #-----------------------Calculation of S_B part of equation for review--------------------
    #-----------------------------------------------------------------------------------------

    S_B=np.zeros((number_of_features,number_of_features))
    for i in range(n_class-1) :
        #Extracting all the view_doc records for a class i
        ar_class_i=class_view[i]
        n_views_i,n_docs_i,n_features_i=ar_class_i.shape
        for j in range(n_class-1) :
            #If Both the Classes same Exit Loop
            S_B_temp_class=np.zeros((number_of_features,number_of_features))
            if i != j :
                #Since interclass Scatter proceed with different Classes
                ar_class_j=class_view[j]
                n_views_j,n_docs_j,n_features_j=ar_class_j.shape
                #For every view in class one
                for s in range(n_views_i) :
                    # S(th) view in class i
                    ar_view_s=ar_class_i[s]
                    #Inner Loop for views
                    for t in range(n_views_j) :
                        #t(th) view in class j
                        ar_view_t=ar_class_j[t]
                        #Number of Documents in View1
                        doc_len_1=ar_view_s.shape[0]
                        for p in range(doc_len_1) :
                            doc_len_2=ar_view_t.shape[0]
                            # for each document in s(th) view take a document in t(th) view
                            for q in range(doc_len_2) :
                                doc_1=ar_view_s[p]
                                doc_2=ar_view_t[q]
                                #The Feature Vectors are in 1D matrix of length(Number Of Features)
                                #Converting the to 2*D matrix of Dimnesion (1 X Number Of Features)
                                doc_1=doc_1.reshape(ar_view_s.shape[1],1)
                                doc_2=doc_2.reshape(ar_view_t.shape[1],1)
                                temp_prod=np.dot(doc_1,doc_2.T)
                                S_B_temp_class=S_B_temp_class+temp_prod
                #l(i) which represents the number of documents in class i
                l_i=n_views_i*n_docs_i
                #l(j) which represents the number of documents in class j
                l_j=n_views_j*n_docs_j
                S_B_temp_class=S_B_temp_class/(l_i*l_j)
                #print (S_W_temp_class.shape)
                S_B=S_B+S_B_temp_class
    S_B=S_B/((n_class-1)*(n_class-2))
    #print("final S_B")
    #print(S_B)

    #-----------------------------------------------------------------------------------------
    #-----------------------Calculation of S_T part of equation for review--------------------
    #-----------------------------------------------------------------------------------------

    S_T=np.zeros((number_of_features,number_of_features))#-----Full S_T Sum-------------------
    S_T_W=np.zeros((number_of_features,number_of_features))#-----S_W part of S_T--------------
    for i in range(n_class) :

        ar_class=class_view[i]
        # print(i)
        # print(ar_class)
        S_T_W_temp_class=np.zeros((number_of_features,number_of_features))
        n_views,n_docs,n_features=ar_class.shape
        print(n_views,n_docs,n_features)

        # Outer Loop for Each view
        for s in range(n_views) :
            # S(th) view in class i
            ar_view_s=ar_class[s]
            #Inner Loop For Views
            for t in range(n_views) :
                #t(th) view in class i
                ar_view_t=ar_class[t]
                doc_len_1=ar_view_s.shape[0]
                # for each document in s(th) view
                for p in range(doc_len_1) :
                    doc_len_2=ar_view_t.shape[0]
                    # for each document in s(th) view take a document in t(th) view
                    for q in range(doc_len_2) :
                        doc_1=ar_view_s[p]
                        doc_2=ar_view_t[q]
                        #The Feature Vectors are in 1D matrix of length(Number Of Features)
                        #Converting the to 2*D matrix of Dimnesion (1 X Number Of Features)
                        doc_1=doc_1.reshape(ar_view_s.shape[1],1)
                        doc_2=doc_2.reshape(ar_view_t.shape[1],1)
                        temp_prod=np.dot(doc_1,doc_2.T)
                        #print (temp_prod)
                        #print(temp_prod.shape)
                        #Keeping Summation in S_W
                        S_T_W_temp_class=S_T_W_temp_class+temp_prod
        #l(i) which represents the number of documents in class i
        l_i=n_views*n_docs
        S_W_temp_class=S_W_temp_class/(l_i*l_i)
        #print (S_W_temp_class.shape)
        S_T_W=S_T_W+S_T_W_temp_class


    #--------------Calculation of the S_B part ---------------
    #---------------------------------------------------------
    S_T_B=np.zeros((number_of_features,number_of_features))
    for i in range(n_class) :
        #Extracting all the view_doc records for a class i
        ar_class_i=class_view[i]
        n_views_i,n_docs_i,n_features_i=ar_class_i.shape
        for j in range(n_class) :
            #If Both the Classes same Exit Loop
            S_T_B_temp_class=np.zeros((number_of_features,number_of_features))
            if i != j :
                #Since interclass Scatter proceed with different Classes
                ar_class_j=class_view[j]
                n_views_j,n_docs_j,n_features_j=ar_class_j.shape
                #For every view in class one
                for s in range(n_views_i) :
                    # S(th) view in class i
                    ar_view_s=ar_class_i[s]
                    #Inner Loop for views
                    for t in range(n_views_j) :
                        #t(th) view in class j
                        ar_view_t=ar_class_j[t]
                        #Number of Documents in View1
                        doc_len_1=ar_view_s.shape[0]
                        for p in range(doc_len_1) :
                            doc_len_2=ar_view_t.shape[0]
                            # for each document in s(th) view take a document in t(th) view
                            for q in range(doc_len_2) :
                                doc_1=ar_view_s[p]
                                doc_2=ar_view_t[q]
                                #The Feature Vectors are in 1D matrix of length(Number Of Features)
                                #Converting the to 2*D matrix of Dimnesion (1 X Number Of Features)
                                doc_1=doc_1.reshape(ar_view_s.shape[1],1)
                                doc_2=doc_2.reshape(ar_view_t.shape[1],1)
                                temp_prod=np.dot(doc_1,doc_2.T)
                                S_T_B_temp_class=S_T_B_temp_class+temp_prod
                #l(i) which represents the number of documents in class i
                l_i=n_views_i*n_docs_i
                #l(j) which represents the number of documents in class j
                l_j=n_views_j*n_docs_j
                S_T_B_temp_class=S_T_B_temp_class/(l_i*l_j)
                #print (S_W_temp_class.shape)
                S_T_B=S_T_B+S_T_B_temp_class

    #Final S_T calculation
    S_T=S_T_B+((n_class-1)*S_T_W)
    S_T=S_T/(2*(n_class)*(n_class-1))

    r1=0.009
    r2=0.009
    #Calculating R
    R=S_W-(r1*S_B)-(r2*S_T)

    print("the R is")
    print (R)
    return R
