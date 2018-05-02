from R_calculation import generate_R_value
# from rand_Data_generate import get_random_matrix
from generate_Class_view import generate_data
from project_KNN import check_KNN
import numpy as np

# #R_matrix=np.random.rand(4,4)
# #class_view=get_random_matrix()
# class_view=generate_data()
# R_matrix=generate_R_value(class_view)
# learning_rate=1

# #From Eq 20 where A=tr(R*R(t))

# A_temp_Matrix=np.dot(R_matrix,R_matrix.T)
# #A_temp_Matrix=A_temp_Matrix.astype(int)
# print(A_temp_Matrix)
# A=np.trace(A_temp_Matrix)
# print(A)
# #Find Alpha from the solution of Eqution 20
# alpha=0
# if(learning_rate >= (1/A)) :
#     alpha=(1/A)
# else:
#     alpha=learning_rate
# print(alpha)

# #From Equation 17 H=alpha*R_matrix
# H=alpha*R_matrix
# #Eigen Decomposition of Matrix
# print ("The H is")
# print(H)
# Eigen_values, Eigen_vectors=np.linalg.eigh(H)
# print(Eigen_values)
# print(Eigen_vectors)
# #Check the Recostruction of H from its Eigen Decomposition
# temp_eig=np.diag(Eigen_values)
# Eigen_inv=np.linalg.inv(Eigen_vectors)
# reconstructed_H=Eigen_vectors.dot(temp_eig).dot(Eigen_vectors.T)
# #reconstructed_H=Eigen_vectors.dot(temp_eig).dot(Eigen_inv)
# print(reconstructed_H)

# is_positive_semi_definite=False
# #Check if matrix is Positive Semi-Definite
# is_positive_semi_definite=np.all(Eigen_values >= 0)
# print(is_positive_semi_definite)

# if(is_positive_semi_definite == True) :
#     #If the Eigen Values are greater than 0, i.e. positie semi definite matrix then find weight vector
#     #based on equation #22
#     W_matrix=Eigen_vectors.dot(np.diag(np.sqrt(Eigen_values)))
# else :
#     Eigen_values[Eigen_values<0]=0
#     W_matrix=Eigen_vectors.dot(np.diag(np.sqrt(Eigen_values)))
# print("W_Matrix")
# print(W_matrix)

# #checking with KNN
# check_KNN(W_matrix,class_view)

final_array = []
for driver_counter_j in range(400,600,100):

	arr = np.empty((0,5), int)
	for driver_counter_i in range(1,4):
		fulltext_train_file = "tfidf_matrix_fulltext_train_"+str(driver_counter_j)+"_fold"+str(driver_counter_i)+".txt"
		inlinks_train_file = "tfidf_matrix_inlinks_train_"+str(driver_counter_j)+"_fold"+str(driver_counter_i)+".txt"
		fulltext_test_file = "tfidf_matrix_fulltext_test_"+str(driver_counter_j)+"_fold"+str(driver_counter_i)+".txt"
		inlinks_test_file = "tfidf_matrix_inlinks_test_"+str(driver_counter_j)+"_fold"+str(driver_counter_i)+".txt"

		print ("-+-+-+-+-+-+-+-+-+-FILE NAME+-+-+-+-+-+-+-+-+-+-+-+-+-+")
		print ("tfidf_matrix_"+str(driver_counter_j)+"_fold"+str(driver_counter_i))

		p_limit_values = [0.1,0.3,0.5,0.7,0.9] 
		a1 = []
		for p_i in p_limit_values:
			
			#R_matrix=np.random.rand(4,4)
			#class_view=get_random_matrix()
			print ("The P value being used->>>-->>>>---->>>>: "+str(p_i)+", "+str(driver_counter_j)+", "+str(driver_counter_i))
			class_view=generate_data(fulltext_train_file,inlinks_train_file,p_i)
			R_matrix=generate_R_value(class_view)
			learning_rate=1

			#From Eq 20 where A=tr(R*R(t))

			A_temp_Matrix=np.dot(R_matrix,R_matrix.T)
			#A_temp_Matrix=A_temp_Matrix.astype(int)
			print(A_temp_Matrix)
			A=np.trace(A_temp_Matrix)
			print(A)
			#Find Alpha from the solution of Eqution 20
			alpha=0
			if(learning_rate >= (1/A)) :
			    alpha=(1/A)
			else:
			    alpha=learning_rate
			print(alpha)

			#From Equation 17 H=alpha*R_matrix
			H=alpha*R_matrix
			#Eigen Decomposition of Matrix
			print ("The H is")
			print(H)
			Eigen_values, Eigen_vectors=np.linalg.eigh(H)
			print(Eigen_values)
			print(Eigen_vectors)
			#Check the Recostruction of H from its Eigen Decomposition
			temp_eig=np.diag(Eigen_values)
			Eigen_inv=np.linalg.inv(Eigen_vectors)
			reconstructed_H=Eigen_vectors.dot(temp_eig).dot(Eigen_vectors.T)
			#reconstructed_H=Eigen_vectors.dot(temp_eig).dot(Eigen_inv)
			print(reconstructed_H)

			is_positive_semi_definite=False
			#Check if matrix is Positive Semi-Definite
			is_positive_semi_definite=np.all(Eigen_values >= 0)
			print(is_positive_semi_definite)

			if(is_positive_semi_definite == True) :
			    #If the Eigen Values are greater than 0, i.e. positie semi definite matrix then find weight vector
			    #based on equation #22
			    W_matrix=Eigen_vectors.dot(np.diag(np.sqrt(Eigen_values)))
			else :
			    Eigen_values[Eigen_values<0]=0
			    W_matrix=Eigen_vectors.dot(np.diag(np.sqrt(Eigen_values)))
			print("W_Matrix")
			print(W_matrix)

			#checking with KNN
			temp_acc = check_KNN(W_matrix,class_view,fulltext_test_file,inlinks_test_file)
			a1.append(temp_acc)
			# print (a1)
		arr = np.append(arr,np.array([a1]), axis = 0)

	final_array.append(arr)






print("-+-+-+-+-+-+-+-+-+-The Final Value+-+-+-+-+-+-+-+-+-+-+-+-+-+")
for i in range (0,2,1):
	print ('This is for count:---->>>>--->>>>'+str((i+4)*100)+' fold data')
	print(final_array[i])

