from R_calculation import generate_R_value
from rand_Data_generate import get_random_matrix
from generate_Class_view import generate_data
from project_KNN import check_KNN
import numpy as np

#R_matrix=np.random.rand(4,4)
#class_view=get_random_matrix()
class_view=generate_data()
R_matrix=generate_R_value(class_view)
regularization_parameter=1

#From Eq 20 where A=tr(R*R(t))

A_temp_Matrix=np.dot(R_matrix,R_matrix.T)
#A_temp_Matrix=A_temp_Matrix.astype(int)
print(A_temp_Matrix)
A=np.trace(A_temp_Matrix)
print(A)
#Find Alpha from the solution of Eqution 20
alpha=0
if(regularization_parameter >= (1/A)) :
    alpha=(1/A)
else:
    alpha=regularization_parameter
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
check_KNN(W_matrix,class_view)
