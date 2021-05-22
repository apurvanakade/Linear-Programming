import numpy as np


class LinearProgStandard:
    """
    A class to store a linear program.

    Attributes
    ----------
    num_eq : int
    num_var : int
    A : matrix of size num_eq x num_var
    b : vector of size num_eq

    The LP is of the form A x = b, x >= 0.
    """
    
    def print(self):
        'Prints equations.'
        for i in range(self.num_eq):
            for j in range(self.num_var - 1):
                print (str(self.A[i][j]) + '*x_' + str(j), end = ' + ')
            print (str(self.A[i][self.num_var - 1]) + '*x_' + str(self.num_var - 1), end = ' <= ')
            print (self.b[i])
                
    def __init__(self, A: np.array, b: np.array):
        if len(A) != len(b):
            # improve the type check here
            print ("Error: Number of rows of A does not match the number of rows of b.")
            return
        self.A = A              # matrix A of size m x n
        self.b = b              # vector b of length n
        self.num_eq = len(A)      # number of equations
        self.num_var = len(A[0])  # number of variables
