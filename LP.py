import numpy as np

np.set_printoptions(precision=2)
class LinearProg:
    """
    A class to store a linear program.

    Attributes
    ----------
    num_eq : int
    num_var : int
    LHS : matrix of size num_eq x num_var
    RHS : vector of size num_eq
    """
    
    def print(self):
        'Prints equations.'
        for i in range(self.num_eq):
            for j in range(self.num_var - 1):
                print (str(self.LHS[i][j]) + '*x_' + str(j), end = ' + ')
            print (str(self.LHS[i][self.num_var - 1]) + '*x_' + str(self.num_var - 1), end = ' <= ')
            print (self.RHS[i])
                
    def __init__(self, LHS: np.array, RHS: np.array):
        if len(LHS) != len(RHS):
            # improve the type check here
            print ("Error: Number of rows of A (LHS) does not match the number of rows of b (RHS).")
            return
        self.LHS = LHS              # matrix A of size m x n
        self.RHS = RHS              # vector b of length n
        self.num_eq = len(LHS)      # number of equations
        self.num_var = len(LHS[0])  # number of variables

