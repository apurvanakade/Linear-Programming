import numpy as np
import LP

np.set_printoptions(precision=2)

class LinearProgAlgo:
    """
    A class for linear prorgamming algorithms.

    Methods
    -------
    fourier_motzkin
    fourier_motzkin_step
    
    """
    @classmethod
    def fourier_motzkin (cls, lp: LP.LinearProg):
        'Runs fourier_motzkin algorithm on lp.'
        print("Received the following LinearProgram: ")
        lp.print()

        for i in range(lp.num_var - 1):
            lp = cls.fourier_motzkin_step(lp, lp.num_var - 1)
            if lp.num_eq == 0:
                print("Number of equations reduced to 0.")
                return True

        pos_coeff = []
        neg_coeff = []
        
        for i in range(lp.num_eq):
            if lp.LHS[i][0] > 0:
                pos_coeff.append([i, lp.LHS[i][0]])
            elif lp.LHS[i][0] < 0:
                neg_coeff.append([i, lp.LHS[i][0]])
                
        if (len(pos_coeff) == 0 or len(neg_coeff) == 0):
            return True

        max_neg = max([lp.RHS[coeff[0]]/coeff[1] for coeff in neg_coeff])
        min_pos = min([lp.RHS[coeff[0]]/coeff[1] for coeff in pos_coeff])

        if max_neg <= min_pos:
            return True

        return False

    @classmethod
    def fourier_motzkin_step (cls, lp: LP.LinearProg, n: int):
        'Eliminates a single variable.'
        print("---------------------------------")
        print("Eliminating variable x_" + str(n))
        
        nul_coeff = []
        pos_coeff = []
        neg_coeff = []
        
        for i in range(lp.num_eq):
            if lp.LHS[i][n] == 0:
                nul_coeff.append(i)
            elif lp.LHS[i][n] > 0:
                pos_coeff.append([i, lp.LHS[i][n]])
            else:
                neg_coeff.append([i, lp.LHS[i][n]])
                
        # print ("Rows with null coefficients:", nul_coeff)
        # print ("Rows with positive coefficients:", pos_coeff)
        # print ("Rows with negative coefficients:", neg_coeff)

        # Normalize rows with positive and negative coefficients
        for coeff in pos_coeff:
            lp.LHS[coeff[0]] = [x/coeff[1] for x in lp.LHS[coeff[0]]]
            lp.RHS[coeff[0]] = lp.RHS[coeff[0]]/coeff[1]
        for coeff in neg_coeff:
            lp.LHS[coeff[0]] = [x/-coeff[1] for x in lp.LHS[coeff[0]]]
            lp.RHS[coeff[0]] = lp.RHS[coeff[0]]/-coeff[1]

        # print("---------------------------------")
        # print("After normalizing")
        # lp.print()

        new_LHS = []
        new_RHS = []
        for row in nul_coeff:
            new_LHS.append(lp.LHS[row])
            new_RHS.append(lp.RHS[row])
            
        # Create new rows for each pairs of positive and negative coefficients
        for row_pos in pos_coeff:
            for row_neg in neg_coeff:
                new_LHS.append([lp.LHS[row_pos[0]][i] + lp.LHS[row_neg[0]][i]
                                for i in range(lp.num_var)])
                new_RHS.append(lp.RHS[row_pos[0]] + lp.RHS[row_neg[0]])
                
        # Delete old rows for positive and negative coefficients
        lp.num_eq = len(nul_coeff) + len(pos_coeff) * len(neg_coeff)
        lp.num_var = lp.num_var - 1
        if lp.num_eq > 0:
            lp.LHS = np.delete(new_LHS, n, axis = 1)
            lp.RHS = new_RHS

        print("---------------------------------")
        print("After eliminating the variable x_" + str(n) + ":")
        lp.print()

        return lp
    

