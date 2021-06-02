import numpy as np

np.set_printoptions(precision=2)


class FourierMotzkin:
    """
    A class to store a linear program.

    Attributes
    ----------
    num_eq : int
    num_var : int
    LHS : matrix of size num_eq x num_var
    RHS : vector of size num_eq

    The LP is of the form LHS x <= RHS.
    """

    def print(self):
        "Prints equations."
        for i in range(self.num_eq):
            for j in range(self.num_var - 1):
                print(str(self.LHS[i][j]) + "*x_" + str(j), end=" + ")
                print(
                    str(self.LHS[i][self.num_var - 1]) + "*x_" + str(self.num_var - 1),
                    end=" <= ",
                )
                print(self.RHS[i])

    def __init__(self, LHS: np.array, RHS: np.array):
        if len(LHS) != len(RHS):
            # improve the type check here
            print(
                "Error: Number of rows of A (LHS) does not match the number of rows of b (RHS)."
            )
            return
        self.LHS = LHS  # matrix A of size m x n
        self.RHS = RHS  # vector b of length n
        self.num_eq = len(LHS)  # number of equations
        self.num_var = len(LHS[0])  # number of variables

    def fourier_motzkin(self):
        lp = self

        "Running Fourier-Motzkin algorithm on lp."
        print("Running Fourier-Motzkin algorithm on the following linear program: ")
        lp.print()

        for i in range(lp.num_var - 1):
            lp = self.fourier_motzkin_step(lp, lp.num_var - 1)
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

        if len(pos_coeff) == 0 or len(neg_coeff) == 0:
            return True

        max_neg = max([lp.RHS[coeff[0]] / coeff[1] for coeff in neg_coeff])
        min_pos = min([lp.RHS[coeff[0]] / coeff[1] for coeff in pos_coeff])

        if max_neg <= min_pos:
            return True

        return False

    @classmethod
    def fourier_motzkin_step(cls, lp: LinearProg, n: int):
        "Eliminates a single variable."
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

        # Normalize rows with positive and negative coefficients
        for coeff in pos_coeff:
            lp.LHS[coeff[0]] = [x / coeff[1] for x in lp.LHS[coeff[0]]]
            lp.RHS[coeff[0]] = lp.RHS[coeff[0]] / coeff[1]
        for coeff in neg_coeff:
            lp.LHS[coeff[0]] = [x / -coeff[1] for x in lp.LHS[coeff[0]]]
            lp.RHS[coeff[0]] = lp.RHS[coeff[0]] / -coeff[1]

        new_LHS = []
        new_RHS = []

        # Add rows with null coefficients unmodified
        for row in nul_coeff:
            new_LHS.append(lp.LHS[row])
            new_RHS.append(lp.RHS[row])

        # Create new rows for each pairs of positive and negative coefficients
        for row_pos in pos_coeff:
            for row_neg in neg_coeff:
                new_LHS.append(
                    [
                        lp.LHS[row_pos[0]][i] + lp.LHS[row_neg[0]][i]
                        for i in range(lp.num_var)
                    ]
                )
                new_RHS.append(lp.RHS[row_pos[0]] + lp.RHS[row_neg[0]])

        # Delete old rows for positive and negative coefficients
        lp.num_eq = len(nul_coeff) + len(pos_coeff) * len(neg_coeff)
        lp.num_var = lp.num_var - 1
        if lp.num_eq > 0:
            lp.LHS = np.delete(new_LHS, n, axis=1)
            lp.RHS = new_RHS

        print("---------------------------------")
        print("After eliminating the variable x_" + str(n) + ":")
        lp.print()

        return lp


lp = FourierMotzkin([[2, -5, 4], [3, -6, 3], [-1, 5, -2], [-3, 2, 6]], [10, 9, -7, 12])
print(lp.fourier_motzkin())
