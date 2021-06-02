import numpy as np


class LinearProgram:
    """
    A class to store a linear program.

    Attributes
    ----------
    num_eq : int
    num_var : int
    A : matrix of size num_eq x num_var
    b : vector of size num_eq
    cost : np.array

    The LP is of the form A x = b, x >= 0.
    """

    def print(self):
        "Prints the system of equations."
        for i in range(self.num_eq):
            for j in range(self.num_var - 1):
                print(str(self.A[i][j]) + "*x_" + str(j), end=" + ")
            print(
                str(self.A[i][self.num_var - 1]) + "*x_" + str(self.num_var - 1),
                end=" = ",
            )
            print(self.b[i])
        for i in range(self.num_var):
            print("x_" + str(i) + " >= 0")

    def __init__(self, A: np.array, b: np.array, cost: np.array):
        if len(A[0]) != len(cost):
            # TODO: improve the type check here
            print(
                "Error: Number of rows of A does not match the number of rows of the cost function."
            )
            return
        if len(A) != len(b):
            # TODO: improve the type check here
            print("Error: Number of rows of A does not match the number of rows of b.")
            return
        self.A = A
        self.b = b
        self.cost = cost
        self.num_eq = len(A)  # number of equations
        self.num_var = len(A[0])  # number of variables


class SimplexMethod:
    """
    A class for implementing Simplex Method.

    Attributes
    ----------
    lp: LinearProgram

    """

    def __init__(
        self,
        lp: LinearProgram,
        basic_columns: np.array,
        starting_vertex: np.array,
    ):
        self.lp = lp
        self.basic_columns = np.array(basic_columns)
        self.vertex = starting_vertex

        self.update_inv_naive()

    def update_inv_naive(self):
        self.B_inv = np.linalg.inv(self.lp.A[:, self.basic_columns])

    def cost(self, vector: np.array):
        return np.matmul(np.transpose(self.cost), vector)

    def reduced_cost_naive(self, index: int):
        "Returns the reduced cost c'_j = c_j - c'_B * B^{-1} * A_j"
        cB = self.lp.cost[self.basic_columns]
        reduced_cost = self.lp.cost[index] - np.matmul(
            cB, np.matmul(self.B_inv, self.lp.A[:, index])
        )
        return reduced_cost

    def simplex_method_step(self, i: int):
        u = np.matmul(self.B_inv, self.lp.A[:, i])
        pos_val = u[u > 0]
        x = self.vertex[u > 0]
        if len(pos_val) == 0:
            return False
        thetas = x / pos_val
        theta_min = min(thetas)
        theta_argmin = thetas.index(theta_min)
        self.basic_columns[theta_argmin] = theta_min
        self.update_inv_naive()
        self.vertex = [self.vertex[i] - theta_min * u[i] for i in range(len(u))]
        self.vertex[theta_argmin] = theta_min
        return True

    def simplex_method(self, reduced_cost_calc=reduced_cost_naive):
        # Loop over the indices `i` which are not in one of the basic_columns.
        for i in self.lp.num_var:
            if not (i in self.basic_columns):
                # Calculate the reduced cost
                reduced_cost = reduced_cost_calc(i)
                # If the reduced cost is neganive, then move in the direction of `i` to the next vertex
                if reduced_cost < 0:
                    # if search for the next direction is succesful, restart the process from the new vertex
                    if self.simplex_method_step(i):
                        break
                    print("Cost is -infinite")
                    return False

            # If all reduced costs are non-negative then `i` is the optimal solution
            print("Found optimal solution")
            return i

        self.simplex_method()


# lp = SimplexMethod(
#     [[2, -5, 4, 2], [3, -6, 3, 3], [-1, 5, -2, -1]],
#     [10, 9, -7],
#     [1, 2, 3, 4],
#     [0, 1, 2],
# )
# # lp.print()
# test = lp.reduced_cost_naive(3)
# print(str(test))
