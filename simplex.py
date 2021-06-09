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

    def __init__(self, A: np.array, b: np.array, cost: np.array):

        self.num_eq = len(A)  # number of equations
        self.num_var = len(A[0])  # number of variables

        if len(cost) != self.num_var:
            print(
                "Error: Number of rows of A does not match\
                the number of rows of the cost function."
            )
            return
        if len(b) != self.num_eq:
            print("Error: Number of rows of A does not match the number of rows of b.")
            return

        self.A = np.array(A)
        self.b = np.array(b)
        self.cost = np.array(cost)

    def print(self):
        "Prints the system of equations."

        print("Minimize: ")
        for i in range(self.num_var):
            print(str(self.cost[i]) + "*x_" + str(i), end=" + ")
        print(str(self.cost[self.num_var - 1]) + "*x_" + str(self.num_var - 1))

        print("Subject to constraints:")
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


class SimplexMethod:
    """
    A class for implementing Simplex Method.

    Attributes
    ----------
    lp: LinearProgram
    basis_columns:
    vertex
    B_inv

    """

    def __init__(self, lp: LinearProgram, basis_columns: np.array):

        self.lp = lp
        self.basis_columns = np.array(basis_columns)
        self.update_inv_naive()
        self.calculate_vertex()

    def update_inv_naive(self):
        self.B_inv = np.linalg.inv(self.lp.A[:, self.basis_columns])

    def calculate_vertex(self):

        self.vertex = np.zeros(self.lp.num_var)

        # aux_vertex = B^{-1} * b
        aux_vertex = np.matmul(self.B_inv, self.lp.b)

        # put the values of `aux_vertex` into appropriate "spots".
        for i in range(self.lp.num_var):
            if i in self.basis_columns:
                self.vertex[i] = aux_vertex[(np.where(self.basis_columns == i))[0]]

    def cost(self):
        return np.dot(self.lp.cost, self.vertex)

    def reduced_cost_naive(self, index: int):
        "Returns the reduced cost in the direction `j`: c'_j = c_j - c'_B * B^{-1} * A_j"
        cB = self.lp.cost[self.basis_columns]
        reduced_cost = self.lp.cost[index] - np.matmul(
            cB, np.matmul(self.B_inv, self.lp.A[:, index])
        )
        return reduced_cost

    def step(self, i: int):

        # u = B^{-1} * A_i
        u = np.matmul(self.B_inv, self.lp.A[:, i])

        # values in `u` that are positive
        u_pos = u[u > 0]

        # If no-coordinates of `u` are positive,
        # then there is no vertex in this direction.
        if len(u_pos) == 0:
            return False

        # Basis vectors corresponding to the positive entries in `u`
        b_pos = self.basis_columns[u > 0]

        # Coordinates of `x` corresponding to the positive values of `u`
        x = self.vertex[b_pos]

        # normalize the positive coordinates and find the min values
        thetas = x / u_pos
        theta_argmin = np.argmin(thetas)
        theta_min = min(thetas)

        # Replace the i^th basis_column with the `theta_argmin` one
        self.basis_columns[theta_argmin] = i
        self.update_inv_naive()
        self.calculate_vertex()

        # Coordinates of the new vertex
        # self.vertex = [
        #     self.vertex[i] - theta_min * u[i] for i in range(len(self.vertex))
        # ]
        # self.vertex[theta_argmin] = theta_min

        # new_vertex = np.zeros(self.lp.num_var)
        # for i in range(self.lp.num_eq):
        #     new_vertex[self.basis_columns[i]] = (
        #         self.vertex[self.basis_columns[i]] - theta_min * u[i]
        #     )
        # new_vertex[self.basis_columns[theta_argmin]] = theta_min
        # self.vertex = new_vertex

        return True

    def solve(self, reduced_cost_calc=reduced_cost_naive):
        # Loop over the indices `i` which are not in one of the basis_columns.
        for i in range(self.lp.num_var):
            if not (i in self.basis_columns):

                # Calculate the reduced cost in the `i` direction.
                reduced_cost = self.reduced_cost_naive(i)

                # If the reduced cost is negative,
                # then move in the direction of `i` to the next vertex.
                if reduced_cost < 0:
                    # in the direction of `i`
                    # if search for the next direction is succesful,
                    # restart the process from the new vertex
                    # else the cost is -infinity
                    if self.step(i):
                        break
                    return -1

                # If all reduced costs are non-negative
                # then `i` is the optimal solution
                return i

        self.solve()

    def print_solution(self):

        print("-" * 20)
        self.lp.print()

        print("-" * 20)
        print("Solution:")
        for i in range(len(self.vertex)):
            print("x_" + str(i) + " = " + str(round(self.vertex[i], 2)))
        print("Minimal cost = ", round(self.cost(), 2))
        return


lp = LinearProgram(
    A=[[1, 2, 2, 1, 0, 0], [2, 1, 2, 0, 1, 0], [2, 2, 1, 0, 0, 1]],
    b=[20, 20, 20],
    cost=[-10, -12, -12, 0, 0, 0],
)

simplex = SimplexMethod(lp, basis_columns=[3, 4, 5])

simplex.solve()

simplex.print_solution()

# lp.print()

# lp = SimplexMethod(
#     [[2, -5, 4, 2], [3, -6, 3, 3], [-1, 5, -2, -1]],
#     [10, 9, -7],
#     [1, 2, 3, 4],
#     [0, 1, 2],
# )
# # lp.print()
# test = lp.reduced_cost_naive(3)
# print(str(test))
