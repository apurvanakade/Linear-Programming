import LP
import FourierMotzkin

l1 = LP.LinearProg([[2, -5, 4],
                 [3, -6, 3],
                 [-1, 5, -2],
                 [-3, 2, 6]],
                [10, 9, -7, 12])
print(FourierMotzkin.LinearProgAlgo.fourier_motzkin(l1))
