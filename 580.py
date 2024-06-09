import numpy as np

A = np.array([[6, 2, -5], [3, 3, -2], [7, 5, -3]])
B = np.array([13, 13, 26])
output = np.linalg.solve(A, B)
print(f"a = {(output[0])}, b =  {(output[2])} c = {(output[2])}")
print(f"a = {round(output[0])}, b = {round(output[1])}, c = {round(output[2])}")