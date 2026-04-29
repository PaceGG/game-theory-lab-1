import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# 1. Сгенерируйте случайную платёжную матрицу A размерности n×m, где n > 10
# и m > 10 Элементы матрицы — целые числа в диапазоне [−100, 100].
np.random.seed(42)
n, m = 50, 50
payment_matrix = np.random.randint(-100, 101, size=(n, m))

# 2 Реализуйте алгоритм метода фиктивного разыгрывания (Брауна-Робинсона)
# для нахождения приближённого решения игры.
def brown_robinson(matrix: ndarray, iterations):
    n, m = matrix.shape

    count_a = np.zeros(n)
    count_b = np.zeros(m)

    sum_a = np.zeros(n)
    sum_b = np.zeros(m)

    i, j = 0, 0

    count_a[i] += 1
    count_b[j] += 1

    lower_prices = []
    upper_prices = []

    for k in range(1, iterations):
        sum_a += matrix[:, j]
        sum_b += matrix[i, :]

        i = np.argmax(sum_a)

        j = np.argmin(sum_b)

        count_a[i] += 1
        count_b[j] += 1

        lower_price = np.max(sum_a) / k
        lower_prices.append(lower_price)

        upper_price = np.min(sum_b) / k
        upper_prices.append(upper_price)
    
    value_min = np.max(sum_a) / iterations
    value_max = np.min(sum_b) / iterations

    game_value = (value_min + value_max) / 2


    return game_value, lower_prices, upper_prices

game_value, lower_prices, upper_prices = brown_robinson(payment_matrix, 50000)

print(f"Цена игры: {game_value:.3f} (метод Брауна-Робинсона)")

# Постройте график сходимости нижней и верхней цены игры в зависимости от
# номера итерации.
plt.figure()

plt.plot(lower_prices, label="нижняя цена игры")
plt.plot(upper_prices, label="верхняя цена игры")

plt.xlabel("итерация")
plt.ylabel("цена игры")

plt.title("график сходимости нижней и верхней цены игры")

plt.legend()
plt.grid()

# plt.show()

# Сравните полученное значение цены игры с решением, полученным через сведе-
# ние к задаче линейного программирования (например, через scipy.optimize.linprog).
def linear(matrix: ndarray):
    n, m = matrix.shape

    shift = abs(matrix.min()) + 1
    matrix_shifted = matrix + shift

    c = np.ones(n)

    A_ub = -matrix_shifted.T
    b_ub = -np.ones(m)

    bounds = [(0, None)] * n

    res = linprog(
        c,
        A_ub = A_ub,
        b_ub = b_ub,        
        bounds = bounds,
        method = "highs"
    )

    x = res.x

    value = 1 / np.sum(x) - shift

    return value

linear_value = linear(payment_matrix)

print(f"Цена игры: {linear_value:.3f} (метод линейного программирования)")

plt.show()