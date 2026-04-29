import numpy as np
import pandas as pd
from scipy.optimize import linprog

# 1 Сформируйте модель безопасности для сети из N ≥ 5 узлов (серверов).
np.random.seed(42)

nodes = 6 # V

values = np.random.randint(50, 150, nodes) # Vi
coasts = np.random.randint(5, 20, nodes) # Ci

Psucc_def = np.random.uniform(0.1, 0.4, nodes)
Psucc_no_def = np.random.uniform(0.6, 0.9, nodes)

budget = 40

df = pd.DataFrame({
    "Ценность узлов": values,
    "Стоимость защиты": coasts,
    "Вероятность успеха при защите": Psucc_def,
    "Вероятность успеха без защиты": Psucc_no_def,
})

# print(f"Ценность узлов: {values}")
# print(f"Стоимость защиты узла: {coasts}")
# print(f"Вероятность успеха атаки при защите: {Psucc_def}")
# print(f"Вероятность успеха атаки без защиты: {Psucc_no_def}")
print(df)
print(f"Бюджет: {budget}")

# 4 Реализуйте поиск равновесия Штакельберга
def stackelberg(df, budget):
    size = df.shape[0]

    num_vars = size + 1

    c = np.zeros(num_vars)
    c[-1] = 1

    A_ub = []
    b_ub = []

    for i in range(size):
        row = np.zeros(num_vars)

        V = df["Ценность узлов"][i]
        Psucc_def = df["Вероятность успеха при защите"][i]
        Psucc_no_def = df["Вероятность успеха без защиты"][i]

        row[i] = V * (Psucc_def - Psucc_no_def)

        row[-1] = -1

        A_ub.append(row)
        b_ub.append(-V * Psucc_no_def)

    C = df["Стоимость защиты"]

    budget_row = np.zeros(num_vars)
    budget_row[:size] = C

    A_ub.append(budget_row)
    b_ub.append(budget)

    bounds = [(0, 1)] * nodes + [(0, None)]

    result = linprog(
        c,
        A_ub = A_ub,
        b_ub = b_ub,
        bounds = bounds,
        method = "highs"
    )

    x = result.x[:size]
    t = result.x[-1]

    return x, t

x, t = stackelberg(df, budget)

print("Оптимальная стратегия защиты:")
for i in range(nodes):
    print(f"Узел {i}: {x[i]:.3f}")

print(f"Ожидаемые потери: {t}")

# Анализ ожидаемого ущерба (выбор цели атакующим)
losses = []

for i in range(nodes):
    V = df["Ценность узлов"][i]
    X = x[i]
    P_def = df["Вероятность успеха при защите"][i]
    P_no_def = df["Вероятность успеха без защиты"][i]

    Li = V * (X * P_def + (1 - X) * P_no_def)

    losses.append(Li)

attacked_node = np.argmax(losses)

print(f"Атакующий выберет узел: {attacked_node}")