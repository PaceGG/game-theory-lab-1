import numpy as np

def get_consistency_ratio(matrix):
    n = matrix.shape[0]
    if n <= 2:
        return 0, 0, np.ones(n) / n
    
    ri_table = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_eigenvalue = np.real(eigenvalues.max())
    priority_vector = np.real(eigenvectors[:, eigenvalues.argmax()])
    priority_vector = priority_vector / priority_vector.sum()
    
    ci = (max_eigenvalue - n) / (n - 1)
    ri = ri_table.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0
    
    return cr, ci, priority_vector

def input_comparison_matrix(items, context_name):
    n = len(items)
    while True:
        print(f"\n--- Заполнение матрицы сравнения для: {context_name} ---")
        print("Используйте шкалу Саати (1-9). Если A важнее B на 3, то B относительно A будет 1/3.")
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                val = input(f"Сравните '{items[i]}' относительно '{items[j]}' (1-9 или дробь): ")
                try:
                    score = float(eval(val))
                    matrix[i, j] = score
                    matrix[j, i] = 1 / score
                except:
                    print("Ошибка ввода. Используйте числа или дроби (например, 1/3).")
                    return input_comparison_matrix(items, context_name)
        
        cr, ci, weights = get_consistency_ratio(matrix)
        print(f"CI: {ci:.4f}, CR: {cr:.4f}")
        
        if cr > 0.1:
            print("Внимание: CR > 0.1. Суждения несогласованы! Повторите ввод матрицы.")
        else:
            return weights

def main():
    print("Метод анализа иерархий (AHP) для выбора СУБД")
    
    # 1-3. Определение цели, альтернатив и критериев
    goal = "Выбор оптимальной СУБД для микросервиса"
    alternatives = ["PostgreSQL", "MongoDB", "Redis", "Cassandra", "MySQL", "DynamoDB"]
    criteria = ["Производительность", "Стоимость", "Масштабируемость", "Простота поддержки"]
    
    # 4-6. Сравнение критериев
    print(f"\nЦель: {goal}")
    criteria_weights = input_comparison_matrix(criteria, "Важность критериев")
    
    # Сравнение альтернатив по каждому критерию
    alt_weights_matrix = np.zeros((len(criteria), len(alternatives)))
    
    for idx, criterion in enumerate(criteria):
        weights = input_comparison_matrix(alternatives, f"Альтернативы по критерию '{criterion}'")
        alt_weights_matrix[idx] = weights
    
    # 7. Расчет глобальных приоритетов
    global_scores = np.dot(criteria_weights, alt_weights_matrix)
    
    # Вывод результатов
    print("\nЛокальные веса критериев:")
    for c, w in zip(criteria, criteria_weights):
        print(f"{c}: {w:.4f}")
        
    results = sorted(zip(alternatives, global_scores), key=lambda x: x[1], reverse=True)
    
    print("\nГлобальный рейтинг альтернатив:")
    for rank, (alt, score) in enumerate(results, 1):
        print(f"{rank}. {alt}: {score:.4f}")
    
    print(f"\nРекомендуемая технология: {results[0][0]}")

if __name__ == "__main__":
    main()