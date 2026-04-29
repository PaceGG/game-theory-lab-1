import numpy as np
from graphviz import Digraph

decision_tree = {
    'Быстрый релиз': {
        'Стабильная нагрузка (0.7)': 
            [('Успех', 0.7, 100), ('Частичный успех', 0.2, 50), ('Провал', 0.1, -100)],
        'Пиковая нагрузка (0.25)': 
            [('Успех', 0.6, 100), ('Частичный успех', 0.3, 50), ('Провал', 0.1, -100)],
        'Сбой зависимости (0.05)': 
            [('Успех', 0.05, 100), ('Частичный успех', 0.15, 50), ('Провал', 0.8, -100)],
    },
    'Тестирование': {
        'Стабильная нагрузка (0.7)': 
            [('Успех', 0.8, 90), ('Частичный успех', 0.15, 40), ('Провал', 0.05, -110)],
        'Пиковая нагрузка (0.25)': 
            [('Успех', 0.7, 90), ('Частичный успех', 0.25, 40), ('Провал', 0.05, -110)],
        'Сбой зависимости (0.05)': 
            [('Успех', 0.1, 90), ('Частичный успех', 0.20, 40), ('Провал', 0.7, -110)],
    },
    'Отмена': {
        'Стабильная нагрузка (0.7)': 
            [('Успех', 1/3, -50), ('Частичный успех', 1/3, -50), ('Провал', 1/3, -50)],
        'Пиковая нагрузка (0.25)': 
            [('Успех', 1/3, -50), ('Частичный успех', 1/3, -50), ('Провал', 1/3, -50)],
        'Сбой зависимости (0.05)': 
            [('Успех', 1/3, -50), ('Частичный успех', 1/3, -50), ('Провал', 1/3, -50)],
    }
}

state_probs = {
    'Стабильная нагрузка (0.7)': 0.7,
    'Пиковая нагрузка (0.25)': 0.25,
    'Сбой зависимости (0.05)': 0.05,
}

def calculate_metrics(tree, state_p):
    results = {}
    for strategy, states in tree.items():
        probabilities = []
        payoffs = []
        
        for state, outcomes in states.items():
            p_state = state_p[state]
            for outcome_name, p_outcome, payoff in outcomes:
                p_total = p_state * p_outcome 
                probabilities.append(p_total)
                payoffs.append(payoff)
                
        probabilities = np.array(probabilities)
        payoffs = np.array(payoffs)
        
        emv = np.sum(probabilities * payoffs)
        variance = np.sum(probabilities * (payoffs - emv)**2)
        sigma = np.sqrt(variance)
        
        results[strategy] = {'EMV': round(emv, 2), 'Sigma': round(sigma, 2)}
        
    return results

def visualize_tree(tree):
    dot = Digraph(comment='Древо решений', format='png')
    dot.attr(dpi='400')
    
    dot.node('Root', 'Решение\nМенеджера')
    
    for strategy, states in tree.items():
        dot.edge('Root', strategy)
        
        for state, outcomes in states.items():
            state_node = f"{strategy}_{state}"
            dot.node(state_node, state)
            dot.edge(strategy, state_node)
            
            for outcome_name, p_outcome, payoff in outcomes:
                leaf_node = f"{strategy}_{state}_{outcome_name}_{payoff}"
                
                label = f"{outcome_name}\n{'+' if payoff > 0 else ''}{payoff} ({p_outcome})"
                
                dot.node(leaf_node, label)
                dot.edge(state_node, leaf_node)
                
    dot.render('decision_tree', view=True, cleanup=True)
    print("Дерево решений сохранено в 'decision_tree.png'")

metrics = calculate_metrics(decision_tree, state_probs)
visualize_tree(decision_tree)

print(f"{'Стратегия':<15} | {'EMV (Доход)':<15} | {'σ (Риск)':<15}")
for strategy, vals in metrics.items():
    print(f"{strategy:<15} | {vals['EMV']:<15} | {vals['Sigma']:<15}")