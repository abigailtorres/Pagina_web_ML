from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import random
from collections import defaultdict, Counter
import json
import time

app = Flask(__name__)
CORS(app)  # Permitir requests desde el frontend

# Variables globales para mantener estado
decision_trees = {}
rl_agents = {}

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
        self.samples = []
        self.entropy = 0
        self.gain = 0
        self.depth = 0
        self.id = ""
        
    def is_leaf(self):
        return self.prediction is not None
    
    def to_dict(self):
        """Convertir nodo a diccionario para JSON"""
        return {
            'feature': self.feature,
            'threshold': self.threshold,
            'prediction': self.prediction,
            'entropy': self.entropy,
            'gain': self.gain,
            'depth': self.depth,
            'id': self.id,
            'isLeaf': self.is_leaf(),
            'samples_count': len(self.samples),
            'left': self.left.to_dict() if self.left else None,
            'right': self.right.to_dict() if self.right else None
        }

class DecisionTreeClassifier:
    def __init__(self, max_depth=3, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None
        self.feature_names = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']
        self.class_names = ['Setosa', 'Versicolor', 'Virginica']
    
    def calculate_entropy(self, labels):
        """Calcular entropía de un conjunto de etiquetas"""
        if len(labels) == 0:
            return 0
        
        counter = Counter(labels)
        total = len(labels)
        entropy = 0
        
        for count in counter.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def find_best_split(self, samples):
        """Encontrar la mejor división para un conjunto de muestras"""
        best_gain = 0
        best_feature = ''
        best_threshold = 0
        
        for feature_idx, feature in enumerate(self.feature_names):
            values = [sample[feature_idx] for sample in samples]
            unique_values = sorted(set(values))
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                left_samples = [s for s in samples if s[feature_idx] <= threshold]
                right_samples = [s for s in samples if s[feature_idx] > threshold]
                
                if len(left_samples) == 0 or len(right_samples) == 0:
                    continue
                
                # Calcular ganancia de información
                total_entropy = self.calculate_entropy([s[-1] for s in samples])
                left_weight = len(left_samples) / len(samples)
                right_weight = len(right_samples) / len(samples)
                
                weighted_entropy = (left_weight * self.calculate_entropy([s[-1] for s in left_samples]) +
                                  right_weight * self.calculate_entropy([s[-1] for s in right_samples]))
                
                information_gain = total_entropy - weighted_entropy
                
                if information_gain > best_gain:
                    best_gain = information_gain
                    best_feature = feature
                    best_threshold = threshold
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'gain': best_gain
        }
    
    def get_most_common_class(self, samples):
        """Obtener la clase más común en un conjunto de muestras"""
        labels = [sample[-1] for sample in samples]
        counter = Counter(labels)
        return counter.most_common(1)[0][0]
    
    def build_tree(self, samples, depth=0, node_id='root'):
        """Construir el árbol de decisión recursivamente"""
        node = DecisionTreeNode()
        node.samples = samples
        node.entropy = self.calculate_entropy([s[-1] for s in samples])
        node.depth = depth
        node.id = node_id
        
        # Condiciones de parada
        unique_classes = set(sample[-1] for sample in samples)
        if (depth >= self.max_depth or 
            len(unique_classes) == 1 or 
            len(samples) < self.min_samples):
            
            node.prediction = self.get_most_common_class(samples)
            return node
        
        # Encontrar la mejor división
        split = self.find_best_split(samples)
        
        if split['gain'] == 0:
            node.prediction = self.get_most_common_class(samples)
            return node
        
        # Crear nodo interno
        node.feature = split['feature']
        node.threshold = split['threshold']
        node.gain = split['gain']
        
        # Dividir muestras
        feature_idx = self.feature_names.index(split['feature'])
        left_samples = [s for s in samples if s[feature_idx] <= split['threshold']]
        right_samples = [s for s in samples if s[feature_idx] > split['threshold']]
        
        # Construir subárboles
        node.left = self.build_tree(left_samples, depth + 1, f"{node_id}-left")
        node.right = self.build_tree(right_samples, depth + 1, f"{node_id}-right")
        
        return node
    
    def fit(self, X, y):
        """Entrenar el árbol de decisión"""
        # Combinar características y etiquetas
        samples = []
        for i in range(len(X)):
            sample = list(X[i]) + [y[i]]
            samples.append(sample)
        
        self.root = self.build_tree(samples)
        return self
    
    def predict_sample(self, sample, node=None):
        """Predecir una sola muestra"""
        if node is None:
            node = self.root
        
        if node.is_leaf():
            return node.prediction
        
        feature_idx = self.feature_names.index(node.feature)
        if sample[feature_idx] <= node.threshold:
            return self.predict_sample(sample, node.left)
        else:
            return self.predict_sample(sample, node.right)
    
    def get_decision_path(self, sample, node=None, path=None):
        """Obtener el camino de decisión para una muestra"""
        if path is None:
            path = []
        if node is None:
            node = self.root
        
        path.append(node.to_dict())
        
        if node.is_leaf():
            return path
        
        feature_idx = self.feature_names.index(node.feature)
        if sample[feature_idx] <= node.threshold:
            return self.get_decision_path(sample, node.left, path)
        else:
            return self.get_decision_path(sample, node.right, path)

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.episode_rewards = []
        self.current_episode = 0
    
    def choose_action(self, state):
        """Elegir acción usando estrategia epsilon-greedy"""
        if random.random() < self.exploration_rate:
            return random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            if not q_values:
                return random.choice(self.actions)
            return max(q_values, key=q_values.get)
    
    def update_q_value(self, state, action, reward, next_state):
        """Actualizar valor Q usando la ecuación de Bellman"""
        current_q = self.q_table[state][action]
        next_q_values = self.q_table[next_state]
        max_next_q = max(next_q_values.values()) if next_q_values else 0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state][action] = new_q
    
    def get_policy(self):
        """Obtener la política óptima"""
        policy = {}
        for state in self.q_table:
            q_values = self.q_table[state]
            if q_values:
                policy[state] = max(q_values, key=q_values.get)
        return policy

class MazeEnvironment:
    def __init__(self):
        self.maze = [
            ['S', '.', '.', '#', '.'],
            ['#', '#', '.', '#', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '#', '#', '#', 'G']
        ]
        self.start_pos = (0, 0)
        self.goal_pos = (3, 4)
        self.current_pos = self.start_pos
        self.actions = ['up', 'down', 'left', 'right']
    
    def reset(self):
        """Reiniciar el entorno"""
        self.current_pos = self.start_pos
        return self.current_pos
    
    def is_valid_position(self, pos):
        """Verificar si una posición es válida"""
        row, col = pos
        return (0 <= row < len(self.maze) and 
                0 <= col < len(self.maze[0]) and 
                self.maze[row][col] != '#')
    
    def get_next_position(self, pos, action):
        """Obtener la siguiente posición dada una acción"""
        row, col = pos
        moves = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        delta_row, delta_col = moves[action]
        return (row + delta_row, col + delta_col)
    
    def step(self, action):
        """Ejecutar una acción en el entorno"""
        next_pos = self.get_next_position(self.current_pos, action)
        
        if self.is_valid_position(next_pos):
            self.current_pos = next_pos
            
            if self.current_pos == self.goal_pos:
                reward = 10
                done = True
            else:
                reward = -0.1
                done = False
        else:
            reward = -1
            done = False
        
        return self.current_pos, reward, done

# Funciones auxiliares
def generate_iris_data():
    """Generar datos sintéticos del dataset Iris"""
    species_data = [
        {
            'name': 'Setosa',
            'ranges': {
                'sepalLength': [4.3, 5.8],
                'sepalWidth': [2.3, 4.4],
                'petalLength': [1.0, 1.9],
                'petalWidth': [0.1, 0.6]
            }
        },
        {
            'name': 'Versicolor',
            'ranges': {
                'sepalLength': [4.9, 7.0],
                'sepalWidth': [2.0, 3.4],
                'petalLength': [3.0, 5.1],
                'petalWidth': [1.0, 1.8]
            }
        },
        {
            'name': 'Virginica',
            'ranges': {
                'sepalLength': [4.9, 7.9],
                'sepalWidth': [2.2, 3.8],
                'petalLength': [4.5, 6.9],
                'petalWidth': [1.4, 2.5]
            }
        }
    ]
    
    data = []
    for species_idx, species in enumerate(species_data):
        for _ in range(50):
            sample = []
            for feature in ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth']:
                min_val, max_val = species['ranges'][feature]
                value = random.uniform(min_val, max_val)
                sample.append(value)
            
            data.append({
                'features': sample,
                'label': species['name'],
                'label_idx': species_idx
            })
    
    random.shuffle(data)
    return data

# Endpoints de la API

@app.route('/api/generate-data', methods=['POST'])
def generate_data():
    """Generar nuevos datos de entrenamiento"""
    try:
        data = generate_iris_data()
        return jsonify({
            'success': True,
            'data': data
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/train-decision-tree', methods=['POST'])
def train_decision_tree():
    """Entrenar un árbol de decisión"""
    try:
        request_data = request.get_json()
        max_depth = request_data.get('max_depth', 3)
        session_id = request_data.get('session_id', 'default')
        
        # Generar datos de entrenamiento
        iris_data = generate_iris_data()
        X = [item['features'] for item in iris_data]
        y = [item['label'] for item in iris_data]
        
        # Crear y entrenar el árbol
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(X, y)
        
        # Guardar el árbol en memoria
        decision_trees[session_id] = tree
        
        return jsonify({
            'success': True,
            'tree': tree.root.to_dict() if tree.root else None,
            'session_id': session_id
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Hacer predicción con el árbol entrenado"""
    try:
        request_data = request.get_json()
        session_id = request_data.get('session_id', 'default')
        features = request_data.get('features')
        
        if session_id not in decision_trees:
            return jsonify({
                'success': False,
                'error': 'No trained tree found for this session'
            }), 400
        
        tree = decision_trees[session_id]
        prediction = tree.predict_sample(features)
        decision_path = tree.get_decision_path(features)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'decision_path': decision_path
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/create-rl-agent', methods=['POST'])
def create_rl_agent():
    """Crear un nuevo agente de Q-Learning"""
    try:
        request_data = request.get_json()
        session_id = request_data.get('session_id', 'default')
        learning_rate = request_data.get('learning_rate', 0.1)
        discount_factor = request_data.get('discount_factor', 0.9)
        exploration_rate = request_data.get('exploration_rate', 0.1)
        
        actions = ['up', 'down', 'left', 'right']
        agent = QLearningAgent(
            actions=actions,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            exploration_rate=exploration_rate
        )
        
        rl_agents[session_id] = {
            'agent': agent,
            'environment': MazeEnvironment()
        }
        
        return jsonify({
            'success': True,
            'session_id': session_id
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/train-rl-agent', methods=['POST'])
def train_rl_agent():
    """Entrenar el agente de Q-Learning"""
    try:
        request_data = request.get_json()
        session_id = request_data.get('session_id', 'default')
        num_episodes = request_data.get('num_episodes', 100)
        
        if session_id not in rl_agents:
            return jsonify({
                'success': False,
                'error': 'No RL agent found for this session'
            }), 400
        
        agent_data = rl_agents[session_id]
        agent = agent_data['agent']
        env = agent_data['environment']
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            steps = 0
            max_steps = 100
            
            while steps < max_steps:
                state_key = f"{state[0]},{state[1]}"
                action = agent.choose_action(state_key)
                next_state, reward, done = env.step(action)
                next_state_key = f"{next_state[0]},{next_state[1]}"
                
                agent.update_q_value(state_key, action, reward, next_state_key)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            agent.episode_rewards.append(episode_reward)
            agent.current_episode = episode + 1
            
            # Decay exploration rate
            agent.exploration_rate = max(0.01, agent.exploration_rate * 0.995)
        
        return jsonify({
            'success': True,
            'episode_rewards': episode_rewards,
            'q_table': dict(agent.q_table),
            'final_exploration_rate': agent.exploration_rate
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/get-optimal-path', methods=['POST'])
def get_optimal_path():
    """Obtener la ruta óptima usando la política aprendida"""
    try:
        request_data = request.get_json()
        session_id = request_data.get('session_id', 'default')
        
        if session_id not in rl_agents:
            return jsonify({
                'success': False,
                'error': 'No RL agent found for this session'
            }), 400
        
        agent_data = rl_agents[session_id]
        agent = agent_data['agent']
        env = agent_data['environment']
        
        # Encontrar ruta óptima
        state = env.reset()
        path = [state]
        visited = set()
        
        while len(path) < 50:
            state_key = f"{state[0]},{state[1]}"
            
            if state_key in visited:
                break
            visited.add(state_key)
            
            action = agent.choose_action(state_key)  # Sin exploración
            next_state, reward, done = env.step(action)
            
            state = next_state
            path.append(state)
            
            if done:
                break
        
        return jsonify({
            'success': True,
            'path': path,
            'steps': len(path) - 1
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/reset-rl-agent', methods=['POST'])
def reset_rl_agent():
    """Reiniciar el agente de Q-Learning"""
    try:
        request_data = request.get_json()
        session_id = request_data.get('session_id', 'default')
        
        if session_id in rl_agents:
            del rl_agents[session_id]
        
        return jsonify({
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Iniciando servidor ML Explorer...")
    print("📊 Endpoints disponibles:")
    print("  - POST /api/generate-data")
    print("  - POST /api/train-decision-tree")
    print("  - POST /api/predict")
    print("  - POST /api/create-rl-agent")
    print("  - POST /api/train-rl-agent")
    print("  - POST /api/get-optimal-path")
    print("  - POST /api/reset-rl-agent")
    print("\n🌐 Servidor corriendo en http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
