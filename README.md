# 🤖 ML Explorer - Interactive Machine Learning Platform

Una plataforma web interactiva para aprender y experimentar con algoritmos de Machine Learning, con arquitectura cliente-servidor separada.

![ML Explorer](https://img.shields.io/badge/ML-Explorer-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.0+-red?style=for-the-badge&logo=flask)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow?style=for-the-badge&logo=javascript)

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Arquitectura](#-arquitectura)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [API Endpoints](#-api-endpoints)
- [Algoritmos Implementados](#-algoritmos-implementados)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Configuración](#-configuración)
- [Solución de Problemas](#-solución-de-problemas)
- [Licencia](#-licencia)

## ✨ Características

### 🎯 **Algoritmos de Machine Learning**
- **Árboles de Decisión**: Implementación desde cero con visualización interactiva
- **Q-Learning**: Agente de aprendizaje por refuerzo para resolver laberintos
- **Dataset Iris**: Generación y clasificación de datos sintéticos
- **Visualizaciones en tiempo real**: Gráficos dinámicos del proceso de aprendizaje

### 🏗️ **Arquitectura Moderna**
- **Backend Python**: API REST con Flask para cálculos de ML
- **Frontend Interactivo**: HTML5, CSS3, JavaScript ES6+
- **Comunicación Asíncrona**: Llamadas AJAX con manejo de errores
- **Sesiones de Usuario**: Soporte para múltiples usuarios simultáneos

### 🎨 **Interfaz de Usuario**
- **Diseño Responsivo**: Adaptable a diferentes tamaños de pantalla
- **Efectos Glassmorphism**: Interfaz moderna con efectos de cristal
- **Animaciones Fluidas**: Transiciones suaves y feedback visual
- **Indicadores de Estado**: Conexión al servidor y progreso de operaciones

## 🏗️ Arquitectura
```
┌─────────────────┐    HTTP/JSON    ┌─────────────────┐
│                 │ ◄──────────────► │                 │
│   Frontend      │                 │   Backend        │
│   (HTML/JS)     │                 │   (Python)       │
│                 │                 │                  │
├─────────────────┤                 ├─────────────────┤
│ • Visualización │                 │ • ML Algorithms  │
│ • Interacciones │                 │ • Data Processing│
│ • UI/UX         │                 │ • API Endpoints  │
│ • Validaciones  │                 │ • Session Mgmt   │
└─────────────────┘                 └─────────────────┘
```

## 🚀 Instalación

### Prerrequisitos
- Python 3.8 o superior
- Navegador web moderno (Chrome, Firefox, Safari, Edge)
- Conexión a internet (para CDNs de librerías)

### Pasos de Instalación

1. **Clonar o descargar el proyecto**
   \`\`\`bash
   Si tienes git
   git clone <repository-url>
   cd ml-explorer
   
   O simplemente descargar y extraer los archivos
   \`\`\`

2. **Instalar dependencias de Python**
   \`\`\`bash
   pip install flask flask-cors numpy
   
   O usando requirements.txt
   pip install -r requirements.txt
   \`\`\`

3. **Verificar la instalación**
   \`\`\`bash
   python --version  # Debe ser 3.8+
   pip list | grep -E "(flask|numpy)"
   \`\`\`

## 🎮 Uso

### 1. Iniciar el Servidor

\`\`\`bash
python server.py
\`\`\`

Deberías ver:
\`\`\`
 * Running on http://127.0.0.1:5000
 * Debug mode: on
ML Explorer Server iniciado correctamente!
\`\`\`

### 2. Abrir la Aplicación

1. Abrir `index.html` en tu navegador web
2. Verificar que el indicador muestre: 🟢 **Conectado**
3. Si muestra 🔴 **Desconectado**, revisar que el servidor esté corriendo

### 3. Explorar los Algoritmos

#### 🌳 **Árboles de Decisión**
1. Navegar a la sección "Decision Trees"
2. Hacer clic en "Entrenar Árbol de Decisión"
3. Observar la visualización del árbol generado
4. Probar predicciones con diferentes valores

#### 🎯 **Q-Learning**
1. Ir a la sección "Reinforcement Learning"
2. Hacer clic en "Crear Agente RL"
3. Presionar "Entrenar Agente" para ver el aprendizaje
4. Usar "Mostrar Ruta Óptima" para ver el resultado

## 🔌 API Endpoints

### Árboles de Decisión

| Endpoint | Método | Descripción | Parámetros |
|----------|--------|-------------|------------|
| `/api/train-decision-tree` | POST | Entrena un árbol de decisión | `max_depth`, `min_samples` |
| `/api/predict` | POST | Hace predicción con el modelo | `sepal_length`, `sepal_width`, `petal_length`, `petal_width` |

### Q-Learning

| Endpoint | Método | Descripción | Parámetros |
|----------|--------|-------------|------------|
| `/api/create-rl-agent` | POST | Crea un nuevo agente RL | `learning_rate`, `discount_factor`, `epsilon` |
| `/api/train-rl-agent` | POST | Entrena el agente | `episodes` |
| `/api/get-optimal-path` | GET | Obtiene la ruta óptima | - |

### Utilidades

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/api/health` | GET | Estado del servidor |
| `/api/generate-data` | GET | Genera datos sintéticos |

### Ejemplos de Uso

#### Entrenar Árbol de Decisión
\`\`\`javascript
const response = await fetch('http://localhost:5000/api/train-decision-tree', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        max_depth: 3,
        min_samples: 2
    })
});
const result = await response.json();
\`\`\`

#### Hacer Predicción
\`\`\`javascript
const prediction = await fetch('http://localhost:5000/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        sepal_length: 5.1,
        sepal_width: 3.5,
        petal_length: 1.4,
        petal_width: 0.2
    })
});
\`\`\`

## 🧠 Algoritmos Implementados

### 1. Árbol de Decisión (Decision Tree)

**Implementación desde cero** con las siguientes características:

- **Criterio de División**: Entropía e Información Mutua
- **Parámetros Configurables**: Profundidad máxima, muestras mínimas
- **Prevención de Overfitting**: Poda automática
- **Visualización**: Representación gráfica del árbol

**Fórmulas utilizadas:**
- Entropía: `H(S) = -Σ(p_i * log2(p_i))`
- Información Mutua: `IG(S,A) = H(S) - Σ((|Sv|/|S|) * H(Sv))`

### 2. Q-Learning

**Algoritmo de aprendizaje por refuerzo** para resolver laberintos:

- **Ecuación de Bellman**: `Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]`
- **Exploración vs Explotación**: Estrategia ε-greedy
- **Parámetros**: Learning rate (α), Discount factor (γ), Epsilon (ε)
- **Ambiente**: Laberinto 5x5 con obstáculos y recompensas

## 📁 Estructura del Proyecto
```
ml-explorer/
│
├── server.py              # Servidor Flask con API REST
├── index.html             # Frontend principal
├── requirements.txt       # Dependencias de Python
├── README.md              # Documentación (este archivo)
│
├── assets/ (en HTML)
│   ├── styles/            # Estilos CSS embebidos
│   ├── scripts/           # JavaScript embebido
│   └── images/            # Recursos gráficos
│
└── api/                   # Endpoints del servidor
    ├── decision_tree/     # Lógica de árboles de decisión
    ├── reinforcement/     # Lógica de Q-Learning
    └── utils/             # Utilidades compartidas
```
## ⚙️ Configuración

### Configuración del Servidor

En `server.py`, puedes modificar:

\`\`\`python
# Puerto del servidor
app.run(host='127.0.0.1', port=5000, debug=True)

# Configuración de CORS
CORS(app, origins=['*'])  # Cambiar por dominios específicos en producción

# Configuración de sesiones
app.config['SECRET_KEY'] = 'tu-clave-secreta-aqui'
\`\`\`

### Parámetros de Algoritmos

#### Árbol de Decisión
\`\`\`python
def __init__(self, max_depth=3, min_samples=2):
    self.max_depth = max_depth      # Profundidad máxima del árbol
    self.min_samples = min_samples  # Muestras mínimas para dividir
\`\`\`

#### Q-Learning
\`\`\`python
def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
    self.alpha = learning_rate      # Tasa de aprendizaje
    self.gamma = discount_factor    # Factor de descuento
    self.epsilon = epsilon          # Probabilidad de exploración
\`\`\`

### Personalización del Frontend

En `index.html`, puedes modificar:

- **Colores y temas**: Variables CSS en `:root`
- **Animaciones**: Duración y efectos en las clases CSS
- **Configuración de gráficos**: Parámetros de Chart.js y D3.js

## 🐛 Solución de Problemas

### Problemas Comunes

#### 1. El servidor no inicia
**Error**: `ModuleNotFoundError: No module named 'flask'`
**Solución**:
\`\`\`bash
pip install flask flask-cors numpy
\`\`\`

#### 2. Frontend muestra "Desconectado"
**Posibles causas**:
- Servidor no está corriendo
- Puerto bloqueado por firewall
- URL incorrecta en el frontend

**Solución**:
\`\`\`bash
# Verificar que el servidor esté corriendo
curl http://localhost:5000/api/health

# Verificar puertos en uso
netstat -an | grep 5000
\`\`\`

#### 3. Errores de CORS
**Error**: `Access to fetch at 'http://localhost:5000' from origin 'null' has been blocked by CORS policy`
**Solución**: Verificar que Flask-CORS esté instalado y configurado correctamente

#### 4. Visualizaciones no aparecen
**Posibles causas**:
- CDNs no cargan (sin internet)
- Errores de JavaScript
- Datos no llegan del servidor

**Solución**:
1. Abrir DevTools (F12)
2. Revisar la consola para errores
3. Verificar la pestaña Network para requests fallidos

### Logs y Debugging

#### Habilitar logs detallados en el servidor:
\`\`\`python
import logging
logging.basicConfig(level=logging.DEBUG)
\`\`\`

#### Debugging en el frontend:
\`\`\`javascript
// Habilitar logs detallados
const DEBUG = true;
if (DEBUG) console.log('Estado actual:', data);
\`\`\`


## 📄 Licencia

Este proyecto está licenciado bajo la **MIT License** - ver el archivo [LICENSE](LICENSE) para más detalles.

### MIT License

\`\`\`
Copyright (c) 2024 ML Explorer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
\`\`\`


**¡Gracias por usar ML Explorer! 🚀**

