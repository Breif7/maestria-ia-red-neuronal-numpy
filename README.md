# Red neuronal desde cero con NumPy

Proyecto de la materia **Sistemas Cognitivos Artificiales** (Maestría en IA, UNIR). Es trabajo propio: no incluye temario, notebooks ni enunciados de la universidad.

## Problema

¿Qué aporta una capa oculta? Se compara un clasificador lineal con un perceptrón multicapa en tres espirales entrelazadas, un problema que ninguna frontera recta separa.

## Método

Todo escrito a mano con NumPy, sin frameworks:

- **Arquitectura:** capas densas, activación ReLU, salida softmax e inicialización He.
- **Entrenamiento:**
  - Entropía cruzada con regularización L2.
  - Retropropagación con la derivada exacta de softmax + entropía cruzada.
  - Optimizador Adam con corrección de sesgo.
- **Verificación:** una **comprobación numérica del gradiente** (diferencias centrales) confirma que la retropropagación es correcta.

| Tema del curso | Dónde está |
| --- | --- |
| Perceptrón y redes multicapa | `MLP.forward` |
| Retropropagación | `MLP.loss_and_grads` |
| Optimización (Adam) y regularización | `MLP.fit` |

## Resultados (20-nov-2026, 300 puntos, 70 % entrenamiento / 30 % prueba)

| Modelo | Exactitud en prueba |
| --- | --- |
| Siempre la clase mayoritaria | 0.300 |
| Softmax lineal `[2, 3]` | 0.589 |
| MLP `[2, 64, 3]` | 1.000 |
| MLP `[2, 64, 64, 3]` | 1.000 |

La capa oculta resuelve el problema; la segunda no aporta nada aquí. Con ruido 0.2 las espirales casi no se solapan, por eso llega a 100 %. Subir `noise` en `spirals()` lo vuelve más difícil.

## Cómo correrlo

```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python mlp.py
pytest -q
```

No usa variables de entorno ni claves. CI corre las pruebas en cada push.

## Límites

- Entrenamiento por lotes completos en CPU; para datos reales conviene PyTorch o TensorFlow.
- Datos sintéticos en 2D: sirve para ver el mecanismo, no como referencia de rendimiento.
