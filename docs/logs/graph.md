# Graph log

The graph log contains human readable information about the constructed graph.

```python
graph_logger: INFO     =============GRAPH=============
graph_logger: INFO     | inputs/X_in     | (?, 150, 150, 3)
graph_logger: INFO     | conv_1/Elu      | (?, 75, 75, 16)
graph_logger: INFO     | pool_1/AvgPool  | (?, 37, 37, 16)
graph_logger: INFO     | conv_2/Elu      | (?, 37, 37, 32)
graph_logger: INFO     | pool_2/MaxPool  | (?, 18, 18, 32)
graph_logger: INFO     | conv_3/Elu      | (?, 18, 18, 64)
graph_logger: INFO     | pool_3/MaxPool  | (?, 9, 9, 64)
graph_logger: INFO     >> dropout: 0.5
graph_logger: INFO     >> flatten: (?, 5184)
graph_logger: INFO     | dense_1/Elu     | (?, 64)
graph_logger: INFO     >> dropout: 0.5
graph_logger: INFO     | dense_2/Elu     | (?, 16)
graph_logger: INFO     >> dropout: 0.5
graph_logger: INFO     | y_proba         | (?, 1)
graph_logger: INFO     Adam
graph_logger: INFO     ==============END==============
```