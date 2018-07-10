# Build log

The build log contains information about the construction of the graph.

```python
[2018-07-09 20:23:31,779 - build_graph.py:36 -          build_graph()][INFO    ]: -> START building graph
[2018-07-09 20:23:31,795 - build_graph.py:51 -          build_graph()][INFO    ]: create inputs
[2018-07-09 20:23:31,801 - build_hidden.py:235 -   build_hidden_block()][INFO    ]: -> START building hidden block
[2018-07-09 20:23:31,802 - build_hidden.py:244 -   build_hidden_block()][DEBUG   ]: loop+start building layers: dict_keys(['conv_1', 'pool_1', 'conv_2', 'pool_2', 'conv_3', 'pool_3', 'dense_1', 'dense_2'])
[2018-07-09 20:23:31,802 - build_hidden.py:248 -   build_hidden_block()][DEBUG   ]: -> START building layer: conv_1 with opts: {'type': 'conv2d', 'options': {'filters': 16, 'kernel_size': 3, 'strides': 2}}
[2018-07-09 20:23:31,802 - build_hidden.py:267 -   build_hidden_block()][DEBUG   ]: activation set: <function elu at 0x7ff5dc9ce2f0>
[2018-07-09 20:23:31,802 - build_hidden.py:271 -   build_hidden_block()][DEBUG   ]: START building: conv2d
[2018-07-09 20:23:31,802 - build_hidden.py:22 -   build_conv2d_layer()][DEBUG   ]: k_init_fn set: None
[2018-07-09 20:23:31,802 - build_hidden.py:28 -   build_conv2d_layer()][DEBUG   ]: k_reg set: None
[2018-07-09 20:23:31,802 - build_hidden.py:34 -   build_conv2d_layer()][DEBUG   ]: b_reg set: None
[2018-07-09 20:23:31,802 - build_hidden.py:41 -   build_conv2d_layer()][DEBUG   ]: kernel_size set: 3
[2018-07-09 20:23:31,802 - build_hidden.py:48 -   build_conv2d_layer()][DEBUG   ]: padding set: SAME
[2018-07-09 20:23:31,802 - build_hidden.py:54 -   build_conv2d_layer()][DEBUG   ]: strides set: 2
[2018-07-09 20:23:31,802 - build_hidden.py:58 -   build_conv2d_layer()][DEBUG   ]: name set: conv_1
[2018-07-09 20:23:31,802 - build_hidden.py:65 -   build_conv2d_layer()][DEBUG   ]: trainable set: True
[2018-07-09 20:23:31,824 - build_hidden.py:81 -   build_conv2d_layer()][DEBUG   ]: Final tensor obj: Tensor("conv_1/Elu:0", shape=(?, 75, 75, 16), dtype=float32)
[2018-07-09 20:23:31,825 - build_hidden.py:84 -   build_conv2d_layer()][DEBUG   ]: [End] building: conv_1
[2018-07-09 20:23:31,825 - build_hidden.py:248 -   build_hidden_block()][DEBUG   ]: -> START building layer: pool_1 with opts: {'type': 'pooling2d', 'options': {'pool_type': 'avg'}}
[2018-07-09 20:23:31,825 - build_hidden.py:267 -   build_hidden_block()][DEBUG   ]: activation set: <function elu at 0x7ff5dc9ce2f0>
[2018-07-09 20:23:31,825 - build_hidden.py:294 -   build_hidden_block()][DEBUG   ]: -> START building: pooling2d
[2018-07-09 20:23:31,825 - build_hidden.py:165 -     build_pool_layer()][DEBUG   ]: pool_size set: [2, 2]
[2018-07-09 20:23:31,825 - build_hidden.py:174 -     build_pool_layer()][DEBUG   ]: strides set: 2
[2018-07-09 20:23:31,825 - build_hidden.py:178 -     build_pool_layer()][DEBUG   ]: name set: pool_1
[2018-07-09 20:23:31,825 - build_hidden.py:189 -     build_pool_layer()][DEBUG   ]: pool_type set: avg
[2018-07-09 20:23:31,827 - build_hidden.py:203 -     build_pool_layer()][DEBUG   ]: tensor obj pre dropout: Tensor("pool_1/AvgPool:0", shape=(?, 37, 37, 16), dtype=float32)
[2018-07-09 20:23:31,827 - build_hidden.py:214 -     build_pool_layer()][DEBUG   ]: dropout_rate set: None
[2018-07-09 20:23:31,827 - build_hidden.py:228 -     build_pool_layer()][DEBUG   ]: [End] building: pool_1
[2018-07-09 20:23:31,827 - build_hidden.py:248 -   build_hidden_block()][DEBUG   ]: -> START building layer: conv_2 with opts: {'type': 'conv2d', 'options': {'filters': 32, 'kernel_size': 3, 'strides': 1}}
[2018-07-09 20:23:31,827 - build_hidden.py:267 -   build_hidden_block()][DEBUG   ]: activation set: <function elu at 0x7ff5dc9ce2f0>
[2018-07-09 20:23:31,827 - build_hidden.py:271 -   build_hidden_block()][DEBUG   ]: START building: conv2d
[2018-07-09 20:23:31,827 - build_hidden.py:22 -   build_conv2d_layer()][DEBUG   ]: k_init_fn set: None
[2018-07-09 20:23:31,827 - build_hidden.py:28 -   build_conv2d_layer()][DEBUG   ]: k_reg set: None
[2018-07-09 20:23:31,827 - build_hidden.py:34 -   build_conv2d_layer()][DEBUG   ]: b_reg set: None
[2018-07-09 20:23:31,827 - build_hidden.py:41 -   build_conv2d_layer()][DEBUG   ]: kernel_size set: 3
[2018-07-09 20:23:31,827 - build_hidden.py:48 -   build_conv2d_layer()][DEBUG   ]: padding set: SAME
[2018-07-09 20:23:31,827 - build_hidden.py:54 -   build_conv2d_layer()][DEBUG   ]: strides set: 1
[2018-07-09 20:23:31,827 - build_hidden.py:58 -   build_conv2d_layer()][DEBUG   ]: name set: conv_2
[2018-07-09 20:23:31,827 - build_hidden.py:65 -   build_conv2d_layer()][DEBUG   ]: trainable set: True
[2018-07-09 20:23:31,843 - build_hidden.py:81 -   build_conv2d_layer()][DEBUG   ]: Final tensor obj: Tensor("conv_2/Elu:0", shape=(?, 37, 37, 32), dtype=float32)
[2018-07-09 20:23:31,843 - build_hidden.py:84 -   build_conv2d_layer()][DEBUG   ]: [End] building: conv_2
[2018-07-09 20:23:31,843 - build_hidden.py:248 -   build_hidden_block()][DEBUG   ]: -> START building layer: pool_2 with opts: {'type': 'pooling2d'}
[2018-07-09 20:23:31,843 - build_hidden.py:267 -   build_hidden_block()][DEBUG   ]: activation set: <function elu at 0x7ff5dc9ce2f0>
[2018-07-09 20:23:31,843 - build_hidden.py:294 -   build_hidden_block()][DEBUG   ]: -> START building: pooling2d
[2018-07-09 20:23:31,843 - build_hidden.py:165 -     build_pool_layer()][DEBUG   ]: pool_size set: [2, 2]
[2018-07-09 20:23:31,843 - build_hidden.py:174 -     build_pool_layer()][DEBUG   ]: strides set: 2
[2018-07-09 20:23:31,843 - build_hidden.py:178 -     build_pool_layer()][DEBUG   ]: name set: pool_2
[2018-07-09 20:23:31,843 - build_hidden.py:189 -     build_pool_layer()][DEBUG   ]: pool_type set: max
[2018-07-09 20:23:31,845 - build_hidden.py:203 -     build_pool_layer()][DEBUG   ]: tensor obj pre dropout: Tensor("pool_2/MaxPool:0", shape=(?, 18, 18, 32), dtype=float32)
[2018-07-09 20:23:31,845 - build_hidden.py:214 -     build_pool_layer()][DEBUG   ]: dropout_rate set: None
[2018-07-09 20:23:31,845 - build_hidden.py:228 -     build_pool_layer()][DEBUG   ]: [End] building: pool_2
[2018-07-09 20:23:31,845 - build_hidden.py:248 -   build_hidden_block()][DEBUG   ]: -> START building layer: conv_3 with opts: {'type': 'conv2d', 'options': {'filters': 64, 'kernel_size': 3, 'strides': 1}}
[2018-07-09 20:23:31,845 - build_hidden.py:267 -   build_hidden_block()][DEBUG   ]: activation set: <function elu at 0x7ff5dc9ce2f0>
[2018-07-09 20:23:31,845 - build_hidden.py:271 -   build_hidden_block()][DEBUG   ]: START building: conv2d
[2018-07-09 20:23:31,846 - build_hidden.py:22 -   build_conv2d_layer()][DEBUG   ]: k_init_fn set: None
[2018-07-09 20:23:31,846 - build_hidden.py:28 -   build_conv2d_layer()][DEBUG   ]: k_reg set: None
[2018-07-09 20:23:31,846 - build_hidden.py:34 -   build_conv2d_layer()][DEBUG   ]: b_reg set: None
[2018-07-09 20:23:31,846 - build_hidden.py:41 -   build_conv2d_layer()][DEBUG   ]: kernel_size set: 3
[2018-07-09 20:23:31,846 - build_hidden.py:48 -   build_conv2d_layer()][DEBUG   ]: padding set: SAME
[2018-07-09 20:23:31,846 - build_hidden.py:54 -   build_conv2d_layer()][DEBUG   ]: strides set: 1
[2018-07-09 20:23:31,846 - build_hidden.py:58 -   build_conv2d_layer()][DEBUG   ]: name set: conv_3
[2018-07-09 20:23:31,846 - build_hidden.py:65 -   build_conv2d_layer()][DEBUG   ]: trainable set: True
[2018-07-09 20:23:31,862 - build_hidden.py:81 -   build_conv2d_layer()][DEBUG   ]: Final tensor obj: Tensor("conv_3/Elu:0", shape=(?, 18, 18, 64), dtype=float32)
[2018-07-09 20:23:31,863 - build_hidden.py:84 -   build_conv2d_layer()][DEBUG   ]: [End] building: conv_3
[2018-07-09 20:23:31,863 - build_hidden.py:248 -   build_hidden_block()][DEBUG   ]: -> START building layer: pool_3 with opts: {'type': 'pooling2d', 'options': {'pool_type': 'max', 'dropout': 0.5}}
[2018-07-09 20:23:31,863 - build_hidden.py:267 -   build_hidden_block()][DEBUG   ]: activation set: <function elu at 0x7ff5dc9ce2f0>
[2018-07-09 20:23:31,863 - build_hidden.py:294 -   build_hidden_block()][DEBUG   ]: -> START building: pooling2d
[2018-07-09 20:23:31,863 - build_hidden.py:165 -     build_pool_layer()][DEBUG   ]: pool_size set: [2, 2]
[2018-07-09 20:23:31,863 - build_hidden.py:174 -     build_pool_layer()][DEBUG   ]: strides set: 2
[2018-07-09 20:23:31,863 - build_hidden.py:178 -     build_pool_layer()][DEBUG   ]: name set: pool_3
[2018-07-09 20:23:31,863 - build_hidden.py:189 -     build_pool_layer()][DEBUG   ]: pool_type set: max
[2018-07-09 20:23:31,865 - build_hidden.py:203 -     build_pool_layer()][DEBUG   ]: tensor obj pre dropout: Tensor("pool_3/MaxPool:0", shape=(?, 9, 9, 64), dtype=float32)
[2018-07-09 20:23:31,865 - build_hidden.py:214 -     build_pool_layer()][DEBUG   ]: dropout_rate set: 0.5
[2018-07-09 20:23:31,885 - build_hidden.py:225 -     build_pool_layer()][DEBUG   ]: tensor obj post dropout: Tensor("dropout/cond/Merge:0", shape=(?, 9, 9, 64), dtype=float32)
[2018-07-09 20:23:31,885 - build_hidden.py:228 -     build_pool_layer()][DEBUG   ]: [End] building: pool_3
[2018-07-09 20:23:31,885 - build_hidden.py:248 -   build_hidden_block()][DEBUG   ]: -> START building layer: dense_1 with opts: {'type': 'dense', 'options': {'units': 64, 'dropout': 0.5}}
[2018-07-09 20:23:31,885 - build_hidden.py:267 -   build_hidden_block()][DEBUG   ]: activation set: <function elu at 0x7ff5dc9ce2f0>
[2018-07-09 20:23:31,885 - build_hidden.py:276 -   build_hidden_block()][DEBUG   ]: -> START building: dense
[2018-07-09 20:23:31,887 - build_hidden.py:287 -   build_hidden_block()][DEBUG   ]: reshaped tensor: Tensor("Reshape:0", shape=(?, 5184), dtype=float32)
[2018-07-09 20:23:31,888 - build_hidden.py:92 -    build_dense_layer()][DEBUG   ]: units set: 64
[2018-07-09 20:23:31,888 - build_hidden.py:98 -    build_dense_layer()][DEBUG   ]: k_init_fn set: None
[2018-07-09 20:23:31,888 - build_hidden.py:104 -    build_dense_layer()][DEBUG   ]: k_reg set: None
[2018-07-09 20:23:31,888 - build_hidden.py:110 -    build_dense_layer()][DEBUG   ]: b_reg set: None
[2018-07-09 20:23:31,888 - build_hidden.py:117 -    build_dense_layer()][DEBUG   ]: trainable set: True
[2018-07-09 20:23:31,905 - build_hidden.py:130 -    build_dense_layer()][DEBUG   ]: tensor obj pre dropout: Tensor("dense_1/Elu:0", shape=(?, 64), dtype=float32)
[2018-07-09 20:23:31,906 - build_hidden.py:138 -    build_dense_layer()][DEBUG   ]: dropout_rate set: 0.5
[2018-07-09 20:23:31,922 - build_hidden.py:149 -    build_dense_layer()][DEBUG   ]: tensor obj post dropout: Tensor("dropout_1/cond/Merge:0", shape=(?, 64), dtype=float32)
[2018-07-09 20:23:31,923 - build_hidden.py:152 -    build_dense_layer()][DEBUG   ]: [End] building: dense_1
[2018-07-09 20:23:31,923 - build_hidden.py:248 -   build_hidden_block()][DEBUG   ]: -> START building layer: dense_2 with opts: {'type': 'dense', 'options': {'units': 16, 'dropout': 0.5}}
[2018-07-09 20:23:31,923 - build_hidden.py:267 -   build_hidden_block()][DEBUG   ]: activation set: <function elu at 0x7ff5dc9ce2f0>
[2018-07-09 20:23:31,923 - build_hidden.py:276 -   build_hidden_block()][DEBUG   ]: -> START building: dense
[2018-07-09 20:23:31,923 - build_hidden.py:92 -    build_dense_layer()][DEBUG   ]: units set: 16
[2018-07-09 20:23:31,923 - build_hidden.py:98 -    build_dense_layer()][DEBUG   ]: k_init_fn set: None
[2018-07-09 20:23:31,923 - build_hidden.py:104 -    build_dense_layer()][DEBUG   ]: k_reg set: None
[2018-07-09 20:23:31,923 - build_hidden.py:110 -    build_dense_layer()][DEBUG   ]: b_reg set: None
[2018-07-09 20:23:31,923 - build_hidden.py:117 -    build_dense_layer()][DEBUG   ]: trainable set: True
[2018-07-09 20:23:31,937 - build_hidden.py:130 -    build_dense_layer()][DEBUG   ]: tensor obj pre dropout: Tensor("dense_2/Elu:0", shape=(?, 16), dtype=float32)
[2018-07-09 20:23:31,938 - build_hidden.py:138 -    build_dense_layer()][DEBUG   ]: dropout_rate set: 0.5
[2018-07-09 20:23:31,951 - build_hidden.py:149 -    build_dense_layer()][DEBUG   ]: tensor obj post dropout: Tensor("dropout_2/cond/Merge:0", shape=(?, 16), dtype=float32)
[2018-07-09 20:23:31,951 - build_hidden.py:152 -    build_dense_layer()][DEBUG   ]: [End] building: dense_2
[2018-07-09 20:23:31,951 - build_hidden.py:302 -   build_hidden_block()][INFO    ]: [END] building hidden block
[2018-07-09 20:23:31,963 - build_graph.py:96 -          build_graph()][DEBUG   ]: pred created as sigmoid: Tensor("y_proba:0", shape=(?, 1), dtype=float32)
[2018-07-09 20:23:31,963 - build_graph.py:102 -          build_graph()][INFO    ]: create /loss
[2018-07-09 20:23:31,969 - build_graph.py:124 -          build_graph()][DEBUG   ]: xentropy created as sigmoid: Tensor("loss/logistic_loss:0", shape=(?, 1), dtype=float32)
[2018-07-09 20:23:31,971 - build_graph.py:138 -          build_graph()][INFO    ]: create /train
[2018-07-09 20:23:32,211 - build_graph.py:146 -          build_graph()][INFO    ]: create /init
[2018-07-09 20:23:32,211 - build_graph.py:152 -          build_graph()][INFO    ]: create /metrics
[2018-07-09 20:23:32,211 - build_graph.py:155 -          build_graph()][DEBUG   ]: create /metrics/common
[2018-07-09 20:23:32,215 - build_graph.py:168 -          build_graph()][DEBUG   ]: create /metrics/train_metrics
[2018-07-09 20:23:34,309 - build_graph.py:182 -          build_graph()][DEBUG   ]: create /metrics/val_metrics
[2018-07-09 20:23:34,400 - build_graph.py:194 -          build_graph()][DEBUG   ]: create /metrics/test_metrics
[2018-07-09 20:23:34,494 - build_graph.py:208 -          build_graph()][DEBUG   ]: create /metrics/train_loss_eval
[2018-07-09 20:23:34,504 - build_graph.py:217 -          build_graph()][DEBUG   ]: create /metrics/val_loss_eval
[2018-07-09 20:23:34,514 - build_graph.py:226 -          build_graph()][DEBUG   ]: create /metrics/test_loss_eval
[2018-07-09 20:23:34,524 - build_graph.py:282 -          build_graph()][DEBUG   ]: create scalar weights: [<tf.Variable 'conv_1/kernel:0' shape=(3, 3, 3, 16) dtype=float32_ref>, <tf.Variable 'conv_2/kernel:0' shape=(3, 3, 16, 32) dtype=float32_ref>, <tf.Variable 'conv_3/kernel:0' shape=(3, 3, 32, 64) dtype=float32_ref>, <tf.Variable 'dense_1/kernel:0' shape=(5184, 64) dtype=float32_ref>, <tf.Variable 'dense_2/kernel:0' shape=(64, 16) dtype=float32_ref>, <tf.Variable 'logits/kernel:0' shape=(16, 1) dtype=float32_ref>]
[2018-07-09 20:23:34,524 - build_graph.py:288 -          build_graph()][DEBUG   ]: create scalar bias: [<tf.Variable 'conv_1/kernel:0' shape=(3, 3, 3, 16) dtype=float32_ref>, <tf.Variable 'conv_2/kernel:0' shape=(3, 3, 16, 32) dtype=float32_ref>, <tf.Variable 'conv_3/kernel:0' shape=(3, 3, 32, 64) dtype=float32_ref>, <tf.Variable 'dense_1/kernel:0' shape=(5184, 64) dtype=float32_ref>, <tf.Variable 'dense_2/kernel:0' shape=(64, 16) dtype=float32_ref>, <tf.Variable 'logits/kernel:0' shape=(16, 1) dtype=float32_ref>]
[2018-07-09 20:23:34,524 - build_graph.py:290 -          build_graph()][DEBUG   ]: len(weights) == len(bias) = 6
[2018-07-09 20:23:34,534 - build_graph.py:311 -          build_graph()][DEBUG   ]: [<tf.Tensor 'conv_1_params/weights:0' shape=() dtype=string>, <tf.Tensor 'conv_1_params/bias:0' shape=() dtype=string>, <tf.Tensor 'conv_2_params/weights:0' shape=() dtype=string>, <tf.Tensor 'conv_2_params/bias:0' shape=() dtype=string>, <tf.Tensor 'conv_3_params/weights:0' shape=() dtype=string>, <tf.Tensor 'conv_3_params/bias:0' shape=() dtype=string>, <tf.Tensor 'dense_1_params/weights:0' shape=() dtype=string>, <tf.Tensor 'dense_1_params/bias:0' shape=() dtype=string>, <tf.Tensor 'dense_2_params/weights:0' shape=() dtype=string>, <tf.Tensor 'dense_2_params/bias:0' shape=() dtype=string>, <tf.Tensor 'logits_params/weights:0' shape=() dtype=string>, <tf.Tensor 'logits_params/bias:0' shape=() dtype=string>] hist opts written
[2018-07-09 20:23:34,538 - build_graph.py:339 -          build_graph()][INFO    ]: [END] building graph

```