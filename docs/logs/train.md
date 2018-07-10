# Train log

The train log contains information about the training of graph (epoch + validation information).

```python
[2018-07-09 20:23:34,538 - train_graph.py:15 -          train_graph()][INFO    ]: -> START training graph
[2018-07-09 20:23:36,197 - train_graph.py:96 -          train_graph()][DEBUG   ]: trace level set: None
[2018-07-09 20:23:36,197 - train_graph.py:100 -          train_graph()][INFO    ]: -> START epoch num: 1
[2018-07-09 20:23:36,238 - train_graph.py:110 -          train_graph()][DEBUG   ]: reset train and validation metric accumulators: [<tf.Operation 'metrics/val_metrics/val_met_reset_op' type=NoOp>, <tf.Operation 'metrics/val_loss_eval/val_loss_reset_op' type=NoOp>, <tf.Operation 'metrics/train_metrics/train_met_reset_op' type=NoOp>, <tf.Operation 'metrics/train_loss_eval/train_loss_reset_op' type=NoOp>]
[2018-07-09 20:23:36,254 - train_graph.py:118 -          train_graph()][DEBUG   ]: reinitialize training iterator: ./examples/cats_v_dogs_01/data/record_holder/150/train.tfrecords
[2018-07-09 20:23:36,254 - train_graph.py:121 -          train_graph()][DEBUG   ]: -> START iterating training dataset
[2018-07-09 20:23:46,553 - train_graph.py:158 -          train_graph()][DEBUG   ]: [END] iterating training dataset
[2018-07-09 20:23:46,588 - train_graph.py:172 -          train_graph()][DEBUG   ]: reinitialize validation iterator: ./examples/cats_v_dogs_01/data/record_holder/150/validation.tfrecords
[2018-07-09 20:23:46,588 - train_graph.py:174 -          train_graph()][DEBUG   ]: -> START iterating validation dataset
[2018-07-09 20:23:48,662 - train_graph.py:184 -          train_graph()][DEBUG   ]: [END] iterating validation dataset
[2018-07-09 20:23:48,673 - train_graph.py:190 -          train_graph()][INFO    ]: epoch 1 validation loss: 0.5968713164329529
[2018-07-09 20:23:48,761 - train_graph.py:195 -          train_graph()][DEBUG   ]: Model checkpoint saved in path: ./examples/cats_v_dogs_01/trial_01/best_params/best_params_saver.ckpt
[2018-07-09 20:23:48,767 - train_graph.py:198 -          train_graph()][INFO    ]: best params saved: val acc: 68.300% val loss: 0.5969
[2018-07-09 20:23:48,790 - train_graph.py:218 -          train_graph()][INFO    ]: [END] epoch num: 1
[2018-07-09 20:23:48,790 - train_graph.py:100 -          train_graph()][INFO    ]: -> START epoch num: 2
[2018-07-09 20:23:48,791 - train_graph.py:110 -          train_graph()][DEBUG   ]: reset train and validation metric accumulators: [<tf.Operation 'metrics/val_metrics/val_met_reset_op' type=NoOp>, <tf.Operation 'metrics/val_loss_eval/val_loss_reset_op' type=NoOp>, <tf.Operation 'metrics/train_metrics/train_met_reset_op' type=NoOp>, <tf.Operation 'metrics/train_loss_eval/train_loss_reset_op' type=NoOp>]
[2018-07-09 20:23:48,792 - train_graph.py:118 -          train_graph()][DEBUG   ]: reinitialize training iterator: ./examples/cats_v_dogs_01/data/record_holder/150/train.tfrecords
[2018-07-09 20:23:48,792 - train_graph.py:121 -          train_graph()][DEBUG   ]: -> START iterating training dataset
[2018-07-09 20:23:56,837 - train_graph.py:158 -          train_graph()][DEBUG   ]: [END] iterating training dataset
[2018-07-09 20:23:56,839 - train_graph.py:172 -          train_graph()][DEBUG   ]: reinitialize validation iterator: ./examples/cats_v_dogs_01/data/record_holder/150/validation.tfrecords
[2018-07-09 20:23:56,839 - train_graph.py:174 -          train_graph()][DEBUG   ]: -> START iterating validation dataset
[2018-07-09 20:23:58,851 - train_graph.py:184 -          train_graph()][DEBUG   ]: [END] iterating validation dataset
[2018-07-09 20:23:58,852 - train_graph.py:190 -          train_graph()][INFO    ]: epoch 2 validation loss: 0.5637639760971069
[2018-07-09 20:23:58,921 - train_graph.py:195 -          train_graph()][DEBUG   ]: Model checkpoint saved in path: ./examples/cats_v_dogs_01/trial_01/best_params/best_params_saver.ckpt
[2018-07-09 20:23:58,927 - train_graph.py:198 -          train_graph()][INFO    ]: best params saved: val acc: 71.050% val loss: 0.5638
[2018-07-09 20:23:58,928 - train_graph.py:218 -          train_graph()][INFO    ]: [END] epoch num: 2
[2018-07-09 20:23:58,928 - train_graph.py:100 -          train_graph()][INFO    ]: -> START epoch num: 3
[2018-07-09 20:23:58,928 - train_graph.py:110 -          train_graph()][DEBUG   ]: reset train and validation metric accumulators: [<tf.Operation 'metrics/val_metrics/val_met_reset_op' type=NoOp>, <tf.Operation 'metrics/val_loss_eval/val_loss_reset_op' type=NoOp>, <tf.Operation 'metrics/train_metrics/train_met_reset_op' type=NoOp>, <tf.Operation 'metrics/train_loss_eval/train_loss_reset_op' type=NoOp>]
[2018-07-09 20:23:58,930 - train_graph.py:118 -          train_graph()][DEBUG   ]: reinitialize training iterator: ./examples/cats_v_dogs_01/data/record_holder/150/train.tfrecords
[2018-07-09 20:23:58,930 - train_graph.py:121 -          train_graph()][DEBUG   ]: -> START iterating training dataset
[2018-07-09 20:24:06,920 - train_graph.py:158 -          train_graph()][DEBUG   ]: [END] iterating training dataset
[2018-07-09 20:24:06,922 - train_graph.py:172 -          train_graph()][DEBUG   ]: reinitialize validation iterator: ./examples/cats_v_dogs_01/data/record_holder/150/validation.tfrecords
[2018-07-09 20:24:06,922 - train_graph.py:174 -          train_graph()][DEBUG   ]: -> START iterating validation dataset
[2018-07-09 20:24:08,788 - train_graph.py:184 -          train_graph()][DEBUG   ]: [END] iterating validation dataset
[2018-07-09 20:24:08,788 - train_graph.py:190 -          train_graph()][INFO    ]: epoch 3 validation loss: 0.5460363626480103
[2018-07-09 20:24:08,856 - train_graph.py:195 -          train_graph()][DEBUG   ]: Model checkpoint saved in path: ./examples/cats_v_dogs_01/trial_01/best_params/best_params_saver.ckpt
[2018-07-09 20:24:08,862 - train_graph.py:198 -          train_graph()][INFO    ]: best params saved: val acc: 72.250% val loss: 0.5460
[2018-07-09 20:24:08,863 - train_graph.py:218 -          train_graph()][INFO    ]: [END] epoch num: 3
[2018-07-09 20:24:08,863 - train_graph.py:226 -          train_graph()][INFO    ]: [END] training graph
```