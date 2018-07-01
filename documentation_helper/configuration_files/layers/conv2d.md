# Convolution 2D Layer

---

## Default/Required

```yaml
<unique name>:
  type: 'conv2d'
    opts:
      filters: <int>
```

The default behavior will create the following layer:

```python
out = tf.layers.conv2d(
    cur_input,
    filters= "<int>",
    kernel_size=3,
    strides=1,
    padding="SAME",
    activation=tf.nn.relu,
    kernel_initializer=None, # defaults to glorot
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    trainable=True,
    name="<unique name>",
)

```

---

## Options

```yaml
<unique name>:
  type: 'conv2d'
    opts:
      filters: <int>
      kernel_size: [<int>, <int>]
      strides: <int>
      padding: ["SAME", "VALID"]
      activation: ["sigmoid","tanh","elu","selu","softplus","softsign","relu","relu6"]
      kernel_initializer: ["glorot","zeros","ones","rand","he"]
      kernel_regularizer: ["l1","l2","l1l2"]
      bias_regularizer: ["l1","l2","l1l2"]
      trainable: <bool>
```

---

## Example

```yaml
conv_3:
  type: 'conv2d'
  options:
    filters: 64
    kernel_size: 3
    strides: 1
```

This will create a conv2d layer named "conv_3" that will have 64 [3,3] filters and a stride of 1.