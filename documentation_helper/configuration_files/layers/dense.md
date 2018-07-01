# Fully Connected / Dense Layer

---

## Default/Required

```yaml
<unique name>:
  type: 'dense'
    opts:
      units: <int>
```

The default behavior will create the following layer:

```python
out = tf.layers.dense(
    inputs=cur_input,
    units="<int>",
    activation=tf.nn.relu,
    kernel_initializer=None,
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
  type: 'dense'
    opts:
      units: <int>
      activation: ["sigmoid","tanh","elu","selu","softplus","softsign","relu","relu6"]
      kernel_initializer: ["glorot","zeros","ones","rand","he"]
      kernel_regularizer: ["l1","l2","l1l2"]
      bias_regularizer: ["l1","l2","l1l2"]
      trainable: <bool>
      dropout: <int> # will wrap dense in dropout layer
```

Including the `dropout` opt will do the following:

```python

out = tf.layers.dense(OPTIONS) # defined

out = tf.layers.dropout(
    inputs=out,
    rate= "<int>",
    noise_shape=None,
    seed=None,
    training="<training>", # func param
    name=None,
)

```

---

## Example

```yaml
dense_1:
  type: 'dense'
  options:
    units: 64
    activation: 'elu'
    dropout: 0.5
```

This will create a layer named "dense_1" with 64 units, elu activated, and 0.5 dropout rate.