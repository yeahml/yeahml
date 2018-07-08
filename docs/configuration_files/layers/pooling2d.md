# Pooling Layer

---

## Default/Required

```yaml
<unique name>:
  type: 'pooling2d'
```

The default behavior will create the following layer:

```python
out = tf.layers.max_pooling2d(
    cur_input, pool_size=[2, 2], strides=2, padding="valid", name="<unique name>"
)
```

---

## Options

```yaml
<unique name>:
  type: 'pooling2d'
    options:
      pool_type: ["max","avg"]
      pool_size: [<int>, <int>]
      strides: <int>
      dropout: <int>
```

The average pooling layer implementation is as follows:

```python
out = tf.layers.average_pooling2d(
    cur_input, pool_size=pool_size, strides=strides, padding="valid", name=name
)
```

Including the `dropout` opt will implement the following logic:

```python

out = tf.layers.average_pooling2d(OPTIONS) # defined

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
pool_3:
  type: 'pooling2d'
    options:
      pool_type: "avg"
      pool_size: [2,2]
      strides: 2
      dropout: 0.5
```

This will create a average pooling layer named "pool_3" with a pool size of [2,2] and a stride of 2, and 0.5 dropout rate.