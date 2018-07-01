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
    cur_input, pool_size=[2, 2], strides=2, name="<unique name>"
)
```

---

## Options

```yaml
<unique name>:
  type: 'pooling2d'
    opts:
      pool_size: [<int>, <int>]
      strides: <int>
      dropout: <int>
```

---

## Example

```yaml
pool_3:
  type: 'pooling2d'
    opts:
      pool_size: [2,2]
      strides: 2
      dropout: 0.5
```

This will create a max pooling layer named "pool_3" with a pool size of [2,2] and a stride of 2, and 0.5 dropout rate.