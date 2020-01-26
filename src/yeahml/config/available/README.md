# available components

This directory contains a script [create_available.py](./create_availble.py)
that writes all currently available comonents to a `.json` file.

An example json file looks like this:
```json
{
  "meta": {
    "update_time_str": "2020-01-26 11:29:47.028075",
    "version": {
      "tensorflow": "2.1.0",
      "yeahml": "0.0.1a1"
    }
  },
  "data": [
    "elu",
    "exponential",
    "hard_sigmoid",
    "linear",
    "relu",
    "selu",
    "sigmoid",
    "softmax",
    "softplus",
    "softsign",
    "tanh"
  ]
}
```
and contains a `meta` to include various versioning information
- :`update_time_str` field (to see when the data was last updated)
- `version`:`tensorflow` field (which version of tf was installed when updated)
- `version`:`yeahml` field (which version of yeahml was installed when updated)
 and a `data` field containing all the available components.

Primarily, these files `<component>.json` are used by the tests to ensure the
current implementation includes the expected components. In the future, this
file may also be used by the config parsing logic of yeahml to ensure a user is
requesting valid components.

## NOTE:
- `.json` was selected because it is human readable and includes a value from the
list on every line (rather than a pickled list)


