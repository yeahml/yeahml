import tensorflow as tf
import pickle

# these two functions are used to manually save the best
# model params to disk
# TODO: file name needs to be managed
def save_obj(obj, name):
    # TODO: need to make the dir, if not already made.
    with open(
        "./example/cats_v_dogs_01/trial_01/best_params/" + name + ".pkl", "wb"
    ) as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# TODO: this could sit in a helper file?
def load_obj(name):
    with open(
        "./example/cats_v_dogs_01/trial_01/best_params/" + name + ".pkl", "rb"
    ) as f:
        return pickle.load(f)


# these two functions (get_model_params and restore_model_params) are
# ad[a|o]pted from:
# https://github.com/ageron/handson-ml/blob/master/11_deep_learning.ipynb
def get_model_params(global_vars):

    return {
        global_vars.op.name: value
        for global_vars, value in zip(
            global_vars, tf.get_default_session().run(global_vars)
        )
    }


# TODO: this could sit in a helper file?
def restore_model_params(model_params, g, sess):
    gvar_names = list(model_params.keys())
    assign_ops = {
        gvar_name: g.get_operation_by_name(gvar_name + "/Assign")
        for gvar_name in gvar_names
    }
    init_values = {
        gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()
    }
    feed_dict = {
        init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names
    }
    sess.run(assign_ops, feed_dict=feed_dict)


# TODO: file name needs to be managed
