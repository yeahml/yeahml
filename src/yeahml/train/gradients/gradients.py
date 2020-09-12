import tensorflow as tf


def _combine_gradients(obj_to_grads):
    # TODO: need to research how best to combine the gradients here...
    # combine all gradients. This portion (with in the optimizer loop)
    # will combine the gradients as if it were trained jointly

    # TODO: this needs to be smarter.. We need to make sure the gradients are
    # combined (probably by tensor name), not just concatenated in a list
    combined_gradients = None
    for obj_name, grad_dict in obj_to_grads.items():
        # TODO: we could add scaling/weighting here
        if not combined_gradients:
            combined_gradients = grad_dict["gradients"]
        else:
            combined_gradients += grad_dict["gradients"]
    return combined_gradients


# @tf.function
def update_model_params(apply_grads_fn, obj_to_grads, model, cur_tf_optimizer):
    # combine gradients and use optimizer to update model. apply contraints to
    # model variables
    combined_gradients = _combine_gradients(obj_to_grads)

    # apply gradients to the model
    apply_grads_fn(model, combined_gradients, cur_tf_optimizer)

    # apply constraints
    for variable in model.variables:
        if variable.constraint is not None:
            variable.assign(variable.constraint(variable))


def get_apply_grad_fn():

    # https://github.com/tensorflow/tensorflow/issues/27120
    # this allows the model to continue to be trained on multiple calls
    @tf.function
    def apply_grad(model, grads, optimizer):

        # NOTE: this will throw an error for params that aren't updated by a
        # specific task

        # NOTE: any gradient adjustments would happen here
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return

    return apply_grad


# TODO: this is supervised -- may make sense to organize these a a bit
def get_get_supervised_grads_fn():

    # https://github.com/tensorflow/tensorflow/issues/27120
    # this allows the model to continue to be trained on multiple calls
    # TODO: at some point, we may need to allow for batches of different sizes
    # - experimental_relax_shapes=True (doesn't appear to work for this case)
    #   https://www.tensorflow.org/api_docs/python/tf/function
    # - (input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
    @tf.function
    def get_grad(model, batch, loss_fns, cur_objective_index, loss_descs_to_update):
        # supervised implies a x, and y.. however, this maybe should change to a
        # dict indexing
        if not isinstance(loss_fns, list):
            loss_fns = [loss_fns]

        """
        # TODO: y_batch maybe shouldn't be taken from here. the issue is that we
        # specify:
        ```
        in_config:
        type: "supervised"
        options:
          prediction: "y_pred"
          target: "x_image"
        `
        """
        x_batch, y_batch = batch
        with tf.GradientTape() as tape:
            prediction = model(x_batch, training=True)

            # NOTE: not sure how big of a performance hit this is
            # TODO: add message
            # tf.debugging.assert_shapes(
            #     [(prediction, y_batch.shape), (y_batch, y_batch.shape)]
            # )
            if isinstance(cur_objective_index, int):
                prediction = prediction[cur_objective_index]

            # TODO: apply mask?
            full_losses = []
            for i, loss_fn in enumerate(loss_fns):
                loss = loss_fn(y_batch, prediction)

                # TODO: need to verify this
                tf_desc_obj = loss_descs_to_update[i]
                if tf_desc_obj:
                    tf_desc_obj.update_state(loss)

                # TODO: custom weighting for training could be applied here
                # weighted_losses = loss * weights_per_instance
                main_loss = tf.reduce_mean(loss)
                # model.losses contains the kernel/bias constraints/regularizers
                cur_loss = tf.add_n([main_loss] + model.losses)
                # full_loss = tf.add_n(full_loss, cur_loss)
                full_losses.append(cur_loss)
                # create joint loss for current optimizer
                # e.g. final_loss = tf.reduce_mean(loss1 + loss2)
            final_loss = tf.reduce_mean(tf.math.add_n(full_losses))

            # TODO: update joint loss/desc?

        # TODO: maybe we should be able to specify which params to be optimized
        # by specific optimizers
        # NOTE: this will calculate a gradient for each model param. these
        # gradients will be combined and applied to the model later using
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))
        grads = tape.gradient(final_loss, model.trainable_variables)

        return {
            "gradients": grads,
            "predictions": prediction,
            "final_loss": final_loss,
            "losses": loss,
            "y_batch": y_batch,
        }
        # return grads, prediction, final_loss, full_losses

    return get_grad


def get_validation_step_fn():
    @tf.function
    def get_preds(model, batch, loss_fns, cur_objective_index, loss_descs_to_update):
        # supervised implies a x, and y.. however, this maybe should change to a
        # dict indexing
        if not isinstance(loss_fns, list):
            loss_fns = [loss_fns]

        """
        # TODO: y_batch maybe shouldn't be taken from here. the issue is that we
        # specify:
        ```
        in_config:
        type: "supervised"
        options:
          prediction: "y_pred"
          target: "x_image"
        `
        """
        x_batch, y_batch = batch
        prediction = model(x_batch, training=False)
        # NOTE: not sure how big of a performance hit this is
        # TODO: add message
        # tf.debugging.assert_shapes(
        #     [(prediction, y_batch.shape), (y_batch, y_batch.shape)]
        # )
        if isinstance(cur_objective_index, int):
            prediction = prediction[cur_objective_index]

        # TODO: apply mask?
        full_losses = []
        for i, loss_fn in enumerate(loss_fns):
            loss = loss_fn(y_batch, prediction)

            # TODO: need to verify this
            tf_desc_obj = loss_descs_to_update[i]
            if tf_desc_obj:
                tf_desc_obj.update_state(loss)

            # TODO: custom weighting for training could be applied here
            # weighted_losses = loss * weights_per_instance
            main_loss = tf.reduce_mean(loss)
            # model.losses contains the kernel/bias constraints/regularizers
            cur_loss = tf.add_n([main_loss] + model.losses)
            # full_loss = tf.add_n(full_loss, cur_loss)
            full_losses.append(cur_loss)
            # create joint loss for current optimizer
            # e.g. final_loss = tf.reduce_mean(loss1 + loss2)
        final_loss = tf.reduce_mean(tf.math.add_n(full_losses))

        # TODO: update joint loss/desc?

        return {
            "predictions": prediction,
            "final_loss": final_loss,
            "losses": loss,
            "y_batch": y_batch,
        }

    return get_preds
