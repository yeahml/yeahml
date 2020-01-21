# import tensorflow as tf


# def init_params_from_file(sess, main_cdict: dict, model_cdict: dict):

#     # TODO: if a special path is listed to load vars from for a particular layer,
#     # a second init is needed. I don't think this use case will be very common
#     # so I will come back and work on this logic later

#     FULL_ERROR = True
#     load_names, layer_tensor_params = [], []
#     for l_name in model_cdict["layers"]:
#         try:
#             if model_cdict["layers"][l_name]["saver"]["load_params"]:
#                 # set name to load var from the indicated path, will default to
#                 # the current name of the layer
#                 try:
#                     load_name = model_cdict["layers"][l_name]["saver"]["load_name"]
#                     if load_name == None:
#                         load_name = l_name
#                 except KeyError:
#                     load_name = l_name
#                 # set the path from which to load the variables. The default path
#                 # is set in the model config but an option is presented to load from other files
#                 # try:
#                 #    load_path = model_cdict["layers"][l_name]["saver"]["load_path"]
#                 # except KeyError:
#                 #    load_name = main_cdict["load_params_path"]

#                 try:
#                     name_str = "{}".format(l_name)
#                     layer_tensor = tf.get_collection(
#                         tf.GraphKeys.GLOBAL_VARIABLES, scope=name_str
#                     )
#                     # filter for only bias or kernel
#                     layer_tensor = [
#                         l
#                         for l in layer_tensor
#                         if l.name.rstrip("0123456789").endswith("kernel:")
#                         or l.name.rstrip("0123456789").endswith("bias:")
#                         or v.name.rstrip("0123456789").endswith("word_embeddings:")
#                     ]

#                     for t_param in layer_tensor:
#                         # the split logic is used to remove the (potentially different) name
#                         # for example conv_2/kernel:0 will become kernel:0 which will become "kernel", for which we can
#                         # append to the name to be used for the layer
#                         p_name = t_param.name.split("/")[1].split(":")[0]

#                         # build list of vars to load and vars to load onto
#                         load_names.append(load_name + "/" + p_name)
#                         layer_tensor_params.append(t_param)

#                 except:
#                     sys.exit("unable to get {}".format(l_name))
#         except KeyError:
#             # no init from saver
#             pass

#     assert len(load_names) == len(
#         layer_tensor_params
#     ), "indicated number of params to load and params found are not equal"

#     ## Initialize indicated vars from file
#     init_vars = dict(zip(load_names, layer_tensor_params))
#     if len(init_vars) > 0:
#         restore_saver = tf.train.Saver(init_vars)
#         try:
#             restore_saver.restore(sess, main_cdict["load_params_path"])
#         except tf.errors.InvalidArgumentError as err:
#             if FULL_ERROR:
#                 print(err)
#             else:
#                 print(err.message)
#                 print(
#                     "The likely cause of this error is:\n 1) {}\n 2) {}".format(
#                         "The shapes are mismatched (check err message for hint)",
#                         "The naming of the target tensor is mismatched",
#                     )
#                 )
#                 print(
#                     "note: if you wish to see the full error message, please enable 'overall:full_error_message: True'"
#                 )
#             sys.exit("ERROR > EXIT: unable to restore indicated params")
