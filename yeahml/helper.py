import tensorflow as tf


def fmt_tensor_info(t):
    return "| {:15s} | {}".format(t.name.rstrip("0123456789").rstrip(":"), t.shape)


def fmt_metric_summary(summary) -> dict:
    summary_proto = tf.Summary()
    summary_proto.ParseFromString(summary)
    summaries = {}
    for val in summary_proto.value:
        # NOTE: Assuming scalar summaries
        summaries[val.tag] = val.simple_value
    return summaries
