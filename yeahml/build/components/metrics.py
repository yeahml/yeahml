import tensorflow as tf
from typing import Any


def get_metrics_fn(metric_str: str) -> Any:

    # TODO: this functionality should mimic that of Layers

    # TODO: this check should be pushed to the config logic
    if metric_str:
        metric_str = metric_str.lower()

    if metric_str == "AUC".lower():
        met_obj = tf.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            name=None,
            dtype=None,
        )
    elif metric_str == "BinaryAccuracy".lower():
        met_obj = tf.metrics.BinaryAccuracy(
            name="binary_accuracy", dtype=None, threshold=0.5
        )
    elif metric_str == "BinaryCrossentropy".lower():
        met_obj = tf.metrics.BinaryCrossentropy(
            name="binary_crossentropy", dtype=None, from_logits=False, label_smoothing=0
        )
    elif metric_str == "CategoricalAccuracy".lower():
        met_obj = tf.metrics.CategoricalAccuracy(
            name="categorical_accuracy", dtype=None
        )
    elif metric_str == "CategoricalCrossentropy".lower():
        met_obj = tf.metrics.CategoricalCrossentropy(
            name="categorical_crossentropy",
            dtype=None,
            from_logits=False,
            label_smoothing=0,
        )
    elif metric_str == "CategoricalHinge".lower():
        met_obj = tf.metrics.CategoricalHinge(name="categorical_hinge", dtype=None)
    elif metric_str == "CosineSimilarity".lower():
        met_obj = tf.metrics.CosineSimilarity(
            name="cosine_similarity", dtype=None, axis=-1
        )
    elif metric_str == "FalseNegatives".lower():
        met_obj = tf.metrics.FalseNegatives(thresholds=None, name=None, dtype=None)
    elif metric_str == "FalsePositives".lower():
        met_obj = tf.metrics.FalsePositives(thresholds=None, name=None, dtype=None)
    elif metric_str == "Hinge".lower():
        met_obj = tf.metrics.Hinge(name="hinge", dtype=None)
    elif metric_str == "KLDivergence".lower():
        met_obj = tf.metrics.KLDivergence(
            name="kullback_leibler_divergence", dtype=None
        )
    elif metric_str == "LogCoshError".lower():
        met_obj = tf.metrics.LogCoshError(name="logcosh", dtype=None)
    elif metric_str == "Mean".lower():
        met_obj = tf.metrics.Mean(name="mean", dtype=None)
    elif metric_str == "MeanAbsoluteError".lower():
        met_obj = tf.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)
    elif metric_str == "MeanAbsolutePercentageError".lower():
        met_obj = tf.metrics.MeanAbsolutePercentageError(
            name="mean_absolute_percentage_error", dtype=None
        )
    elif metric_str == "MeanIoU".lower():
        raise NotImplementedError
    #     # TODO: need to pass num_classes
    #     met_obj = tf.metrics.MeanIoU(num_classes, name=None, dtype=None)
    elif metric_str == "MeanRelativeError".lower():
        raise NotImplementedError
    #     # TODO: need to pass normalizer
    #     met_obj = tf.metrics.MeanRelativeError(normalizer, name=None, dtype=None)
    elif metric_str == "MeanSquaredError".lower():
        met_obj = tf.metrics.MeanSquaredError(name="mean_squared_error", dtype=None)
    elif metric_str == "MeanSquaredLogarithmicError".lower():
        met_obj = tf.metrics.MeanSquaredLogarithmicError(
            name="mean_squared_logarithmic_error", dtype=None
        )
    elif metric_str == "MeanTensor".lower():
        met_obj = tf.metrics.MeanTensor(name="mean_tensor", dtype=None)
    elif metric_str == "Poisson".lower():
        met_obj = tf.metrics.Poisson(name="poisson", dtype=None)
    elif metric_str == "Precision".lower():
        met_obj = tf.metrics.Precision(
            thresholds=None, top_k=None, class_id=None, name=None, dtype=None
        )
    elif metric_str == "Recall".lower():
        met_obj = tf.metrics.Recall(
            thresholds=None, top_k=None, class_id=None, name=None, dtype=None
        )
    elif metric_str == "RootMeanSquaredError".lower():
        met_obj = tf.metrics.RootMeanSquaredError(
            name="root_mean_squared_error", dtype=None
        )
    elif metric_str == "SensitivityAtSpecificity".lower():
        raise NotImplementedError
    #     # TODO: need to specify specificity
    #     met_obj = tf.metrics.SensitivityAtSpecificity(
    #         specificity, num_thresholds=200, name=None, dtype=None
    #     )
    elif metric_str == "SparseCategoricalAccuracy".lower():
        met_obj = tf.metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy", dtype=None
        )
    elif metric_str == "SparseCategoricalCrossentropy".lower():
        met_obj = tf.metrics.SparseCategoricalCrossentropy(
            name="sparse_categorical_crossentropy",
            dtype=None,
            from_logits=False,
            axis=-1,
        )
    elif metric_str == "SparseTopKCategoricalAccuracy".lower():
        met_obj = tf.metrics.SparseTopKCategoricalAccuracy(
            k=5, name="sparse_top_k_categorical_accuracy", dtype=None
        )
    elif metric_str == "SpecificityAtSensitivity".lower():
        raise NotImplementedError
        # # TODO: need to specify specificity
        # met_obj = tf.metrics.SpecificityAtSensitivity(
        #     sensitivity, num_thresholds=200, name=None, dtype=None
        # )
    elif metric_str == "Sum".lower():
        met_obj = tf.metrics.Sum(name="sum", dtype=None)
    elif metric_str == "TopKCategoricalAccuracy".lower():
        # TODO: specify k
        met_obj = tf.metrics.TopKCategoricalAccuracy(
            k=5, name="top_k_categorical_accuracy", dtype=None
        )
    elif metric_str == "TrueNegatives".lower():
        met_obj = tf.metrics.TrueNegatives(thresholds=None, name=None, dtype=None)
    elif metric_str == "TruePositives".lower():
        met_obj = tf.metrics.TruePositives(thresholds=None, name=None, dtype=None)
    else:
        raise NotImplementedError

    return met_obj
