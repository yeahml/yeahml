
'''
TODO:
'''
try:
    import matplotlib.pyplot as plt
except ImportError:
    # TODO: better instructions
    raise ImportError("please install matplotlib")


def _basic_plot_loss_tracker(
    loss_tracker,
    skip=0,
    local=False,
    training=False,
    validation=False,
    obj_name=None,
    opt_name=None,
):
    if not training and not validation:
        raise ValueError("please specify validation or training")
    for ds_name, ds_dict in loss_tracker.items():
        for split_name, split_dict in ds_dict.items():
            if split_name == "train" and not training:
                pass
            elif split_name == "val" and not validation:
                pass
            else:
                for loss_name, loss_desc_d in split_dict.items():
                    for description, tracker in loss_desc_d.items():
                        values = tracker.values[skip:]
                        steps = (
                            tracker.steps[skip:]
                            if local
                            else tracker.global_step[skip:]
                        )
                        plt.plot(
                            steps,
                            values,
                            "o--",
                            linewidth=2,
                            markersize=5,
                            alpha=0.7,
                            label=f"{opt_name} {obj_name} {ds_name} {split_name} {loss_name} {description}",
                        )

def _basic_plot_metrics_tracker(
    loss_tracker,
    skip=0,
    local=False,
    training=False,
    validation=False,
    obj_name=None,
    opt_name=None,
):
    if not training and not validation:
        raise ValueError("please specify validation or training")
    for ds_name, ds_dict in loss_tracker.items():
        for split_name, split_dict in ds_dict.items():
            if split_name == "train" and not training:
                pass
            elif split_name == "val" and not validation:
                pass
            else:
                for metric_name, tracker in split_dict.items():
                    values = tracker.values[skip:]
                    steps = (
                        tracker.steps[skip:] if local else tracker.global_step[skip:]
                    )
                    plt.plot(
                        steps,
                        values,
                        "o--",
                        linewidth=2,
                        markersize=5,
                        alpha=0.7,
                        label=f"{opt_name} {obj_name} {ds_name} {split_name} {metric_name}",
                    )


def basic_plot_tracker(
    tracker,
    loss=False,
    metrics=False,
    local=False,
    training=False,
    validation=False,
    size=(8, 4),
):

    if not loss and not metrics:
        raise ValueError("please specify validation or training")

    if loss:
        plt.figure(figsize=size)
        for opt_name, opt_tracker_d in tracker.items():

            # TODO: eventually, also plot joint
            objective_names = opt_tracker_d["objectives"]
            for obj_name in objective_names:
                # metrics, loss
                _basic_plot_loss_tracker(
                    opt_tracker_d[obj_name]["loss"],
                    local=local,
                    training=training,
                    validation=validation,
                    obj_name=obj_name,
                    opt_name=opt_name,
                )
        plt.legend()
        plt.show()
    if metrics:
        plt.figure(figsize=size)
        for opt_name, opt_tracker_d in tracker.items():

            # TODO: eventually, also plot joint
            objective_names = opt_tracker_d["objectives"]
            for obj_name in objective_names:
                # metrics, loss
                _basic_plot_metrics_tracker(
                    opt_tracker_d[obj_name]["metrics"],
                    local=local,
                    training=training,
                    validation=validation,
                    obj_name=obj_name,
                    opt_name=opt_name,
                )
        plt.legend()
        plt.show()