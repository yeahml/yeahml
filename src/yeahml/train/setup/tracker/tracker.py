class Tracker:
    def __init__(self, to_track):
        if not to_track:
            self.to_track = None
        elif isinstance(to_track, list):
            lower = []
            for obj in to_track:
                if not isinstance(obj, str):
                    raise TypeError(f"object {obj} of {to_track} is not type string")
                else:
                    lower.append(obj.lower())
            self.to_track = lower
        elif isinstance(to_track, str):
            self.to_track = [to_track.lower()]
        else:
            raise TypeError(f"{to_track} must be a list of strings or string")

        self.values = None
        self.steps = None

        # if not step_description:
        #     raise ValueError(f"must specify a step_description")
        # elif not isinstance(step_description, str):
        #     raise TypeError(
        #         f"step_description ({step_description}) must be type string not {type(step_description)}"
        #     )
        # self.step_description = step_description

        for obj in self.to_track:
            if obj == "max":
                self.max = None
                self.step_at_max = None
            elif obj == "min":
                self.min = None
                self.step_at_min = None
            else:
                raise ValueError(f"{obj} is not an accepted description to track")

    def update(self, step, value):
        UPDATED = {}
        if self.steps and self.values:
            self.steps.append(step)
            self.values.append(value)
        else:
            self.steps = [step]
            self.values = [value]
        try:
            cur_max = self.max
            UPDATED["max"] = False
            if cur_max:
                if value > cur_max:
                    UPDATED["max"] = True
                    self.max = value
                    self.step_at_max = step
            else:
                UPDATED["max"] = True
                self.max = value
                self.step_at_max = step
        except AttributeError:
            pass

        try:
            cur_min = self.min
            UPDATED["min"] = False
            if cur_min:
                if value < cur_min:
                    UPDATED["min"] = True
                    self.min = value
                    self.step_at_min = step
            else:
                UPDATED["min"] = True
                self.min = value
                self.step_at_min = step
        except AttributeError:
            pass

        return UPDATED

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.__dict__


## create


def create_joint_dict_tracker(optimizer_to_loss_name_map):
    joint_dict_tracker = {}
    for _, temp_dict in optimizer_to_loss_name_map.items():
        try:
            jd = temp_dict["losses_to_optimize"]["joint_record"]
            joint_name = temp_dict["losses_to_optimize"]["joint_name"]
        except KeyError:
            pass

        if jd:
            joint_dict_tracker[joint_name] = {}
            for ds_name, do in jd.items():
                for description, met_tensor in do.items():
                    joint_dict_tracker[joint_name][ds_name] = {
                        f"{description}": {
                            "epoch": {"best": None, "steps": None, "values": None}
                        }
                    }
    return joint_dict_tracker


## record


def record_joint_losses(
    ds_name,
    step_name,
    step_value,
    joint_dict_tracker,
    joint_loss_name,
    joint_loss_record,
):

    # {
    #     "main_obj__second_obj__joint_train": {
    #         "train": {"mean": {"epoch": {"best": None, "steps": None, "values": None}}},
    #         "val": {"mean": {"epoch": {"best": None, "steps": None, "values": None}}},
    #     }
    # }

    best_update = {}
    for name, mets in joint_loss_record.items():
        if not isinstance(mets, list):
            mets = [mets]
        for i, metric_object in enumerate(mets):

            if not joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                "steps"
            ]:
                joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                    "steps"
                ] = [step_value]
            else:
                joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                    "steps"
                ].append(step_value)

            cur_val = metric_object.result().numpy()
            if not joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                "values"
            ]:
                joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                    "values"
                ] = [cur_val]
            else:
                joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                    "values"
                ].append(cur_val)

            prev_best = joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                "best"
            ]
            if not prev_best:
                joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                    "best"
                ] = cur_val
                update = True
                best_update[joint_loss_name] = {name: True}
            else:
                # NOTE: currently assuming min
                if cur_val < prev_best:
                    joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                        "best"
                    ] = cur_val
                    update = True
                else:
                    update = False
            best_update[joint_loss_name] = {name: update}
    return best_update
