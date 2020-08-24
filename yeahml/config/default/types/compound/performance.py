from yeahml.build.components.loss import return_available_losses
from yeahml.build.components.metric import return_available_metrics
from yeahml.config.default.types.base_types import (
    categorical,
    list_of_categorical,
    list_of_dict,
)

# TODO: I'm not sure where this belongs yet
AVAILABLE_TRACKERS = ["mean"]


class loss_config:
    def __init__(self, loss_type=None, loss_options=None, loss_track=None):

        self.type = categorical(
            default_value=None,
            required=True,
            is_in_list=return_available_losses().keys(),
            is_type=str,
            to_lower=True,
        )(loss_type)
        # TODO: error check that options are valid
        self.options = list_of_dict(default_value=None, is_type=list, required=False)(
            loss_options
        )
        self.track = list_of_categorical(
            default_value=None,
            is_type=str,
            required=False,
            is_in_list=AVAILABLE_TRACKERS,
            to_lower=True,
        )(loss_track)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.__dict__


class metric_config:
    def __init__(self, metric_type=None, metric_options=None):

        # TODO: this should parse each option.. not do them all at once.

        self.type = list_of_categorical(
            default_value=None,
            is_type=str,
            required=True,
            is_in_list=return_available_metrics().keys(),
            to_lower=True,
            allow_duplicates=True,
        )(metric_type)
        self.options = list_of_dict(default_value=None, is_type=list, required=False)(
            metric_options
        )

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.__dict__


# NOTE: this doesn't feel good, but I'm not sure how else to approach this yet.
# that is, I don't like the idea of specifying supervised/unsupervised etc.
# TODO: I think a better approach may be to see what metric/loss is specified at
# make sure that we are including values for each
PERFORMANCE_OPTIONS = [("supervised", ["prediction", "target"])]


class performance_options:
    # this is a really ugly class. I need to revisit this.
    def __init__(self, cur_type=None, cur_options=None, cur_dataset=None):
        if not cur_type:
            raise ValueError("no type is specified")
        performance_options_names = [v[0] for v in PERFORMANCE_OPTIONS]
        self.type = categorical(
            default_value=None,
            required=True,
            is_type=str,
            to_lower=True,
            is_in_list=performance_options_names,
        )(cur_type)

        ind = performance_options_names.index(self.type)
        required_opts = PERFORMANCE_OPTIONS[ind][1]

        if cur_options:
            # check cur options
            # TODO: this will need to be improved at some point - error checking
            if isinstance(cur_options, dict):
                cur_opt_keys = cur_options.keys()
                for v in required_opts:
                    if v not in cur_opt_keys:
                        raise ValueError(
                            f"required performance options for {cur_type} are {required_opts} but only {cur_opt_keys} were specified. at least {v} is missing"
                        )
            else:
                raise TypeError(
                    f"cur options are not a dict: {cur_options}, type:{type(cur_options)}"
                )
        self.options = cur_options

        self.dataset = cur_dataset

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.__dict__


class performance_config:
    def __init__(
        self,
        loss_type=None,
        loss_options=None,
        loss_track=None,
        metric_type=None,
        metric_options=None,
        in_dict=None,
    ):
        # TODO: there are consistency issues here with the names of classes
        # and where the types are being created/checked

        if loss_type:
            self.loss = loss_config(
                loss_type=loss_type, loss_options=loss_options, loss_track=loss_track
            )()
        else:
            self.loss = None

        if metric_type:
            self.metric = metric_config(
                metric_type=metric_type, metric_options=metric_options
            )()
        else:
            self.metric = None

        if not self.metric and not self.loss:
            raise ValueError(f"no metric or loss was defined")

        # val_in is the input to the loss/metric. right now a val_in is required
        # (by name of the layer) and it is assumed the user is requesting the
        # output of the indicated layer.
        # TODO: this could be checked to ensure we have these inputs that
        # are being requested. also, this will need to be modified, because,
        # what if we want values from an internal component of a layer, not
        # the layer output?
        if in_dict:
            try:
                in_type = in_dict["type"]
            except KeyError:
                raise ValueError(
                    f"no type was specified for current objective in_config: {in_dict}"
                )
            try:
                in_options = in_dict["options"]
            except KeyError:
                in_options = None

            try:
                in_dataset_name = in_dict["dataset"]
            except KeyError:
                raise ValueError(
                    f"no dataset was specified for currect objective in_config: {in_dict}"
                )
        else:
            raise ValueError(f"in dict was not specified")

        self.in_config = performance_options(
            cur_type=in_type, cur_options=in_options, cur_dataset=in_dataset_name
        )()

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.__dict__


class performances_parser:
    def __init__(self):
        pass

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self, performance_spec_dict):
        if isinstance(performance_spec_dict, dict):
            temp_dict = {}
            for k, d in performance_spec_dict.items():

                performance_name = categorical(
                    default_value=None, required=True, is_type=str, to_lower=False
                )(k)

                try:
                    loss_dict = d["loss"]
                except KeyError:
                    loss_dict = None
                try:
                    metric_dict = d["metric"]
                except KeyError:
                    metric_dict = None
                try:
                    in_dict = d["in_config"]
                except KeyError:
                    raise KeyError(
                        f"no :in_config dict was specified for the performance of {k}: {d}"
                    )

                if isinstance(d, dict):
                    if loss_dict:
                        try:
                            loss_type = loss_dict["type"]
                        except KeyError:
                            loss_type = None

                        try:
                            loss_options = loss_dict["options"]
                        except KeyError:
                            loss_options = None

                        try:
                            loss_track = loss_dict["track"]
                        except KeyError:
                            loss_track = None

                    if metric_dict:
                        try:
                            metric_type = metric_dict["type"]
                        except KeyError:
                            metric_type = None
                        try:
                            metric_options = metric_dict["options"]
                        except KeyError:
                            metric_options = None
                    # a loss must be specified, but no metrics are ok
                    else:
                        metric_type = None
                        metric_options = None

                    val = performance_config(
                        loss_type=loss_type,
                        loss_options=loss_options,
                        loss_track=loss_track,
                        metric_type=metric_type,
                        metric_options=metric_options,
                        in_dict=in_dict,
                    )()
                    temp_dict[performance_name] = val
                else:
                    raise TypeError(
                        f"creating performance config, the performance of {k} does not have a valid dict - {d} is type {type(d)}"
                    )

        else:
            raise ValueError(
                f"{performance_spec_dict} is type {type(performance_spec_dict)} not type {type({})}"
            )
        return temp_dict
