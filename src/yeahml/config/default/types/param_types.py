class optional_config:
    def __init__(self, conf_dict=None):
        if conf_dict is None:
            self.conf_dict = None
        else:
            self.conf_dict = conf_dict

    # def __str__(self):
    #     return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.conf_dict


class parameter_config:
    def __init__(self, known_dict=None, unknown_dict=None):
        if known_dict is None:
            self.known_dict = None
        else:
            self.known_dict = known_dict

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self, opts_raw):
        if isinstance(opts_raw, dict):
            out_dict = {}
            # get all knowns
            for opt_name, opt_config in self.known_dict.items():
                try:
                    raw_val = opts_raw[opt_name]
                except KeyError:
                    raw_val = None
                out = opt_config(raw_val)
                out_dict[opt_name] = out

            # include others (pass them through)
            for opt_name, opt_config in opts_raw.items():
                # TODO: verify this
                if opt_name not in self.known_dict:
                    out_dict[opt_name] = opt_config

        else:
            raise TypeError(f"param config is not a dict is type:{type(opts)}")

        return out_dict
