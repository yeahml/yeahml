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

        # if unknown_dict is None:
        #     self.unknown_dict = None
        # else:
        #     self.unknown_dict = unknown_dict

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    # def __call__(self):
    #     # out_dict = {**known_dict, **unknown_dict}
    #     out_dict = self.known_dict
    #     return out_dict
