def implemented(method):
    method._is_implemented = True
    return method


class Callback:
    """ used to build new callbacks 
    levels


    - train/eval* (implemented by child)
        - task
            - obtain_task
            - 
        - obtain_dataset
            # sample_dataset?
            - dataset_pass (epoch? -- not always applicable) 
                - batch
                    - obtain_data
                    - prediction
                    - performance
                        - loss
                            - calc_gradient* (train specific)
                            - apply_gradient* (train specific)
                        - metric

    control flow
    - pre (immediately)
    - post (immediately)

    """

    # task
    @implemented
    def pre_task():
        """[summary]
        """

    @implemented
    def post_task():
        """[summary]
        """

    # obtain_task
    @implemented
    def pre_obtain_task():
        """[summary]
        """

    @implemented
    def post_obtain_task():
        """[summary]
        """

    # obtain_dataset
    @implemented
    def pre_obtain_dataset():
        """[summary]
        """

    @implemented
    def post_obtain_dataset():
        """[summary]
        """

    # dataset_pass
    @implemented
    def pre_dataset_pass():
        """[summary]
        """

    @implemented
    def post_dataset_pass():
        """[summary]
        """

    # batch
    @implemented
    def pre_batch():
        """[summary]
        """

    @implemented
    def post_batch():
        """[summary]
        """

    # obtain_data
    @implemented
    def pre_obtain_data():
        """[summary]
        """

    @implemented
    def post_obtain_data():
        """[summary]
        """

    # prediction
    @implemented
    def pre_prediction():
        """[summary]
        """

    @implemented
    def post_prediction():
        """[summary]
        """

    # performance
    @implemented
    def pre_performance():
        """[summary]
        """

    @implemented
    def post_performance():
        """[summary]
        """

    # loss
    @implemented
    def pre_loss():
        """[summary]
        """

    @implemented
    def post_loss():
        """[summary]
        """

    # metric
    @implemented
    def pre_metric():
        """[summary]
        """

    @implemented
    def post_metric():
        """[summary]
        """


class TrainCallback(Callback):
    # calc gradient
    @implemented
    def pre_calc_gradient():
        """[summary]
        """

    @implemented
    def post_calc_gradient():
        """[summary]
        """

    # apply gradient
    @implemented
    def pre_apply_gradient():
        """[summary]
        """

    @implemented
    def post_apply_gradient():
        """[summary]
        """

