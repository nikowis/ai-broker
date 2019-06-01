class BenchmarkParams:

    def __init__(self, processingParams, modelParams, learningParams) -> None:
        super().__init__()
        self.processingParams = processingParams
        self.modelParams = modelParams
        self.learningParams = learningParams


class PreprocessingParams:
    def __init__(self) -> None:
        super().__init__()
        self.pca = 1
        self.test_size = 0.2
        self.standarize = True
        self.robust_scaler = False
        self.difference_non_stationary = True
        self.binary_classification = True


class ModelParams:
    def __init__(self) -> None:
        super().__init__()


class LearningParams:
    def __init__(self) -> None:
        super().__init__()


def default_params():
    preprocparams = PreprocessingParams()
    modelparams = ModelParams()
    learningparams = LearningParams()
    params = BenchmarkParams(preprocparams, modelparams, learningparams)
    return params
