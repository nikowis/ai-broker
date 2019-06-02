import random
import time


class BenchmarkParams:

    def __init__(self, preprocessing_params, model_params, learning_params, binary_classification) -> None:
        super().__init__()
        self.preprocessing_params: PreprocessingParams = preprocessing_params
        self.model_params: ModelParams = model_params
        self.learning_params: LearningParams = learning_params

        self.preprocessing_params.binary_classification = binary_classification
        if binary_classification:
            self.classes_count = 2
            self.model_params.output_neurons = 1
            self.model_params.output_activation = 'sigmoid'
            self.model_params.loss = 'binary_crossentropy'
            self.model_params.metric = 'binary_accuracy'
        else:
            self.classes_count = 3
            self.model_params.output_neurons = 3
            self.model_params.output_activation = 'softmax'
            self.model_params.loss = 'categorical_crossentropy'
            self.model_params.metric = 'categorical_accuracy'

    def update_from_dictionary(self, params_dict):
        self.preprocessing_params.update_from_dictionary(params_dict)
        self.model_params.update_from_dictionary(params_dict)
        self.learning_params.update_from_dictionary(params_dict)

    def jsonable(self):
        return self.__dict__


class PreprocessingParams:
    def __init__(self) -> None:
        super().__init__()
        self.pca = 0.999
        self.test_size = 0.2
        self.standarize = True
        self.robust_scaler = False
        self.difference_non_stationary = True
        self.binary_classification = True

    def update_from_dictionary(self, params_dict):
        if 'pca' in params_dict:
            self.pca = params_dict['pca']
        if 'test_size' in params_dict:
            self.pca = params_dict['test_size']
        if 'standarize' in params_dict:
            self.pca = params_dict['standarize']
        if 'robust_scaler' in params_dict:
            self.pca = params_dict['robust_scaler']
        if 'difference_non_stationary' in params_dict:
            self.pca = params_dict['difference_non_stationary']

    def jsonable(self):
        return self.__dict__


class ModelParams:
    def __init__(self) -> None:
        super().__init__()
        self.layers = [10]
        self.regularizer = 0.005
        self.activation = 'relu'
        self.output_activation = 'sigmoid'  # softmax
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'  # categorical_crossentropy
        self.metric = 'binary_accuracy'  # categorical_accuracy
        self.use_bias = True
        self.output_neurons = 1

    def update_from_dictionary(self, params_dict):
        if 'layers' in params_dict:
            self.layers = params_dict['layers']
        if 'regularizer' in params_dict:
            self.regularizer = params_dict['regularizer']
        if 'activation' in params_dict:
            self.activation = params_dict['activation']
        if 'output_activation' in params_dict:
            self.output_activation = params_dict['output_activation']
        if 'optimizer' in params_dict:
            self.optimizer = params_dict['optimizer']
        if 'loss' in params_dict:
            self.loss = params_dict['loss']
        if 'metric' in params_dict:
            self.metric = params_dict['metric']
        if 'use_bias' in params_dict:
            self.use_bias = params_dict['use_bias']

    def jsonable(self):
        return self.__dict__


class LearningParams:
    def __init__(self) -> None:
        super().__init__()
        self.id = str(time.time())
        self.epochs = 10
        self.batch_size = 10
        self.early_stopping_patience = 40
        self.early_stopping_min_delta = 0.005

    def update_from_dictionary(self, params_dict):
        self.id = str(time.time())
        if 'epochs' in params_dict:
            self.epochs = params_dict['epochs']
        if 'batch_size' in params_dict:
            self.batch_size = params_dict['batch_size']

    def jsonable(self):
        return self.__dict__


def default_params(binary_classification):
    preprocparams = PreprocessingParams()
    modelparams = ModelParams()
    learningparams = LearningParams()
    params = BenchmarkParams(preprocparams, modelparams, learningparams, binary_classification)
    return params
