import os
import random
import time

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

SAVE_MODEL_PATH = './../target/models/'
if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)


class BenchmarkParams:

    def __init__(self, preprocessing_params, model_params, learning_params, binary_classification) -> None:
        super().__init__()
        self.preprocessing_params = preprocessing_params
        self.model_params = model_params
        self.learning_params = learning_params
        self.binary_classification = binary_classification

        self.preprocessing_params.binary_classification = binary_classification
        if binary_classification:
            self.model_params.output_neurons = 1
        else:
            self.model_params.output_neurons = 3

    def update_from_dictionary(self, params_dict):
        self.preprocessing_params.update_from_dictionary(params_dict)
        self.model_params.update_from_dictionary(params_dict)
        self.learning_params.update_from_dictionary(params_dict)


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


class LearningParams:
    def __init__(self) -> None:
        super().__init__()
        self.id = str(time.time()) + '-' + str(random.randint(0, 10))
        self.epochs = 10
        self.batch_size = 10
        self.early_stopping_patience = 40
        self.early_stopping_min_delta = 0.005
        self.callbacks = self.setup_callbacks()

    def update_from_dictionary(self, params_dict):
        self.id = str(time.time()) + '-' + str(random.randint(0, 10))
        if 'epochs' in params_dict:
            self.epochs = params_dict['epochs']
        if 'batch_size' in params_dict:
            self.batch_size = params_dict['batch_size']

        self.callbacks = self.setup_callbacks()

    def setup_callbacks(self):
        earlyStopping = EarlyStopping(monitor='val_binary_accuracy', min_delta=self.early_stopping_min_delta,
                                      patience=self.early_stopping_patience, verbose=0, mode='max', restore_best_weights=True)
        mcp_save = ModelCheckpoint(SAVE_MODEL_PATH + 'nn_weights-' + self.id + '.hdf5', save_best_only=True,
                                   monitor='val_binary_accuracy',
                                   mode='max')
        return [earlyStopping, mcp_save]


def default_params(binary_classification):
    preprocparams = PreprocessingParams()
    modelparams = ModelParams()
    learningparams = LearningParams()
    params = BenchmarkParams(preprocparams, modelparams, learningparams, binary_classification)
    return params
