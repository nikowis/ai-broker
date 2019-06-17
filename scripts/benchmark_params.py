import time

TARGET_PATH = './../target'
# TARGET_PATH = './drive/My Drive/ai-broker/target'
CSV_FILES_DIR = '/data'
SAVE_MODEL_DIR = '/models'
BENCHMARKS_DIR = '/benchmarks'
SAVE_IMG_DIR = '/img'
SAVE_PARTIAL_IMG_DIR = '/partial'
SAVE_FILES = True
CLEANUP_FILES = True
SATYSFYING_TRESHOLD_BINARY = 0.86
SATYSFYING_TRESHOLD_DISCRETE = 0.7


class BenchmarkParams:

    def __init__(self, binary_classification, examined_param, benchmark_name) -> None:
        self.benchmark_name = benchmark_name
        if benchmark_name is not None:
            self.benchmark_name_dir = '/' + benchmark_name
        else:
            self.benchmark_name_dir = ''
        self.examined_param = examined_param
        self.curr_iter_num = None
        self.plot_partial = False
        self.curr_sym = None
        self.csv_files_path = TARGET_PATH + '/data'
        self.benchmark_path = TARGET_PATH + BENCHMARKS_DIR + self.benchmark_name_dir
        self.save_model_path = self.benchmark_path + SAVE_MODEL_DIR
        self.save_img_path = self.benchmark_path + SAVE_IMG_DIR
        self.save_partial_img_path = self.save_img_path + SAVE_PARTIAL_IMG_DIR
        self.save_files = SAVE_FILES
        self.cleanup_files = CLEANUP_FILES
        self.verbose = True

        self.input_size = None
        self.binary_classification = binary_classification
        if binary_classification:
            self.classes_count = 2
            self.output_activation = 'sigmoid'
            self.loss = 'binary_crossentropy'
            self.metric = 'binary_accuracy'
            self.satysfying_treshold = SATYSFYING_TRESHOLD_BINARY
        else:
            self.classes_count = 3
            self.output_activation = 'softmax'
            self.loss = 'categorical_crossentropy'
            self.metric = 'categorical_accuracy'
            self.satysfying_treshold = SATYSFYING_TRESHOLD_DISCRETE
        self.pca = None
        self.test_size = 0.2
        self.standardize = True
        self.difference_non_stationary = True
        self.walk_forward_testing = False
        self.walk_forward_max_train_window_size = None
        self.walk_forward_test_window_size = 200
        self.iterations = 3

    def update_from_dictionary(self, params_dict):
        self.id = str(time.time())
        if 'pca' in params_dict:
            self.pca = params_dict['pca']
        if 'test_size' in params_dict:
            self.test_size = params_dict['test_size']
        if 'standardize' in params_dict:
            self.standardize = params_dict['standardize']
        if 'difference_non_stationary' in params_dict:
            self.difference_non_stationary = params_dict['difference_non_stationary']
        if 'walk_forward_testing' in params_dict:
            self.walk_forward_testing = params_dict['walk_forward_testing']
        if 'walk_forward_max_train_window_size' in params_dict:
            self.walk_forward_max_train_window_size = params_dict['walk_forward_max_train_window_size']
        if 'walk_forward_test_window_size' in params_dict:
            self.walk_forward_test_window_size = params_dict['walk_forward_test_window_size']

    def jsonable(self):
        return self.__dict__


class NnBenchmarkParams(BenchmarkParams):

    def __init__(self, binary_classification, examined_param=None, benchmark_name='') -> None:
        super().__init__(binary_classification, examined_param, benchmark_name)

        if binary_classification:
            self.output_neurons = 1
        else:
            self.output_neurons = 3

        self.layers = []
        self.regularizer = 0.005
        self.activation = 'relu'
        self.output_activation = 'sigmoid'  # softmax
        self.optimizer = 'adam'
        self.loss = 'binary_crossentropy'  # categorical_crossentropy
        self.metric = 'binary_accuracy'  # categorical_accuracy
        self.use_bias = True
        self.epochs = 150
        self.batch_size = 10
        self.early_stopping_patience = 20
        self.early_stopping_min_delta = 0.005

        self.walk_forward_retrain_epochs = 10

    def update_from_dictionary(self, params_dict):
        super().update_from_dictionary(params_dict)
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
        if 'epochs' in params_dict:
            self.epochs = params_dict['epochs']
        if 'batch_size' in params_dict:
            self.batch_size = params_dict['batch_size']
        if 'walk_forward_retrain_epochs' in params_dict:
            self.walk_forward_retrain_epochs = params_dict['walk_forward_retrain_epochs']
