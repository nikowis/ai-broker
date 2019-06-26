import time
import uuid

TARGET_PATH = './../target'
# TARGET_PATH = './drive/My Drive/ai-broker/target'
CSV_FILES_DIR = '/data'
SAVE_MODEL_DIR = '/models'
BENCHMARKS_DIR = '/benchmarks'
SAVE_IMG_DIR = '/img'
SAVE_PARTIAL_IMG_DIR = '/partial'
SAVE_FILES = True
CLEANUP_FILES = True
SATYSFYING_TRESHOLD_BINARY = 0.91
SATYSFYING_TRESHOLD_DISCRETE = 0.87


class BenchmarkParams:

    def __init__(self, binary_classification, examined_param='', benchmark_name=str(uuid.uuid4())) -> None:
        self.id = str(time.time())
        self.benchmark_name = benchmark_name
        if benchmark_name is not None:
            self.benchmark_name_dir = '/' + benchmark_name
        else:
            self.benchmark_name_dir = ''
        self.examined_params = examined_param
        self.curr_iter_num = None
        self.plot_partial = False
        self.curr_sym = None
        self.csv_files_path = TARGET_PATH + '/data'
        self.benchmark_path = TARGET_PATH + BENCHMARKS_DIR + self.benchmark_name_dir
        self.save_model_path = self.benchmark_path + SAVE_MODEL_DIR
        self.save_img_path = self.benchmark_path + SAVE_IMG_DIR
        self.save_partial_img_path = self.save_img_path + SAVE_PARTIAL_IMG_DIR
        self.save_files = SAVE_FILES
        self.save_model = SAVE_FILES
        self.cleanup_files = CLEANUP_FILES
        self.verbose = True

        self.input_size = None
        self.binary_classification = binary_classification
        if binary_classification:
            self.classes_count = 2
            self.satysfying_treshold = SATYSFYING_TRESHOLD_BINARY
        else:
            self.classes_count = 3
            self.satysfying_treshold = SATYSFYING_TRESHOLD_DISCRETE
        self.pca = 0.9999
        self.test_size = 0.2
        self.standardize = True
        self.difference_non_stationary = True
        self.walk_forward_testing = False
        self.max_train_window_size = None
        self.walk_forward_learn_from_scratch = True
        self.iterations = 3
        self.one_hot_encode_labels = True
        self.feature_names = []

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
        if 'max_train_window_size' in params_dict:
            self.max_train_window_size = params_dict['max_train_window_size']
        if 'walk_forward_test_window_size' in params_dict:
            self.walk_forward_test_window_size = params_dict['walk_forward_test_window_size']
        if 'walk_forward_learn_from_scratch' in params_dict:
            self.walk_forward_learn_from_scratch = params_dict['walk_forward_learn_from_scratch']

    def jsonable(self):
        return self.__dict__


class NnBenchmarkParams(BenchmarkParams):

    def __init__(self, binary_classification, examined_param='', benchmark_name=str(uuid.uuid4())) -> None:
        super().__init__(binary_classification, examined_param, benchmark_name)

        if binary_classification:
            self.output_neurons = 1
            self.layers = []
            self.output_activation = 'sigmoid'
            self.loss = 'binary_crossentropy'
            self.metric = 'binary_accuracy'
            self.walk_forward_test_window_size = 360
        else:
            self.output_neurons = 3
            self.layers = [10]
            self.output_activation = 'softmax'
            self.loss = 'categorical_crossentropy'
            self.metric = 'categorical_accuracy'
            self.walk_forward_test_window_size = 22

        self.regularizer = 0.0025
        self.activation = 'relu'
        self.optimizer = 'adam'
        self.use_bias = True
        self.epochs = 150
        self.batch_size = 10
        self.early_stopping_patience = 40
        self.early_stopping_min_delta = 0.005
        self.walk_forward_retrain_epochs = 5

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


class SVMBenchmarkParams(BenchmarkParams):

    def __init__(self, binary_classification, examined_param=None, benchmark_name=str(uuid.uuid4())) -> None:
        super().__init__(binary_classification, examined_param, benchmark_name)
        self.walk_forward_test_window_size = 360
        self.c = 10
        self.kernel = 'linear'
        self.degree = 3
        self.gamma = 0.005
        self.epsilon = 0.1
        self.one_hot_encode_labels = False
        self.save_model = False
        self.iterations = 1

    def update_from_dictionary(self, params_dict):
        super().update_from_dictionary(params_dict)
        if 'c' in params_dict:
            self.c = params_dict['c']
        if 'kernel' in params_dict:
            self.kernel = params_dict['kernel']
        if 'epsilon' in params_dict:
            self.epsilon = params_dict['epsilon']
        if 'degree' in params_dict:
            self.degree = params_dict['degree']
        if 'gamma' in params_dict:
            self.gamma = params_dict['gamma']


class LightGBMBenchmarkParams(BenchmarkParams):

    def __init__(self, binary_classification, examined_param=None, benchmark_name=str(uuid.uuid4())) -> None:
        super().__init__(binary_classification, examined_param, benchmark_name)

        if self.binary_classification:
            self.objective = 'binary'
            self.model_num_class = 1
            self.max_depth = -1
            self.boosting = 'gbdt'
        else:
            self.objective = 'multiclassova'
            self.model_num_class = 3
            self.max_depth = -1
            self.boosting = 'dart'  # gbdt, gbrt, rf, random_forest, dart, goss
        self.one_hot_encode_labels = False
        self.save_model = False
        self.iterations = 1
        self.num_leaves = 31
        self.learning_rate = 0.05
        self.num_boost_round = 300
        self.max_bin = 255
        self.pca = None
        self.feature_fraction = 1
        self.min_sum_hessian_in_leaf = 1e-3
        self.min_data_in_leaf = 20
        self.walk_forward_test_window_size = 360

    def update_from_dictionary(self, params_dict):
        super().update_from_dictionary(params_dict)
        if 'num_leaves' in params_dict:
            self.num_leaves = params_dict['num_leaves']
        if 'learning_rate' in params_dict:
            self.learning_rate = params_dict['learning_rate']
        if 'max_bin' in params_dict:
            self.max_bin = params_dict['max_bin']
        if 'max_depth' in params_dict:
            self.max_depth = params_dict['max_depth']
        if 'boosting' in params_dict:
            self.boosting = params_dict['boosting']
        if 'feature_fraction' in params_dict:
            self.feature_fraction = params_dict['feature_fraction']
        if 'num_boost_round' in params_dict:
            self.num_boost_round = params_dict['num_boost_round']
        if 'min_sum_hessian_in_leaf' in params_dict:
            self.min_sum_hessian_in_leaf = params_dict['min_sum_hessian_in_leaf']
        if 'min_data_in_leaf' in params_dict:
            self.min_data_in_leaf = params_dict['min_data_in_leaf']
