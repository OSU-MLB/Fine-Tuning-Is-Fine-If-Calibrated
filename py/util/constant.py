import os

# Path
ASSET_DIR_NAME = 'asset'
OUT_DIR_NAME = 'out'
EXPERIMENT_DIR_NAME = 'experiment'
EXPERIMENT_PATH = os.path.join(OUT_DIR_NAME, EXPERIMENT_DIR_NAME)

# Path Constants
SOURCE_EPOCH = -1
MODEL_CKPT_PATH = 'model'
EVALUATION_CKPT_PATH = 'evaluation'
ORACLE_TRAINING_EVALUATION_CKPT_PATH = os.path.join(EVALUATION_CKPT_PATH, 'oracle_training')
TESTING_EVALUATION_CKPT_PATH = os.path.join(EVALUATION_CKPT_PATH, 'testing')

# Setting
EVALUATE_TOP_K = (1, 5)

# OfficeHome dataset
OFFICEHOME_DIR_PATH = os.path.join(ASSET_DIR_NAME, 'OfficeHome')
OFFICEHOME_DATA_PATH = os.path.join(OFFICEHOME_DIR_PATH, 'data')
OFFICEHOME_IMAGE_LIST_PATH = os.path.join(OFFICEHOME_DIR_PATH, 'image_list')
OFFICEHOME_N_CLASSES = 65
OFFICEHOME_DEFAULT_N_VISIBLE_CLASSES = 30
OFFICEHOME_DEFAULT_N_INVISIBLE_CLASSES = 35

# ImageNet dataset
IMAGENET_DIR_PATH = os.path.join(ASSET_DIR_NAME, 'ImageNet')
IMAGENET_DATA_PATH = os.path.join(IMAGENET_DIR_PATH, 'data')
IMAGENET_IMAGE_LIST_PATH = os.path.join(IMAGENET_DIR_PATH, 'image_list')
IMAGENET_R_N_CLASSES = 200
IMAGENET_R_DEFAULT_N_VISIBLE_CLASSES = 100
IMAGENET_R_DEFAULT_N_INVISIBLE_CLASSES = 100
IMAGENET_S_N_CLASSES = 1000
IMAGENET_S_DEFAULT_N_VISIBLE_CLASSES = 500
IMAGENET_S_DEFAULT_N_INVISIBLE_CLASSES = 500

# Logging
LOGGER_FORMAT = "%(asctime)s - %(levelname)-5s - (%(filename)-10s: %(lineno)4d): %(message)s "
LOGGER_NAME = f'HT_{os.getpid()}'

# Space
SPACE_LEVELS = ['base_path', 'dataset', 'arch', 'source', 'target', 'model_config', 'n_visible_classes', 'visible_classes', 'n_invisible_classes', 'optimizer', 'optimizer_parameters', 'seed']
SPACE_DEPTH = len(SPACE_LEVELS)
