import os
import logging

# Path
ASSET_DIR_NAME = 'asset'
OUT_DIR_NAME = 'out'
EXPERIMENT_DIR_NAME = 'experiment'
EXPERIMENT_PATH = os.path.join(OUT_DIR_NAME, EXPERIMENT_DIR_NAME)

# Setting
EVALUATE_TOP_K = (1, 5)

# OfficeHome dataset
OFFICEHOME_DIR_PATH = os.path.join(ASSET_DIR_NAME, 'OfficeHome')
OFFICEHOME_DATA_PATH = os.path.join(OFFICEHOME_DIR_PATH, 'data')
OFFICEHOME_IMAGE_LIST_PATH = os.path.join(OFFICEHOME_DIR_PATH, 'image_list')

# Logging
LOG_FORMAT = "%(asctime)s - %(levelname)-5s - (%(filename)-10s: %(lineno)4d): %(message)s "

# Space
SPACE_LEVELS = ['base_path', 'dataset', 'arch', 'source', 'target', 'n_seen_classes', 'model_config', 'optimizer', 'optimizer_parameters', 'seed']
SPACE_DEPTH = len(SPACE_LEVELS)
