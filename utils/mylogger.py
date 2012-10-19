__author__ = 'makov'

import logging


def set_logger(log_level=logging.INFO):
    logging.basicConfig(level=log_level,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        filename='preprocess.log',
                        filemode='w')
    console_logger = logging.StreamHandler()
    console_logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_logger.setFormatter(formatter)
    logging.getLogger('').addHandler(console_logger)


def add_logging_file(log_file, logger_id=None):
    if logger_id is None:
        logger_name = ''
    else:
        logger_name = str(logger_id)
    logger = logging.getLogger(logger_name)
    my_log_handler = logging.FileHandler(log_file, mode='w')
    #formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    my_log_handler.setFormatter(formatter)
    logger.addHandler(my_log_handler)
    logger.setLevel(logging.INFO)
    return my_log_handler