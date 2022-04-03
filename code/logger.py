import logging


def get_logger(name):
    
    format = '%(asctime)s - %(name)s | %(levelname)s - %(message)s'
    logger = logging.getLogger(name)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter(format))

    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)
    
    return logger