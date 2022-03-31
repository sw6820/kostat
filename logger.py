import logging

def get_logger(name="root"):

    format = '%(asctime)s - %(name)s | %(levelname)s - %(message)s'
    # logging.basicConfig(format=format)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(format))

    file_handler = logging.FileHandler(filename="runtime_log.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(format))

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
   
    return logger