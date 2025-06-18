import logging

def getLogger(name):
    logger = logging.getLogger(name)
    filename = name + '.log'
    fh = logging.FileHandler(filename, mode='w')
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return logger

def summarize(**kwargs):
    return kwargs