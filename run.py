import argparse
import logging
import model

if __name__ == '__main__':
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    # create the logging file handler
    fh = logging.FileHandler("densenet.log", mode='w')
    # create formatter and set it
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add handler to logger object
    logger.addHandler(fh)
    logging.info(' Program started')

    model.DenseNet161.train_model()


