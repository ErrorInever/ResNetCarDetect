import argparse
import logging
import model
import time
from argparse import RawTextHelpFormatter


def create_parser():
    parser = argparse.ArgumentParser(prog='DenseNet', formatter_class=RawTextHelpFormatter)
    parser.add_argument("-d", "--dir", type=str, help="Root directory where the training data is stored.")
    parser.add_argument("-c", "--cuda", type=bool, help="Enable CUDA kernels (0 or 1) default False (0)", default=False)
    parser.add_argument("-e", "--epochs", type=int, help="Number of epochs.", default=1)
    parser.add_argument("-l", "--learning_rate", type=float, help="Number of step.", default=0.000025)
    parser.add_argument("-b", "--batch_size", type=int, help="Number of batch.", default=8)
    parser.add_argument("-w", "--workers", type=int, help="Number of train workers", default=8)
    parser.add_argument("-k", "--key", type=str, help="Api key of losswise", default='NA')
    args = parser.parse_args()

    parser.print_help()
    time.sleep(0.5)

    params = {
        "data_dir": args.dir,
        "cuda": args.cuda,
        "epochs": args.epochs,
        "lr": args.learning_rate,
        'bs': args.batch_size,
        "workers": args.workers,
        "key": args.key
    }

    if params["data_dir"] is None:
        raise TypeError('there is no data directory')
    return params


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

    logger.info(' Program started')

    params = create_parser()
    logger.info(' Arguments of cmd:{}'.format(params))

    model.DenseNet161.train_model(params)
