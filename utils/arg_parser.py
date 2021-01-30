import argparse


def parse_args():
    """
    Argument parser for cli.

    :Returns:
    ---------
    args : ArgumentParser object
        Contains all the cli arguments
    """
    parser = argparse.ArgumentParser(description='Facial detection and \
                                     smoothing using OpenCV.')
    parser.add_argument('--cfg_file', 
                        type=str, 
                        help='Project configuration file',
                        default='configs/configs.yaml')
    parser.add_argument('--debug', 
                        dest='debug',
                        help='Debug mode',
                        action='store_true')
    parser.add_argument('--single',
                        dest='single',
                        help='Single cards dataset builder.',
                        action='store_true')
    parser.add_argument('--blackjack',
                        dest='blackjack',
                        help='Blackjack hand dataset builder.',
                        action='store_true')
    parser.add_argument('--extract',
                        dest='extract',
                        help='Extracts all card images.',
                        action='store_true')
    parser.add_argument('--size', 
                        nargs='?',
                        default=10000,
                        type=int,
                        help='Size of dataset to be output')            
    return parser.parse_args()
