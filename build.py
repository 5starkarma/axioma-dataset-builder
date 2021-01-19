import os
import yaml
from tqdm import trange
from random import randrange

from loaders.background import BackgroundLoader
from loaders.alpha_mask import CardAlphaMask
from loaders.card import CardLoader

from scenes.scene import Scene
from utils.arg_parser import parse_args


class Builder:
    """
    Extracts cards and creates scenes.
    """
    def __init__(self, args):
        """
        :Params:
        --------
        cfg_file : str
            Path to project configuration file.

        :Returns:
        ---------
        self.cfg : dict
            A dictionary containing the configs.
        self.bg_loader : class
            Randomly loads background images from input dir.
        """
        self.args = args
        self.debug = self.args.debug
        self.num_scenes = self.args.size
        if self.debug:
            print('Running build...')

        with open(self.args.cfg_file, 'r') as file:
            self.cfg = yaml.load(file, Loader=yaml.FullLoader)

        self.bg_loader = BackgroundLoader(self.cfg, debug=self.debug)
        self.alpha_mask = CardAlphaMask(self.cfg, debug=self.debug)
        self.card_loader = CardLoader(self.cfg, debug=self.debug)
        self.scene = Scene(self.cfg, self.bg_loader, debug=self.debug)

    def create_output_directory(self):
        img_dir = self.cfg['output']['dir']
        lbl_dir = self.cfg['output']['dir'].replace('images', 'labels')
        dirs = [img_dir, lbl_dir]
        for folder in dirs:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def run(self):
        self.create_output_directory()
        for scene in trange(self.num_scenes):
            self.scene.num_cards_in_scene = randrange(self.scene.max_cards) + 1
            self.scene.create()
        # alpha_mask = self.alpha_mask.create()
        # img = self.card_loader.get_random_image(remove_bg=True)
        # self.card_loader.extract_all(alpha_mask)


if __name__ == '__main__':
    args = parse_args()
    builder = Builder(args)
    builder.run()
