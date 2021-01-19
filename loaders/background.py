import os, random, glob, wget, tarfile

import cv2

from configs import ROOT_DIR


class BackgroundLoader:
    """
    Randomly loads background images from input dir.
    """
    def __init__(self, cfg, debug=False):
        """
        :Params:
        ------------
        img_dir : str
            Parent directory containing bg images.
        dims : tuple
            Width and height of output e.g. (1280, 720)
        """
        self.cfg = cfg
        self.debug = debug

        if not os.path.isdir(os.path.join(ROOT_DIR, 'data/dtd')):
            url = self.cfg['bg']['dtd_link']
            os.chdir(os.path.join(ROOT_DIR, 'data'))
            print('Downloading DTD dataset...')
            fname = wget.download(url)
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
        else:
            print('DTD already downloaded... continuing...')
        os.chdir(ROOT_DIR)

        self.img_dir = self.cfg['bg']['dtd_dir']
        if self.debug:
            print(f'DTD image dir: {self.img_dir}')
        
        self.dims = self.cfg['output']['dims']
    
    def get_list_of_files(self):
        """
        Gets all filenames from directory and all sub-directories.

        :Returns:
        ---------
        file_list : list
            List of paths to files.
        """
        # Init empty list
        file_list = list()
        # Get all folders and files
        for root, dirs, files in os.walk(self.img_dir):
            # For files
            for file in files:
                # Append to file list
                file_list.append(os.path.join(root, file))
        return file_list

    def resize_img(self, img):
        """
        Resizes image to input dimensions.

        :Params:
        --------
        img : np.array [H, W, 3]
            Image to resize.
        
        :Returns:
        ---------
        img : np.array [H, W, 3]
            Resized image.
        """
        return cv2.resize(img, self.dims, interpolation=cv2.INTER_AREA)

    def get_num_images(self):
        """
        Retrieves amount of files.

        :Returns:
        ---------
        num_files : int
            Number of files.
        """
        num_images = int(len(self.get_list_of_files()))
        if self.debug: print(f'Image count: {num_images}')
        return num_images

    def get_random_idx(self):
        """
        Retrieves a random index of file in file list.

        :Returns:
        ---------
        idx : int
            Random index of file.
        """
        idx = random.randint(0, self.get_num_images() - 1)
        if self.debug: print(f'Background random idx: {idx}')
        return idx

    def read_image(self, files):
        """
        Reads random image file.

        :Params:
        --------
        files : list
            List of files.

        :Returns:
        ---------
        image : np.array [H, W, 3]
        """
        return cv2.imread(files[self.get_random_idx()])

    def get_random(self, display=False):
        """
        Retrieves random image from list of all images.

        :Params:
        --------
        display : bool
            Whether to display an image.

        :Returns:
        ---------
        resized_bg : np.array [H, W, 3]
        """
        files = self.get_list_of_files()

        for attempt in range(20):
            try:
                bg = self.read_image(files)
                resized_bg = self.resize_img(bg)
            except:
                print('Image failed to resize. Retrying..')
            else:
                break

        if display or self.debug: 
            print(f'Original bg shape: {bg.shape}')
            print(f'Resized bg shape: {resized_bg.shape}')
            debug_dir = self.cfg['debug']['dir']
            files = glob.glob(os.path.join(debug_dir, '*'))
            for f in files:
                os.remove(f)
            debug_file = os.path.join(debug_dir, 'resized_bg.jpg')
            cv2.imwrite(debug_file, resized_bg)
            print(f'Background saved: {debug_file}')
            print('--------------------------------------')

        return resized_bg

