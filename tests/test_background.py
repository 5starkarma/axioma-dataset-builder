from loaders.background import BackgroundLoader


class TestBackgroundLoader(unittest.TestCase):

    def setUp(self):
        with open('dataset/tests/test_configs.yml', 'r') as file:
            self.cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.bg_loader = BackgroundLoader(self.cfg)
        url = 'https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz'
        output_directory = 'dataset/tests/test_data'
        self.dtd = wget.download(url, out=output_directory)

    def tearDown(self) -> None:
        pass

    def test_get_list_of_files(self):
        assert isinstance(self.bg_loader.get_list_of_files(), list)
        assert hasattr(self.bg_loader.get_list_of_files(), '__len__')

    def test_read_image(self):
        files = self.bg_loader.get_list_of_files()
        assert isinstance(self.bg_loader.read_image(files), (list, np.ndarray))

    def test_resize_img(self):
        dims = self.cfg['output']['dims']
        
    # Test input shape [0] == 300, shape [1] == 300
    # Test output shape [0] == 1280, shape [1] == 720


    
