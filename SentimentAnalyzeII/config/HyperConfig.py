import configparser


class Config(object):
    def __init__(self, conf_path):
        conf = configparser.ConfigParser()
        conf.read(conf_path, encoding='utf-8')

        for sec in conf.sections():
            print(sec)
            for k, v in conf.items(sec):
                print(k, ':', v)

        self._conf = conf
        self._use_cuda = False
        self._device = None

    @property
    def use_cuda(self):
        return self._use_cuda

    @use_cuda.setter
    def use_cuda(self, use_cuda):
        self._use_cuda = use_cuda

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @property
    def train_data_path(self):
        return self._conf.get('Data', 'train_data_path')

    @property
    def dev_data_path(self):
        return self._conf.get('Data', 'dev_data_path')

    @property
    def test_data_path(self):
        return self._conf.get('Data', 'test_data_path')

    @property
    def lexicon_path(self):
        return self._conf.get('Data', 'lexicon_path')

    @property
    def embedding_path(self):
        return self._conf.get('Data', 'embedding_path')

    @property
    def save_vocab_path(self):
        return self._conf.get('Save', 'save_vocab_path')

    @property
    def load_vocab_path(self):
        return self._conf.get('Save', 'load_vocab_path')

    @property
    def save_model_path(self):
        return self._conf.get('Save', 'save_model_path')

    @property
    def load_model_path(self):
        return self._conf.get('Save', 'load_model_path')

    @property
    def learning_rate(self):
        return self._conf.getfloat('Optimizer', 'learning_rate')

    @property
    def weight_decay(self):
        return self._conf.getfloat('Optimizer', 'weight_decay')

    @property
    def nb_layers(self):
        return self._conf.getint('Network', 'nb_layers')

    @property
    def max_len(self):
        return self._conf.getint('Network', 'max_len')

    @property
    def hidden_size(self):
        return self._conf.getint('Network', 'hidden_size')

    @property
    def batch_size(self):
        return self._conf.getint('Network', 'batch_size')

    @property
    def drop_rate(self):
        return self._conf.getfloat('Network', 'drop_rate')

    @property
    def drop_embed_rate(self):
        return self._conf.getfloat('Network', 'drop_embed_rate')

    @property
    def epochs(self):
        return self._conf.getint('Network', 'epochs')

    @property
    def theta(self):
        return self._conf.getfloat('Network', 'theta')
