import torch
import os

class DataHandler():
    def __init__(self):
        self.text_data = ''

    def load_dir(self, dir_path):
        filenames = os.listdir(dir_path)

        for filename in filenames:
            if os.path.isfile(dir_path + '/' + filename):
                with open(dir_path + '/' + filename, 'r', encoding='utf-8') as f:
                    self.text_data += f.read()

    def load_file(self, filepath):

        with open(filepath, 'r', 'utf-8') as f:
            self.text_data += f.read()

    def get_unique_chars(self, verbose=False):
        chars = sorted(list(set(self.text_data)))
        unique_chars = ''.join(chars)

        if verbose:
            print('Number of unique characters =', len(unique_chars))
            print('Unique chars:', unique_chars)

        return unique_chars

    def show_data(self, seq_length=100):
        return self.text_data[:seq_length]

    def data_to_tokens(self, encode):
        self.tokens_data = torch.tensor(encode(self.text_data), dtype=torch.long)

    def train_val_split(self, train_size):
        n = int(train_size*len(self.tokens_data))
        self.train_data = self.tokens_data[:n]
        self.val_data = self.tokens_data[n:]

    def get_batch(self, split, cfg):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([data[i:i+cfg.block_size] for i in ix])
        y = torch.stack([data[i+1:i+cfg.block_size+1] for i in ix])
        x, y = x.to(cfg.device), y.to(cfg.device)
        return x, y