import numpy as np
import random
import time
from pathlib import Path
from pathy import Pathy
from torch.utils.data import IterableDataset

from tools import *


class OfflineDataSequential(IterableDataset):
    """Offline data which processes episodes sequentially"""

    def __init__(self, input_dir: str, batch_length, batch_size, skip_first=True, reload_interval=0):
        super().__init__()
        if input_dir.startswith('gs:/') or input_dir.startswith('s3:/'):
            self.input_dir = Pathy(input_dir)
        else:
            self.input_dir = Path(input_dir)
        self.batch_length = batch_length
        self.batch_size = batch_size
        self.skip_first = skip_first
        self.reload_interval = reload_interval
        self._reload_files()
        assert len(self._files) > 0, 'No data found'

    def _reload_files(self):
        print(f'Loading data from {str(self.input_dir)}...')
        self._files = list(sorted(self.input_dir.glob('*.npz')))
        self._last_reload = time.time()
        print(f'Found data: {len(self._files)} episodes')

    def _should_reload_files(self):
        return self.reload_interval and (time.time() - self._last_reload > self.reload_interval)

    def __iter__(self):
        # Parallel iteration over (batch_size) iterators
        # Iterates forever

        iters = [self._iter_single() for _ in range(self.batch_size)]
        for batches in zip(*iters):
            batch = {}
            for key in batches[0]:
                batch[key] = np.stack([b[key] for b in batches]).swapaxes(0, 1)
            yield batch

    def _iter_single(self):
        # Iterates "single thread" forever
        # TODO: join files so we don't miss the last step indicating done

        is_first = True
        for file in self._iter_shuffled_files():
            for batch in self._iter_file(file, self.batch_length, skip_random=is_first and self.skip_first):
                yield batch
            is_first = False

    def _iter_file(self, file, batch_length, skip_random=False):
        try:
            data = load_npz(file)
        except Exception as e:
            print('Error reading file - skipping')
            print(e)
            return

        # Undo the transformation for better compression
        if 'image' not in data and 'image_t' in data:
            data['image'] = data['image_t'].transpose(3, 0, 1, 2)  # CHWN => NCHW
            del data['image_t']

        n = data['image'].shape[0]
        data['reset'] = np.zeros(n, bool)
        data['reset'][0] = True  # Indicate episode start

        i_start = 0
        if skip_random:
            i_start = np.random.randint(n - batch_length)

        for i in range(i_start, n - batch_length + 1, batch_length):
            # TODO: should return last shorter batch
            j = i + batch_length
            batch = {key: data[key][i:j] for key in data}
            yield batch

    def _iter_shuffled_files(self):
        while True:
            i = random.randint(0, len(self._files) - 1)
            f = self._files[i]
            if not f.exists() or self._should_reload_files():
                self._reload_files()
            else:
                yield self._files[i]


class OfflineDataRandom:
    """Offline data with random sampling from middle of episodes"""

    def __init__(self, input_dir):
        input_dir = Path(input_dir)
        self._files = list(sorted(input_dir.glob('*.npz')))
        print(f'Offline data: {len(self._files)} episodes in {str(input_dir)}, loading...')
        self._data = self._load_data()
        assert len(self._data) > 0, 'No data found'

    def _load_data(self):
        all_data = []
        for path in self._files:
            data = load_npz(path)
            all_data.append(data)
        return all_data

    def iterate(self, batch_length, batch_size, skip_first=True):
        while True:
            yield self._sample_batch(batch_length, batch_size)

    def _sample_batch(self, batch_length, batch_size):
        sequences = [self._sample_single(batch_length) for _ in range(batch_size)]
        batch = {}
        for key in sequences[0]:
            batch[key] = np.stack([b[key] for b in sequences]).swapaxes(0, 1)
        return batch

    def _sample_single(self, batch_length):
        i_episode = np.random.randint(len(self._data))
        data = self._data[i_episode]

        # TODO: this sampling undersamples episode starts and ends
        n = data['image'].shape[0]
        i = np.random.randint(n - batch_length + 1)
        j = i + batch_length
        batch = {key: data[key][i:j] for key in data}
        return batch
