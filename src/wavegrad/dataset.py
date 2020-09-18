# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import os
import random
import torch
import torchaudio

from glob import glob
from tqdm import tqdm

class NumpyDataset(torch.utils.data.Dataset):
  def __init__(self, params, paths):
    super().__init__()
    self.params = params
    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)
    pre_filenames = len(self.filenames)
    self.filenames = [filename for filename in tqdm(self.filenames) if len(np.load(self.get_spec_name(filename)).T) >= self.params.crop_mel_frames]
    post_filenames = len(self.filenames)
    print(f'{post_filenames} ({post_filenames/pre_filenames:.1%}) usable files in dataset.')
  
  def __len__(self):
    return len(self.filenames)
  
  def get_spec_name(self, audio_filename):
    return audio_filename.replace('.wav', '.npy')
  
  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    spec_filename = self.get_spec_name(audio_filename)
    signal, _ = torchaudio.load_wav(audio_filename)
    spectrogram = np.load(spec_filename)
    return {
        'audio': signal[0] / 32767.5,
        'spectrogram': spectrogram.T # [n_mel, enc_T]
    }


class Collator:
  def __init__(self, params):
    self.params = params

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    for record in minibatch:
      # Filter out records that aren't long enough.
      if len(record['spectrogram']) < self.params.crop_mel_frames:
        del record['spectrogram']
        del record['audio']
        continue
      
      start = random.randint(0, record['spectrogram'].shape[0] - self.params.crop_mel_frames)
      end = start + self.params.crop_mel_frames
      record['spectrogram'] = record['spectrogram'][start:end].T

      start *= samples_per_frame
      end *= samples_per_frame
      record['audio'] = record['audio'][start:end]
      record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')
    assert len([record for record in minibatch if 'audio' in record.keys()]), 'minibatch has no samples of sufficient length.'
    
    audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
    spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
    return {
        'audio': torch.from_numpy(audio),
        'spectrogram': torch.from_numpy(spectrogram),
    }


def from_path(data_dirs, params):
  return torch.utils.data.DataLoader(
      NumpyDataset(params, data_dirs),
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate,
      shuffle=True,
      num_workers=os.cpu_count())
