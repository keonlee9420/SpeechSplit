import os
import sys
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from utils import butter_highpass
from utils import speaker_normalization
from utils import pySTFT
import time

# sys.path.append('/ssd2/FastStyle')
# sys.path.append('/ssd2/FastStyle/data')
# from libritts import get_unaligned_wavs
from tqdm import tqdm
from pathlib import Path

############## hparams ##############
np.random.seed(9420)
lo, hi = 71.0, 799.9

mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)
#####################################

# spk2gen = pickle.load(open('assets/spk2gen.pkl', "rb"))
 
# Modify as needed
rootDir = '/home/keon/speech-datasets/LibriTTS_preprocessed/train-clean-100' # 'assets/wavs'
targetDir_f0 = 'assets/raptf0'
targetDir = 'assets/spmel'

in_dir = rootDir
spkers = os.listdir(in_dir)
spkers.sort()

def get_unaligned_wavs(out_dir):
    unaligned=list()
    with open(os.path.join(out_dir, 'output_errors.txt'), encoding='utf-8') as f:
        all_txt = f.read()
        all_txt = all_txt.split(":\nTraceback")
        unaligned += [t.split('\n')[-1] for t in all_txt if 'CB' in t]
    with open(os.path.join(out_dir, 'unaligned.txt'), encoding='utf-8') as f:
        for line in f:
            unaligned.append(line.strip().split(' ')[0].split('\t')[0])
    return unaligned

unaligned_basenames = get_unaligned_wavs('/ssd2/FastStyle/preprocessed/LibriTTS')
print("Total unaligned wavs: {}".format(len(unaligned_basenames)))

if __name__ == "__main__":
    if not os.path.exists(os.path.join(targetDir)):
        os.makedirs(os.path.join(targetDir))
    if not os.path.exists(os.path.join(targetDir_f0)):
        os.makedirs(os.path.join(targetDir_f0))

    start_time_total = time.perf_counter()
    for spker in tqdm(spkers):
        spker_dir = os.path.join(in_dir, spker)
        spker_id = spker_dir.split('/')[-1]
        # print(spker_id)

        file_paths = []
        for dirpath, dirnames, filenames in os.walk(spker_dir):
            # print('Found directory: %s' % dirnames)
            for f in filenames:
                if f.endswith(".normalized.txt"):
                    if f.replace(".normalized.txt", "") in unaligned_basenames:
                        continue
                    subdir = Path(dirpath).relative_to(in_dir)
                    file_paths.append((subdir, f))

        prng = RandomState(int(spker_id))
        np.random.shuffle(file_paths)

        for i, file_path in enumerate(file_paths):
            subdir = file_path[0]
            filename = file_path[1]
            basename = filename.replace(".normalized.txt", "")
            wav_path = os.path.join(in_dir, subdir, '{}.wav'.format(basename))

            # read audio file
            try:
                x, fs = sf.read(wav_path)
            except Exception as e:
                # print("Error on {}".format(basename))
                print(e) if 'System error' not in str(e) else None # preprocessed dir can have no wav file due to the lenght constraint.
                continue
            # assert fs == 16000
            if x.shape[0] % 256 == 0:
                x = np.concatenate((x, np.array([1e-06])), axis=0)
            y = signal.filtfilt(b, a, x)
            wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
            
            # compute spectrogram
            D = pySTFT(wav).T
            D_mel = np.dot(D, mel_basis)
            D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
            S = (D_db + 100) / 100        
            
            # extract f0
            f0_rapt = sptk.rapt(wav.astype(np.float32)*32768, fs, 256, min=lo, max=hi, otype=2)
            index_nonzero = (f0_rapt != -1e10)
            mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
            f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
            
            assert len(S) == len(f0_rapt)
                
            np.save(os.path.join(targetDir, basename),
                    S.astype(np.float32), allow_pickle=False)
            np.save(os.path.join(targetDir_f0, basename),
                f0_norm.astype(np.float32), allow_pickle=False)
    print("All done in {:.3f}s".format(time.perf_counter()-start_time_total))