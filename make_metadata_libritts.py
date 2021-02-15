import os
import pickle
import numpy as np
from make_spect_f0_libritts import in_dir, spkers, unaligned_basenames
from tqdm import tqdm
from pathlib import Path
import time

rootDir = 'assets/spmel'
spkerEmbeddDir = '/ssd2/FastStyle/preprocessed/LibriTTS/spker_embed'

speakers_train = []
speakers_val = []
print("Divide spker_ids into train and val set...")
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

    np.random.shuffle(file_paths)
    for i, file_path in enumerate(file_paths):
        filename = file_path[1]
        basename = filename.replace(".normalized.txt", "")
        
        if i == 0: 
            speakers_val.append(basename + ".npy")
        else:
            speakers_train.append(basename + ".npy")

def gather_info(mel_filename):
    # print('Processing mel: %s' % mel_filename)
    spker_id = int(mel_filename.strip().split('_')[0])
    utterances = []
    utterances.append(spker_id)
    
    # use hardcoded onehot embeddings in order to be cosistent with the test mels
    # modify as needed
    # may use generalized mel embedding for zero-shot conversion
    spker_embed = np.load(os.path.join(spkerEmbeddDir, 'LibriTTS-spker_embed-{}.npy'.format(spker_id))).astype('float32')
    utterances.append(spker_embed)
    
    # create file list
    utterances.append(os.path.join(mel_filename))
    return utterances

train = []
val = []
print("Gather information of both train and val set respectively...")
for mel_filename in tqdm(sorted(os.listdir(rootDir))):
    if mel_filename in speakers_train:
        train.append(gather_info(mel_filename))
    elif mel_filename in speakers_val:
        val.append(gather_info(mel_filename))

print("Save metadatas...")
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(train, handle)
with open(os.path.join(rootDir, 'val.pkl'), 'wb') as handle:
    pickle.dump(val, handle)
print("All done in {:.3f}s".format(time.perf_counter()-start_time_total))