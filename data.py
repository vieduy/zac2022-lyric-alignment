import h5py
import os
import numpy as np
from sortedcontainers import SortedList
from torch.utils.data import Dataset
import string
from tqdm import tqdm
import logging
from utils import load, load_lyrics, gen_phone_gt, ToolFreq2Midi
from tqdm import tqdm
from g2p_en import G2p
import glob
import json
import sys
import re
from sklearn.model_selection import KFold

g2p = G2p()
phone_dict = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY',
             'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y',
             'Z', 'ZH', ' ']
phone2int = {phone_dict[i]: i for i in range(len(phone_dict))}

def getZalo(database_path, vocal_path, public_test=True):
    zalo_annot_path = os.path.join(database_path, 'labels')
    
    subset = list()
    total_line_num = 0
    discard_line_num = 0

    zalo_annots = glob.glob(os.path.join(zalo_annot_path, "*"))
    if public_test:
      zalo_annots += glob.glob(os.path.join(database_path, "public_labels/public_pseudo_label", "*"))
    
    for zalo_anno in tqdm(zalo_annots):
      uuid = zalo_anno.split("/")[-1].replace('.json', '')
      song = {"id": uuid, "words": [], \
              "path": zalo_anno, \
              "vocal_path": os.path.join(vocal_path, uuid + "_Vocals.wav")}
      file = open(zalo_anno)
      gt = json.load(file)
      words = []
      phonemes_encode = []
      max_phone = -1
      for line in gt:
        for word in line['l']:
          sample = {}
          start = word["s"]/1000
          end = word["e"]/1000
          sample["duration"] = end - start
          if sample["duration"] < 0:
            continue
          sample["time"] = [start, end]
          word["d"] = remove_accent(word["d"].lower())
          sample["text"] = re.sub('[^a-zA-Z0-9 \n\.]', '', word["d"])
          if len(sample["text"].split(' ')) > 1:
            continue
          if sample["duration"] > 7: # remove words which are too long
            discard_line_num += 1
          words.append(sample)
          total_line_num += 1
          phoneme = g2p(sample["text"])
          if len(phoneme) > max_phone:
              max_phone = len(phoneme)
          phoneme = [s.encode() for s in phoneme]
          phonemes_encode.append(phoneme)
      if max_phone < 0:
        continue
      song["words"] = words
      song["phonemes"] = phonemes_encode
      song["max_phone"] = max_phone

      subset.append(song)

    return np.array(subset, dtype=object)

def get_zalo_folds(database_path, vocal_path, public_test=True):
    dataset = getZalo(database_path, vocal_path, public_test=True)

    total_len = len(dataset)
    train_len = np.int(0.9 * total_len)

    train_list = np.random.choice(dataset, train_len, replace=False)
    val_list = [elem for elem in dataset if elem not in train_list]

    return {"train" : train_list, "val" : val_list}


class LyricsAlignDataset(Dataset):
    def __init__(self, dataset, partition, sr, input_sample, hdf_dir, in_memory=False):
        super(LyricsAlignDataset, self).__init__()

        self.hdf_dataset = None
        os.makedirs(hdf_dir, exist_ok=True)
        self.hdf_file = os.path.join(hdf_dir, partition + ".hdf5")

        self.sr = sr
        self.input_sample = input_sample
        self.hop = (input_sample // 2)
        self.in_memory = in_memory

        # Check if HDF file exists already
        if not os.path.exists(self.hdf_file):
            # Create folder if it did not exist before
            if not os.path.exists(hdf_dir):
                os.makedirs(hdf_dir)

            # Create HDF file
            with h5py.File(self.hdf_file, "w") as f:
                f.attrs["sr"] = sr

                print("Adding audio files to dataset (preprocessing)...")
                for idx, example in enumerate(tqdm(dataset[partition])):
                    # Load song
                    y, _ = load(example["vocal_path"], sr=self.sr, mono=True)
                    annot_num = len(example["words"])

                    # Add to HDF5 file
                    grp = f.create_group(str(idx))
                    grp.create_dataset("inputs", shape=y.shape, dtype=y.dtype, data=y)

                    grp.attrs["audio_name"] = example["id"]
                    grp.attrs["input_length"] = y.shape[1]
                    grp.attrs["annot_num"] = annot_num

                    # word level annotation
                    annot_num = len(example["words"])
                    lyrics = [sample["text"].encode() for sample in example["words"]]
                    times = np.array([sample["time"] for sample in example["words"]])

                    # phoneme
                    phonemes_encode = example["phonemes"]
                    max_phone = example["max_phone"]


                    # words and corresponding times
                    grp.create_dataset("lyrics", shape=(annot_num, 1), dtype='S100', data=lyrics)
                    grp.create_dataset("times", shape=(annot_num, 2), dtype=times.dtype, data=times)
                    grp.create_dataset("phonemes", shape=(annot_num, max_phone), dtype='S2')
                    for i in range(annot_num):
                        phonemes_sample = phonemes_encode[i]
                        grp["phonemes"][i, :len(phonemes_sample)] = np.array(phonemes_sample)

        # In that case, check whether sr and channels are complying with the audio in the HDF file, otherwise raise error
        with h5py.File(self.hdf_file, "r", libver='latest', swmr=True) as f:
            if f.attrs["sr"] != sr:
                raise ValueError(
                    "Tried to load existing HDF file, but sampling rate is not as expected.")

        # Go through HDF and collect lengths of all audio files
        with h5py.File(self.hdf_file, "r") as f:

            # length of song
            lengths = [f[str(song_idx)].attrs["input_length"] for song_idx in range(len(f))]
            # Subtract input_size from lengths and divide by hop size to determine number of starting positions
            lengths = [( (l - input_sample) // self.hop) + 1 for l in lengths]

        self.start_pos = SortedList(np.cumsum(lengths))
        self.length = self.start_pos[-1]

    def __getitem__(self, index):

        # open HDF5
        if self.hdf_dataset is None:
            driver = "core" if self.in_memory else None  # Load HDF5 fully into memory if desired
            self.hdf_dataset = h5py.File(self.hdf_file, 'r', driver=driver)

        while True:
            # Loop until it finds a valid sample

            # Find out which slice of targets we want to read
            song_idx = self.start_pos.bisect_right(index)
            if song_idx > 0:
                index = index - self.start_pos[song_idx - 1]

            # length of audio signal
            audio_length = self.hdf_dataset[str(song_idx)].attrs["input_length"]
            # number of words in this song
            annot_num = self.hdf_dataset[str(song_idx)].attrs["annot_num"]

            # determine where to start
            start_pos = index * self.hop
            end_pos = start_pos + self.input_sample

            # front padding
            if start_pos < 0:
                # Pad manually since audio signal was too short
                pad_front = abs(start_pos)
                start_pos = 0
            else:
                pad_front = 0

            # back padding
            if end_pos > audio_length:
                # Pad manually since audio signal was too short
                pad_back = end_pos - audio_length
                end_pos = audio_length
            else:
                pad_back = 0

            # read audio and zero padding
            audio = self.hdf_dataset[str(song_idx)]["inputs"][0, start_pos:end_pos].astype(np.float32)
            if pad_front > 0 or pad_back > 0:
                audio = np.pad(audio, [(0, 0), (pad_front, pad_back)], mode="constant", constant_values=0.0)

            # find the lyrics within (start_target_pos, end_target_pos)
            words_start_end_pos = self.hdf_dataset[str(song_idx)]["times"][:]
            try:
                first_word_to_include = next(x for x, val in enumerate(list(words_start_end_pos[:, 0]))
                                             if val > start_pos/self.sr)
            except StopIteration:
                first_word_to_include = np.Inf

            try:
                last_word_to_include = annot_num - 1 - next(x for x, val in enumerate(reversed(list(words_start_end_pos[:, 1])))
                                             if val < end_pos/self.sr)
            except StopIteration:
                last_word_to_include = -np.Inf

            targets = ""
            phonemes_list = []
            if first_word_to_include - 1 == last_word_to_include + 1: # the word covers the whole window
                # invalid sample, skip
                targets = None
                index = np.random.randint(self.length)
                continue
            if first_word_to_include <= last_word_to_include: # the window covers word[first:last+1]
                # build lyrics target
                lyrics = self.hdf_dataset[str(song_idx)]["lyrics"][first_word_to_include:last_word_to_include+1]
                lyrics_list = [s[0].decode() for s in list(lyrics)]
                targets = " ".join(lyrics_list)
                targets = " ".join(targets.split())

                phonemes = self.hdf_dataset[str(song_idx)]["phonemes"][first_word_to_include:last_word_to_include+1]
                phonemes_list = self.convert_phone_list(phonemes)

            seq = self.text2seq(targets)
            phone_seq = self.phone2seq(phonemes_list)
            min_indx = np.inf
            times = []
            for i, (start, end) in enumerate(words_start_end_pos):
              if (start > start_pos/self.sr) and (end < end_pos/self.sr):
                min_indx = min(i, min_indx)
                times.append([start, end])
            try:
              times -= start_pos/self.sr
            except:
              times = []
            break

        return audio, targets, seq, phone_seq

    def text2seq(self, text):
      text = remove_accent(text)
      seq = []
      for c in text.lower():
          idx = string.ascii_lowercase.find(c)
          if idx == -1:
              if c == "'":
                  idx = 26
              elif c == " ":
                  idx = 27
              else:
                  continue # remove unknown characters
          seq.append(idx)
      return np.array(seq)

    def phone2seq(self, text):
        seq = []
        for c in text:
          try:
            idx = phone2int[c]
            seq.append(idx)
          except Exception as e:
            continue
        return np.array(seq)

    def convert_phone_list(self, phonemes):
        ret = []
        for l in phonemes:
            l_decode = [' '] + [s.decode() for s in l if len(s) > 0]
            ret += l_decode
        if len(ret) > 1:
            return ret[1:]
        else:
            return []

    def __len__(self):
        return self.length

def remove_accent(txt):
  replace = False
  s = ''
  for c in txt:
    replace = False
    for key in dict_1.keys():
      if c in key:
        s += dict_1[key]
        replace = True
        break
    if replace:
      pass
    else:
      s += c
  return s

dict_1 = {
    "àáãạả": "a",
    "ấầẫậẩâ": "a",
    "ăắặằẳă": "a",
    "ễếềệểê": "e",
    "ẹẽèéẻ": "e",
    "ữựừứửư": "u",
    "úùụủũ": "u",
    "ởớờợỡơ": "o",
    "ổỗộồốô": "o",
    "íìịĩỉ": "i",
    "ỏõọóò": "o",
    "ýỳỵỹỷ": "y",
    "đ": "d"
}
