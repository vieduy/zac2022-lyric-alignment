import warnings, librosa
from librosa.core import yin
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import utils
from data import remove_accent
from model import train_audio_transforms, AcousticModel
import json
import argparse
from lib import nets
from lib import spec_utils
from lib import dataset
import json
from utils import json2txt, preprocess_from_file, preprocess_lyrics, preprocess_audio

np.random.seed(7)
new_rate = 22050
# constants
resolution = 256 / 22050 * 3
alpha = 0.8
n_fft = 2048

def main(args):
    ls_path_audio = args.ls_path_audio
    ls_json_lyrics = args.ls_json_lyrics
    ls_path_lyrics = args.ls_path_lyrics
    resolution = 256 / 22050 * 3
    ckp_path = args.ckp_path
    save_folder = args.save_folder
    cuda = True
    model_type="Baseline"

    # convert json to txt
    if not os.path.isdir(ls_path_lyrics):
      os.mkdir(ls_path_lyrics)
    ls_path = os.listdir(ls_json_lyrics)
    for i, path in enumerate(ls_path):
      if os.path.exists(os.path.join(ls_path_lyrics, path.replace(".json", ".txt"))):
        continue
      _ = json2txt(os.path.join(ls_json_lyrics, path), \
      os.path.join(ls_path_lyrics, path.replace(".json", ".txt")))

    # prepare acoustic model params
    if model_type == "Baseline":
        n_class = 41"

    hparams = {
        "n_cnn_layers": 1,
        "n_rnn_layers": 3,
        "rnn_dim": 256,
        "n_class": n_class,
        "n_feats": 32,
        "stride": 1,
        "dropout": 0.1
    }

    # Init Acoustic model
    print("Loading acoustic model from checkpoint...")
    device = 'cuda' if (cuda and torch.cuda.is_available()) else 'cpu'
    ac_model = AcousticModel(
        hparams['n_cnn_layers'], hparams['rnn_dim'], hparams['n_class'], \
        hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)
    state = utils.load_model(ac_model, ckp_path, cuda=(device=="gpu"))
    ac_model.eval()

    # Init vocal-remover models
    pretrained_model = "./checkpoints/vocal_baseline.pth"
    model_vocal = nets.CascadedNet(n_fft, 32, 128)
    model_vocal.load_state_dict(torch.load(pretrained_model, map_location=device))
    model_vocal.to(device)

    for i, path in enumerate(ls_path):
      # start timer
      t_s = time()
      print(f"Computing phoneme posteriorgram {path, i}...")
      path = path.replace("json", "txt")
    
      audio_file =  os.path.join(ls_path_audio, path.replace(".txt", ".wav"))
      lyrics_file = os.path.join(ls_path_lyrics, path)
      audio, words, lyrics_p, idx_word_p, idx_line_p = preprocess_from_file(audio_file, model_vocal, lyrics_file, word_file=None)
      x = audio.reshape(1, 1, -1)
      x = utils.move_data_to_device(x, device)
      x = x.squeeze(0).squeeze(1)
      x = train_audio_transforms.to(device)(x)
      x = nn.utils.rnn.pad_sequence(x, batch_first=True).unsqueeze(1)
      
      # predict
      all_outputs = ac_model(x)
      all_outputs = F.log_softmax(all_outputs, dim=2)

      batch_num, output_length, num_classes = all_outputs.shape
      song_pred = all_outputs.data.cpu().numpy().reshape(-1, num_classes)  # total_length, num_classes
      total_length = int(audio.shape[1] / 22050 // resolution)
      song_pred = song_pred[:total_length, :]

      # smoothing
      P_noise = np.random.uniform(low=1e-11, high=1e-10, size=song_pred.shape)
      song_pred = np.log(np.exp(song_pred) + P_noise)

      # start alignment
      word_align, score = utils.alignment(song_pred, lyrics_p, idx_word_p)

      t_end = time() - t_s
      print("Alignment Score:\t{}\tTime:\t{}".format(score, t_end))
      
      # Write json for submission
      json_lyrics = json.load(open(os.path.join(ls_json_lyrics, path.replace(".txt", ".json"))))
      id = 0
      lines_arr = []
      for line in json_lyrics:
        for word in line['l']:
            if remove_accent(word['d'].lower().strip()) == str(words[id]):
                word['s'] = int(word_align[id][0]*1000*resolution)
                word['e'] = int(word_align[id][1]*1000*resolution)
                id += 1
            else:
                continue
        
        line['s'] = line["l"][0]["s"]
        line['e'] = line["l"][-1]["e"]
        lines_arr.append(line)
        lines_arr = utils.fix_blank(lines_arr)
    
      # Saving...
      json_object = json.dumps(lines_arr, indent=4, ensure_ascii=False)
      if not os.path.exists(save_folder):
        os.mkdir(save_folder)
      with open(os.path.join(save_folder, path.replace(".txt", ".json")), "w", encoding="utf-8") as outfile:
          outfile.write(json_object)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ls_path_audio', type=str, default="./data/songs"
                        help='List audio vocals')
    parser.add_argument('--ls_path_lyrics', type=str, default="./data/new_labels_txt"
                        help='Where all the lyrics of the vocals.')
    parser.add_argument('--ls_json_lyrics', type=str, default="./data/new_labels_json"
                        help='Where all the json lyrics format for submit')
    parser.add_argument('--save_folder', type=str, required=False, default="./result/",
                        help='Saving all json for submit')
    parser.add_argument('--ckp_path', type=str, required=False, default="./checkpoints/checkpoint_best",
                        help='Checkpoint path for inference')
    args = parser.parse_args()
    main(args)
