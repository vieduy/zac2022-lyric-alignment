from tqdm import tqdm
import utils, time
import numpy as np
import torch, os
import torch.nn as nn
import torch.nn.functional as F
from model import train_audio_transforms
from utils import preprocess_lyrics, iou_zalo
from scipy.signal import medfilt
from utils import alignment


def validate(batch_size, model, target_frame, criterion, dataloader, device, model_type, loss_w=0.1):
    resolution = 256 / 22050 * 3
    avg_time = 0.
    model.eval()

    total_loss = 0.
    data_len = len(dataloader.dataset) // batch_size

    with tqdm(total=data_len) as pbar:
        for batch_idx, _data in enumerate(dataloader):
            spectrograms, phones, input_lengths, phone_lengths = _data
            spectrograms, phones = spectrograms.to(device), phones.to(device)
            t = time.time()

            output_phone = model(spectrograms)  # (batch, time, n_class)
            output_phone = F.log_softmax(output_phone, dim=2)
            
            loss = criterion(output_phone.transpose(0, 1), phones, input_lengths, phone_lengths)

            t = time.time() - t
            avg_time += (1. / float(batch_idx + 1)) * (t - avg_time)

            total_loss += loss.item()

            pbar.set_description("Current loss: {:.4f}".format(loss))
            pbar.update(1)

            if batch_idx == data_len:
                break

    return total_loss / data_len, total_loss / data_len, None
