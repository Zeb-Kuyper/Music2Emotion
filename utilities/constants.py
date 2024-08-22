import torch
import os
# VERSION = "v1_xxx"
# VERSION = "v1.08_crnn"

MODEL = "AMT"
VERSION = "1.1_amt"

# MODE = "TRAIN"

CHECKPOINT = "tb_logs/train_audio_classification/version_4/checkpoints/last.ckpt"

BATCH_SIZE = 8

GENRE_CLASS_SIZE = 87
MOOD_CLASS_SIZE = 56

DAC_LATENTS_SIZE = 72
DAC_RVQ_SIZE = 9


