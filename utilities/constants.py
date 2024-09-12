import torch
import os

# --------------------------------------- #
VERSION = "1.23"

# --------------------------------------- #
ENCODER = "MERT"

# - - - 
# MERT
# M2L
# LIBROSA
# - - - 
# Encodec
# DAC

# --------------------------------------- #

SEGMENT = "all"
# all
# f10s - first 10s 
# f30s - first 10s 

# 10s
# 30s


AGGREGATION_METHOD = "mean"
# mean
# median
# 80th_percentile
# max

# --------------------------------------- #
CLASSIFIER = "linear-mt-multitask"
# transformer
# linear
# linear-small
# linear-multitask
# linear-small-multitask
# linear-mt (mert-like classifier)
#
# --------------------------------------- #
CHECKPOINT = "tb_logs/train_audio_classification/version_109/checkpoints/42-0.3631.ckpt"
# --------------------------------------- #
BATCH_SIZE = 8

GENRE_CLASS_SIZE = 87
MOOD_CLASS_SIZE = 56
INSTR_CLASS_SIZE = 40

DAC_LATENTS_SIZE = 72
DAC_RVQ_SIZE = 9
# --------------------------------------- #
