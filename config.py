import tensorflow as tf
import numpy as np
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.metrics import CategoricalAccuracy, AUC

class Config():
    def __init__(self):
        pass

    sample_rate = 16000
    audio_duration = 2
    stft_window_seconds = 0.025
    stft_hop_seconds = 0.010
    epsilon = 1e-30

    # dataloader_num_parallel = 1
    dataloader_num_parallel = tf.data.AUTOTUNE

    window_length_samples = int(round(sample_rate * stft_window_seconds))
    hop_length_samples = int(round(sample_rate * stft_hop_seconds))
    fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

    num_fbanks = 64

    save_root = "./output_yamnet_augment"
    background_folder = "/home/nuttawac/wakeword_batch1/background_for_augment"

    gpu_name = 0

    dataset = [
        # {
        #     "ds_root": "/home/nuttawac/lab_dataset/thai_ser/data/preprocess/wav_file",
        #     "train_df": "/home/nuttawac/lab_dataset/thai_ser/annotation/person_level/train.csv",
        #     "valid_df": "/home/nuttawac/lab_dataset/thai_ser/annotation/person_level/valid.csv",
        #     "test_df": "/home/nuttawac/lab_dataset/thai_ser/annotation/person_level/valid.csv"
        # }
        {
            "ds_root": "/home/nuttawac/lab_dataset/commonvoice11/data/clips_wav",
            "train_df": "/home/nuttawac/beam/commonvoice11/train.csv",
            "valid_df": "/home/nuttawac/beam/commonvoice11/dev.csv",
            "test_df": "/home/nuttawac/beam/commonvoice11/test.csv"
        }
    ]

    model = {
        "model_path": "/home/nuttawac/nlp-gender-classification/output_yamnet_augment/model_w",
        "model_name": "09-0.21",
        "load_weights": True,
        "class_name": ["Male", "Female"],
        "train_topn_layer": False
    }

    hyperparameters = {
        "optimizer": Adam(learning_rate=1e-4),
        "metrics": [CategoricalAccuracy(), AUC()],
        "loss": CategoricalCrossentropy(from_logits=False),
        "batch_size": 32,
        "n_epoch": 20
    }
    
    seed = 6024