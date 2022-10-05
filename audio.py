import os
import numpy as np
import librosa
from python_speech_features import fbank
import tensorflow as tf
from audiomentations import (SomeOf,
                             AddGaussianNoise,
                             AddGaussianSNR,
                             AddBackgroundNoise,
                             TimeStretch,
                             PitchShift,
                             Compose,
                             OneOf
                            )

class Audio():
    # https://github.com/philipperemy/deep-speaker/blob/master/audio.py
    def __init__(self, config, df, mode="inference"):
        """Audio preprocessing object.

        Attributes
        ----------
        config : object
            Config for training
        df : pd.DataFrame
            DataFrame of the selected sample to preprocess
        mode : str
            Currently support train, valid, inference
        
        Methods
        -------
        get_item
            Return processed audio file and its ground-truth.
            if mode set to "train" audio file will be augmented
        """
        self.config = config
        self.df = df
        self.mode = mode
        np.random.seed(self.config.seed)
        if self.mode == "train":
            background_path = [os.path.join(self.config.background_folder,f) \
                for f in os.listdir(self.config.background_folder)
                ]
            gaussian_noise = [AddGaussianSNR(p=1),
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1)
                    ]
            background_noise = [AddBackgroundNoise(p=1, sounds_path=f) for f in background_path]
            self.augment = Compose([
                OneOf(
                    gaussian_noise+background_noise
                    ),
                # Compose([
                #     TimeStretch(p=0.5),
                #     PitchShift(p=0.5)
                # ])
            ])

    def get_item(self, i):
        i = i.numpy()
        # x = self._read_mfcc(i)
        x = self.tranfrom_data(i)
        # x = self._log_mel_spectrogram_3(i)
        y = self._extract_gt(i)
        
        return x, y

    def _extract_gt(self, i):
        batch_df = self.df.loc[i]
        
        return batch_df[self.config.model["class_name"]].values

    def _log_mel_spectrogram(self, i):
        input_filename = self.df.loc[i, "File Name"]
        waveform, sr = self._read(input_filename)
        # waveform += self.config.epsilon
        mel_signal = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=self.config.fft_length,
            win_length=self.config.window_length_samples,
            hop_length=self.config.hop_length_samples,
            htk=True,
            window='hann'
            )
        
        mel_spectrogram = np.abs(mel_signal)
        log_mel_spectrogram = np.log(mel_spectrogram + self.config.epsilon) 
        log_mel_spectrogram = np.expand_dims(log_mel_spectrogram, axis=-1)
        
        return log_mel_spectrogram

    def _log_mel_spectrogram_2(self, i):
        input_filename = self.df.loc[i, "File Name"]
        waveform, sr = self._read(input_filename)

        num_spectrogram_bins = self.config.fft_length // 2 + 1
        magnitude_spectrogram = tf.abs(tf.signal.stft(
            signals=waveform,
            frame_length=self.config.window_length_samples,
            frame_step=self.config.hop_length_samples,
            fft_length=self.config.fft_length
            ))
        
        # magnitude_spectrogram has shape [<# STFT frames>, num_spectrogram_bins]
        # Convert spectrogram into log mel spectrogram.
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=128,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sr,
            lower_edge_hertz=125,
            upper_edge_hertz=7500)
        
        mel_spectrogram = tf.matmul(magnitude_spectrogram, linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + self.config.epsilon)
        log_mel_spectrogram = tf.expand_dims(log_mel_spectrogram, axis=-1)
        # log_mel_spectrogram has shape [<# STFT frames>, params.mel_bands]

        # Frame spectrogram (shape [<# STFT frames>, params.mel_bands]) into patches
        # (the input examples). Only complete frames are emitted, so if there is
        # less than params.patch_window_seconds of waveform then nothing is emitted
        # (to avoid this, zero-pad before processing).
        """
        spectrogram_hop_length_samples = int(round(sr * self.config.stft_hop_seconds))
        spectrogram_sample_rate = sr / spectrogram_hop_length_samples
        patch_window_length_samples = int(round(spectrogram_sample_rate * self.config.stft_window_seconds))
        patch_hop_length_samples = int(round(spectrogram_sample_rate * self.config.stft_hop_seconds))
        features = tf.signal.frame(
            signal=log_mel_spectrogram*2,
            frame_length=patch_window_length_samples,
            frame_step=patch_hop_length_samples,
            axis=0)
        """
        return log_mel_spectrogram

    def _log_mel_spectrogram_3(self, i):
        mel_spectrogram = self._get_melspectrogram_db(i)
        spec_image = self._spec_to_image(mel_spectrogram)

        return spec_image

    def _get_melspectrogram_db(self, i):
        input_filename = self.df.loc[i, "File Name"]
        waveform, sr = self._read(input_filename)

        spec=librosa.feature.melspectrogram(
            y=waveform, 
            sr=sr, 
            n_fft=self.config.fft_length,
            win_length=self.config.window_length_samples,
            hop_length=self.config.hop_length_samples,
            n_mels=128,
            fmin=20,
            fmax=8300)
        spec_db=librosa.power_to_db(spec, top_db=80)

        return spec_db

    def _spec_to_image(self, spec):
        mean = spec.mean()
        std = spec.std()
        spec_norm = (spec - mean) / (std + self.config.epsilon)
        spec_min, spec_max = spec_norm.min(), spec_norm.max()
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
        spec_scaled = np.expand_dims(spec_scaled, axis=-1)
        spec_scaled = spec_scaled.astype(np.uint8)

        return spec_scaled
    

    def _read_mfcc(self, i):
        input_filename = self.df.loc[i, "File Name"]
        waveform, sr = self._read(input_filename)

        # mfcc = self._mfcc_fbank(waveform, sr)
        mfcc = librosa.feature.mfcc(
            y=waveform, 
            sr=sr, 
            # n_mfcc=40, 
            n_mfcc=80, 
            n_fft = 1024,
            hop_length= int(sr * 0.01),
            win_length=1024,
            power=2,
            center=False,
            window='hann',
            n_mels=80,
            htk=True
        )
        mfcc = np.expand_dims(mfcc, axis=-1)
        # mfcc = tf.image.grayscale_to_rgb(mfcc)
        mfcc = np.concatenate((mfcc,mfcc,mfcc), axis=-1)

        return mfcc


    def tranfrom_data(self, i):
        MFCC_NUM = 20
        MFCC_MAX_LEN = 2000
        feature_dim_1 = MFCC_NUM
        # Second dimension of the feature is dim2
        feature_dim_2 = MFCC_MAX_LEN
        channel = 1
        input_filename = self.df.loc[i, "File Name"]
        waveform, sr = self._read(input_filename)
        result = []
        waveform = waveform[::3]
        mfcc = self.wav2mfcc(waveform)
        result.append(mfcc)
        result = np.array(result)

        # return result.reshape(result.shape[0], feature_dim_1, feature_dim_2, channel)
        return result.reshape(feature_dim_1, feature_dim_2, channel)



    def wav2mfcc(self, wave, max_len=2000, sr=44100):
        mfcc = librosa.feature.mfcc(y=wave, n_mfcc=20, sr=sr)
        if (max_len > mfcc.shape[1]):
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        
        return mfcc

    def _read(self, input_filename):
        waveform, sr = librosa.load(input_filename, sr=self.config.sample_rate, dtype=np.float32)

        dst_length = self.config.sample_rate*self.config.audio_duration
        waveform = self._trim_audio(waveform, dst_length)

        if self.mode == "train":
            waveform = self.augment(waveform, sample_rate=sr)
            # pass

        return waveform, sr

    def _trim_audio(self, waveform, dst_length):
        if waveform.shape[0] < dst_length:
            waveform = librosa.util.fix_length(waveform, size=dst_length)
            
        elif waveform.shape[0] > dst_length:
            start_idx = np.random.randint(len(waveform) - dst_length, size=1, dtype=int).item()
            waveform = waveform[start_idx: start_idx+dst_length]
        
        return waveform


    def _mfcc_fbank(self, signal: np.array, sample_rate: int):  # 1D signal array.
        # Returns MFCC with shape (num_frames, n_filters, 3).
        filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=self.config.num_fbanks)
        frames_features = self._normalize_frames(filter_banks)

        return np.array(frames_features, dtype=np.float32)


    def _normalize_frames(self, m, epsilon=1e-12):

        return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]