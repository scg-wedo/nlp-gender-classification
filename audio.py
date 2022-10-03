import os
import numpy as np
import librosa
from python_speech_features import fbank
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
                Compose([
                    TimeStretch(p=0.5),
                    PitchShift(p=0.5)
                ])
            ])

    def get_item(self, i):
        i = i.numpy()
        # x = self._read_mfcc(i)
        x = self._log_mel_spectrogram(i)
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

    def _read_mfcc(self, i):
        input_filename = self.df.loc[i, "File Name"]
        waveform, sr = self._read(input_filename)

        mfcc = self._mfcc_fbank(waveform, sr)
        mfcc = np.expand_dims(mfcc, axis=-1)

        return mfcc

    def _read(self, input_filename):
        waveform, sr = librosa.load(input_filename, sr=self.config.sample_rate, mono=True, dtype=np.float32)
        dst_length = self.config.sample_rate*self.config.audio_duration
        
        if waveform.shape[0] != dst_length:
            waveform = librosa.util.fix_length(waveform, size=dst_length)

        if self.mode == "train":
            waveform = self.augment(waveform, sample_rate=sr)

        return waveform, sr

    def _mfcc_fbank(self, signal: np.array, sample_rate: int):  # 1D signal array.
        # Returns MFCC with shape (num_frames, n_filters, 3).
        filter_banks, energies = fbank(signal, samplerate=sample_rate, nfilt=self.config.num_fbanks)
        frames_features = self._normalize_frames(filter_banks)

        return np.array(frames_features, dtype=np.float32)


    def _normalize_frames(self, m, epsilon=1e-12):

        return [(v - np.mean(v)) / max(np.std(v), epsilon) for v in m]