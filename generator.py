import os
import pandas as pd
import tensorflow as tf

from audio import Audio

class TFDataset():
    """tf dataset implementation.
    Attributes
    ----------
    config : object
        Config for training
    model : str
        Currently support train, valid, inference

    Methods
    -------
    get_dataset
        Return tf dataset for training or inference
    """
    def __init__(self, config, mode="inference"):
        self.config = config
        self.mode = mode

        self.df = self._read_df()
        self.audio = Audio(self.config, self.df, mode)
        self.dataset = self._generate_tfdataset()

    def get_dataset(self):
        return self.dataset

    def get_df(self):
        return self.df

    def _read_df(self):
        dfs = []
        for data_obj in self.config.dataset:
            if self.mode == "train":
                df_path = data_obj["train_df"]
            elif self.mode == "valid":
                df_path = data_obj["valid_df"]
            else:
                df_path = data_obj["test_df"]
            
            tmp_df = self._read_preproc_df(df_path, data_obj["ds_root"])
            dfs.append(tmp_df)
        
        return pd.concat(dfs, ignore_index=True)
        
    def _read_preproc_df(self, df_path, ds_root):
        df = pd.read_csv(df_path)
        df = df.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)
        df["File Name"] = df["File Name"].apply(lambda x: os.path.join(ds_root, x))

        return df

    def _generate_tfdataset(self):
        dataset_size = self.df.shape[0]
        dataset_index = list(range(dataset_size))
        dataset = tf.data.Dataset.from_generator(lambda: dataset_index, tf.uint8)
        dataset = dataset.shuffle(
                                buffer_size=dataset_size, 
                                seed=self.config.seed,  
                                reshuffle_each_iteration=True
                                )
        
        dataset = dataset.map(lambda i: tf.py_function(func=self.audio.get_item, 
                                                    inp=[i], 
                                                    Tout=[tf.float32, tf.float32]
                                                    ), 
                            num_parallel_calls=self.config.dataloader_num_parallel)
        
        dataset = dataset.batch(self.config.hyperparameters["batch_size"]).map(self._fixup_shape)
        dataset = dataset.prefetch(self.config.dataloader_num_parallel)
        if self.mode != "test":
            dataset = dataset.repeat()

        return dataset

    def _fixup_shape(self, x, y):
        x.set_shape([None, None, None, 1]) # n, h, w, c
        y.set_shape([None, len(self.config.model["class_name"])]) # n, nb_classes

        return x, y