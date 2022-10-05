"""Implementation of Keras custom callback.
This callback will evaluate model with the testset. Saved prediction score of each in CSV.
"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import os
from config import Config

class TestPredictionCallback(keras.callbacks.Callback):
    def __init__(self, config, TFDataset):
        self.config = config
        # self.config.hyperparameters["batch_size"] = 1

        valid_tf_dataset = TFDataset(
                                config=self.config,
                                mode="valid"
                                )

        self.valid_df = valid_tf_dataset.get_df()
        self.valid_generator = valid_tf_dataset.get_dataset()

        self.save_root = os.path.join(self.config.save_root, "report")

        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)
        
    def _generate_report(self, prediction_score, generator_df):
        prediction_score = prediction_score.tolist()
        report = {
            "File Name": generator_df["File Name"],
        }
        for c in self.config.model["class_name"]:
            report["{} Prediction Score".format(c)] = [] # Prediction Score
            report[c] = generator_df[c] # Ground truth
        
        for pred in prediction_score:
            for class_pred, c in zip(pred, self.config.model["class_name"]):
                report["{} Prediction Score".format(c)].append(class_pred)
        
        return pd.DataFrame(report)

    def _get_acc(self, prediction_score, generator_df):
        gt_df = generator_df[self.config.model["class_name"]]
        gt = np.array(gt_df.values.tolist())
        gt = np.argmax(gt, axis=1)

        pred = np.argmax(prediction_score, axis=1)

        equal = np.equal(gt, pred).astype(np.float32)

        acc = np.mean(equal).item() * 100

        return acc


    def generate_prediction_report(self, epoch, generator, dataset_df, dataset):
        steps = (self.valid_df.shape[0]//self.config.hyperparameters["batch_size"])+1
        prediction_score = self.model.predict(generator, steps=steps)
        # Generate prediction report
        report = self._generate_report(prediction_score, dataset_df)
        # Save report
        save_path = os.path.join(self.save_root, "{}_ep_{}_acc_{}.csv".format(dataset, epoch, self._get_acc(prediction_score, dataset_df)))
        print("Save to {}".format(save_path))
        report.to_csv(save_path, index=False)

    def on_epoch_end(self, epoch, logs=None):
        print("\nEvaluating model")
        if epoch % 2 == 0:
            self.generate_prediction_report(epoch, self.valid_generator, self.valid_df, "valid")