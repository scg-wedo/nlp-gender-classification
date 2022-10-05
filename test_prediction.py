import os
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from generator import TFDataset
from config import Config

class BatchInference():
    """Keras trainer class.
    
    This class will perform
        - Generate model
        - Generate train, valid generator
        - Compile model
            with specific optimizer, loss, metrics
        - Fit model
    
    # TODO: Save config
    """
    def __init__(self, config):
        self.config = config
        self.config.hyperparameters["batch_size"] = 1
        self.model = self._generate_model()
        self.test_generator, self.test_df = self._generate_generator()
        self.raw_prediction = self.prediction()
        self.result_df = self.generate_result_df()

    def _generate_model(self):
        """Generate Keras model."""
        json_path = os.path.join(self.config.model["model_path"], self.config.model["model_name"]+".json")
        w_path = json_path.replace(".json", ".h5")

        print("Generating model: ", json_path)
        with open(json_path) as f:
            model = tf.keras.models.model_from_json(f.read())
        
        if self.config.model["load_weights"]:
            print("Loading weights")
            model.load_weights(w_path)
        
        # Train only topn layer.
        if self.config.model["train_topn_layer"]: 
            print("Train only top {} layers".format(self.config.model["train_topn_layer"]))
 
            # Unfreeze only top n layers
            for i, layer in enumerate(reversed(model.layers)):
                if i < self.config.model["train_topn_layer"]:
                    print("Unfreeze: ", layer.name)
                    layer.trainable = True
                else:
                    layer.trainable = False

        print(model.summary())

        return model

    def _generate_generator(self):
        """Generate Keras generator.
        
        This function will return train_generator and valid_generator
        """
        print("Generating Tensorflow dataset...")
        test_tf_dataset = TFDataset(
                                    config=self.config,
                                    mode="test"
                                    )

        test_generator = test_tf_dataset.get_dataset()

        test_df = test_tf_dataset.get_df()

        return test_generator, test_df

    def prediction(self):
        print("Your data is being predicted...")
        result = self.model.predict(self.test_generator, steps=self.test_df.shape[0])
        
        return result

    def generate_result_df(self):
        prediction = []
        pred_score = {
            c: [] for c in self.config.model["class_name"]
        }
        for sample_pred in self.raw_prediction:
            pred = np.argmax(sample_pred)
            class_pred = self.config.model["class_name"][pred]
            prediction.append(class_pred)
            for c, c_pred in zip(self.config.model["class_name"], sample_pred):
                pred_score[c].append(c_pred)
        
        result_df = self.test_df[["File Name"]+self.config.model["class_name"]]
        result_df["Prediction"] = prediction
        for c, scores in pred_score.items():
            result_df["{} Score".format(c)] = scores

        return result_df

    def get_raw_prediction(self):
        return self.raw_prediction
    
    def get_result_df(self):
        print(self.result_df.head())
        return self.result_df

if __name__ == "__main__":
    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_name)
    batch_infer = BatchInference(config)

    df = batch_infer.get_result_df()
    df.to_csv("result_valid_ser_cv11.csv")
