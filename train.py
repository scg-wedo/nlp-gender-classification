
import os
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D

from generator import TFDataset
from config import Config
from custom_metrics import category_acc
from test_prediction_callback import TestPredictionCallback

class Trainer():
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
        self.model = self._generate_model()
        self.train_generator, self.valid_generator, self.train_df, self.valid_df = self._generate_generator()
        self.class_weight = self._calculate_class_weights()

        os.makedirs(self.config.save_root, exist_ok=True)

    def _generate_model(self):
        """Generate Keras model."""
        # json_path = os.path.join(self.config.model["model_path"], self.config.model["model_name"]+".json")
        # w_path = json_path.replace(".json", ".h5")

        # print("Generating model: ", json_path)
        # with open(json_path) as f:
        #     model = tf.keras.models.model_from_json(f.read())
        
        # if self.config.model["load_weights"]:
        #     print("Loading weights")
        #     model.load_weights(w_path)
        
        # # Train only topn layer.
        # if self.config.model["train_topn_layer"]: 
        #     print("Train only top {} layers".format(self.config.model["train_topn_layer"]))
 
        #     # Unfreeze only top n layers
        #     for i, layer in enumerate(reversed(model.layers)):
        #         if i < self.config.model["train_topn_layer"]:
        #             print("Unfreeze: ", layer.name)
        #             layer.trainable = True
        #         else:
        #             layer.trainable = False

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(20, 2000, 1)))
        model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
        model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(2, activation='softmax'))

        print(model.summary())

        return model

    def _generate_generator(self):
        """Generate Keras generator.
        
        This function will return train_generator and valid_generator
        """
        print("Generating Tensorflow dataset...")
        train_tf_dataset = TFDataset(
                                    config=self.config,
                                    mode="train"
                                    )
        valid_tf_dataset = TFDataset(
                                    config=self.config,
                                    mode="valid"
                                    )

        train_generator = train_tf_dataset.get_dataset()
        valid_generator = valid_tf_dataset.get_dataset()

        train_df = train_tf_dataset.get_df()
        valid_df = valid_tf_dataset.get_df()

        return train_generator, valid_generator, train_df, valid_df

    def _calculate_class_weights(self):
        label = []
        for _, row in self.train_df.iterrows():
            for l in self.config.model["class_name"]:
                if row[l]:
                    label.append(l)
        weights = compute_class_weight(
                                        class_weight="balanced", 
                                        classes=self.config.model["class_name"],
                                        y=label
                                     )
        class_weight = {
            c: w for c,w in zip(range(len(self.config.model["class_name"])), weights)
        }

        print("Class weight: ", class_weight)

        return class_weight
    
    def _define_callbacks(self):
        lr_scheduler = ReduceLROnPlateau(
                                        monitor='val_loss',
                                        factor=0.3,
                                        patience=2,
                                        min_lr=1e-7,
                                    )
        w_save_path = os.path.join(self.config.save_root, "model_w")
        board_save_path = os.path.join(self.config.save_root, "logs")
        if not os.path.exists(w_save_path):
            os.makedirs(w_save_path)

        model_name = "{epoch:02d}-{val_loss:.2f}.h5"
        file_path = os.path.join(w_save_path, model_name)

        checkpoint = ModelCheckpoint(filepath=file_path, save_weights_only=True, monitor="val_loss")
        board = TensorBoard(log_dir=board_save_path)

        test_prediction = TestPredictionCallback(self.config, TFDataset)

        return [lr_scheduler, checkpoint, board, test_prediction]
        # return [lr_scheduler, checkpoint, board]

    def _define_matrics(self):

        return []

    def get_hyperparameters(self):
        # print("Class weights: ", self.class_weights)
        print("Model config: ", self.config.model)
        print("Hyperparameter config: ", self.config.hyperparameters)

    def compile_train(self):
        print("list of parameters: ", self.get_hyperparameters())
        self.model.compile(
            optimizer=self.config.hyperparameters["optimizer"],
            loss=self.config.hyperparameters["loss"],
            metrics=self.config.hyperparameters["metrics"]+self._define_matrics(),
        )
        print("Model compiled")
        history = self.model.fit(
            x=self.train_generator,
            validation_data=self.valid_generator,
            epochs=self.config.hyperparameters["n_epoch"],
            callbacks=self._define_callbacks(),
            class_weight=self.class_weight,
            steps_per_epoch=(self.train_df.shape[0]//self.config.hyperparameters["batch_size"])+1,
            validation_steps=(self.valid_df.shape[0]//self.config.hyperparameters["batch_size"])+1,
            use_multiprocessing = True,
            workers=self.config.dataloader_num_parallel
        )


if __name__ == '__main__':

    config = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_name)
    trainer = Trainer(config)
    trainer.compile_train()