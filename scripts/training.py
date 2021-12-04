import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.callbacks import ModelCheckpoint

sys.path.append("./src")

import argparse
from azureml.core import Run

import models
from data_v2 import (ImgGen, get_center_box, get_random_subbox,
                     random_subbox_and_augment)
from helpers import CheckpointCallback

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-prefix", type=str, default="experiment")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.00001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--data-path", type=str, dest="data_path", help="data folder mounting point"
    )

    args = parser.parse_args()

    data_path = args.data_path

    print("============================================")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    os.makedirs("./outputs", exist_ok=True)

    train_data_all = pd.read_csv(f"{data_path}/train.csv")
    print("Shape of train_data_all:", train_data_all.shape)

    sirnas = [
        "sirna_706",
        "sirna_1046",
        "sirna_1045",
        "sirna_586",
        "sirna_747",
        "sirna_705",
        "sirna_1044",
        "sirna_1043",
        "sirna_1042",
        "sirna_1041",
    ]

    train_data_all = train_data_all.loc[train_data_all.sirna.isin(sirnas), :]

    train_data_all.loc[:, ["site"]] = "1"
    train_data_second_site = train_data_all.copy()
    train_data_second_site.loc[:, ["site"]] = "2"
    train_data_all = pd.concat([train_data_all, train_data_second_site], axis=0)

    sirna_label_encoder_all = LabelEncoder().fit(train_data_all.sirna)
    print(f"Sirna classes in train_data_all {len(sirna_label_encoder_all.classes_)}")

    train_data_strat = train_data_all
    # train_data_strat=train_data_all.groupby('sirna',group_keys=False).apply(lambda x: x.sample(frac=0.10).reset_index(drop=True))

    train_data_sample_1 = train_data_strat.sample(frac=1, random_state=42).reset_index(
        drop=True
    )
    print("Shape of train_data_sample_1:", train_data_sample_1.shape)
    sirna_label_encoder_sample_1 = LabelEncoder().fit(train_data_sample_1.sirna)
    print(
        f"Sirna classes in train_data_sample_1 {len(sirna_label_encoder_sample_1.classes_)}"
    )

    # save encoders to output
    joblib.dump(sirna_label_encoder_all, f"./outputs/sirna_label_encoder_all.pkl")
    joblib.dump(
        sirna_label_encoder_sample_1, f"./outputs/sirna_label_encoder_sample_1.pkl"
    )

    run = Run.get_context()

    model = models.create_inception(
        learning_rate=args.learning_rate,
        nr_classes=len(sirna_label_encoder_sample_1.classes_),
    )

    try:
        tf.keras.utils.plot_model(model, to_file="./outputs/model.png", show_shapes=True)
    except ImportError as e:
        print(f"Cannot plot model: {e}")

    test_size = 0.3
    batch_size = args.batch_size

    train, val = train_test_split(
        train_data_sample_1, test_size=test_size, random_state=42
    )

    print(f"Training set size {len(train)}")
    print(train.sirna.value_counts())
    print(f"Validation set size {len(val)}")
    print(val.sirna.value_counts())

    run.log("Batch Size", batch_size)
    run.log("Test fraction", test_size)
    run.log("Training samples", len(train))
    run.log("Learning rate", args.learning_rate)

    train_gen = ImgGen(
        train,
        batch_size=batch_size,
        preprocess=get_center_box,
        shuffle=True,
        label_encoder=sirna_label_encoder_sample_1,
        path=f"{data_path}/train/",
        cache=True,
    )
    val_gen = ImgGen(
        val,
        batch_size=batch_size,
        preprocess=random_subbox_and_augment,
        shuffle=True,
        label_encoder=sirna_label_encoder_sample_1,
        path=f"{data_path}/train/",
        cache=True,
    )

    print(f"Training set batched size {len(train_gen)}")
    print(f"Validation set batched size {len(val_gen)}")

    filepath = f"./outputs/ModelCheckpoint.h5"

    tqdm_callback = tfa.callbacks.TQDMProgressBar()
    aml_callback = CheckpointCallback(run)

    callbacks = [
        ModelCheckpoint(
            filepath,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
            save_freq="epoch",
        ),
        # tqdm_callback,
        aml_callback,
    ]

    try:
        history = model.fit(
            train_gen,
            steps_per_epoch=len(train) // batch_size,
            epochs=args.epochs,
            verbose=1,
            validation_data=val_gen,
            validation_steps=len(val) // batch_size,
            callbacks=callbacks,
        )
    except KeyboardInterrupt:
        print("Training ended early")

        history = model.history

    # plot history to png
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(f"./outputs/learning_curve_history.png")
    # plt.show()
