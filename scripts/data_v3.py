
import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.utils import Sequence


def get_image(
    experiment,
    plate,
    well,
    site=1,
    channels=(1, 2, 3, 4, 5, 6),
    path="./data/train.zip",
):
    """
    Loads image from folder or zip file using given parameters
    """

    archive = ZipFile(path)

    def load_image(channel):
        try:
            path = f"{experiment}/Plate{plate}/{well}_s{str(site)}_w{str(channel)}.png"
            img = Image.open(archive.open(path))
        except FileNotFoundError as err:
            print(f"Error loading input - {err}")
            print("Will default to other site")
            # hack mis aitab kahe puuduva pildi puhul
            # pm kui puudub pilt siis proovib lihtsalt teist saiti võtta
            if site == 2:
                path = f"{experiment}/Plate{plate}/{well}_s1_w{str(channel)}.png"
                img = Image.open(archive.open(path))

            else:
                path = f"{experiment}/Plate{plate}/{well}_s2_w{str(channel)}.png"
                img = Image.open(archive.open(path))

        return img

    img_channels = [load_image(c) for c in channels]

    return img_channels


def parse_path_for_metadata(path):
    """
    Parse a path to extract the metadata from it.
    Example path format: RPE-01/Plate1/H18_s2_w2.png where RPE-01 is the experiment name, Plate1 is the plate name, H18_s2_w2 is the with location and img channel
    Parse experiment, plate, well name and well position.
    """
    path_split = path.split("/")
    experiment = path_split[0]

    # keep only plate number as plate designation
    plate = int(path_split[1].strip("Plate"))

    well_name = path_split[2].split("_")[0]
    well_position = path_split[2].split("_")[1]
    # channel = path_split[2].split('_')[2].split('.')[0]
    return experiment, plate, well_name, well_position


class ImgGen(Sequence):
    """
    Loads images from a given directory, gets label data from metadata csv
    """

    def __init__(
        self,
        img_dir_or_list,
        metadata_df,
        batch_size,
        preprocess_fn=(lambda x: x),
        shuffle=False,
    ):

        # if img_dir_or_list is a string, assume it's a directory, load list of files recursively
        if isinstance(img_dir_or_list, str):
            self.img_file_list = self.get_img_list(img_dir_or_list)
        elif isinstance(img_dir_or_list, list):
            self.img_file_list = img_dir_or_list

        # check that experiment, plate and well_name columns are in metadata_df, throw error if not
        if not all(
            [col in metadata_df.columns for col in ["experiment", "plate", "well_name"]]
        ):
            raise ValueError(
                "metadata_df must have columns experiment, plate and well_name"
            )

        self.metadata = metadata_df

        self.file_path_metadata = pd.DataFrame(
            self.img_file_list, columns=["file_path"]
        )
        self.file_path_metadata[
            ["experiment", "plate", "well_name", "well_position"]
        ] = list(self.file_path_metadata.file_path.apply(parse_path_for_metadata))

        self.file_path_metadata = self.file_path_metadata.merge(
            self.metadata, on=["experiment", "plate", "well_name"], how="left"
        )

        # throw error if file path metadata has NaN values
        if self.file_path_metadata.isnull().values.any():
            raise ValueError(
                "file_path_metadata must have no NaN values, maybe there is missing metadata?"
            )

        # TODO, batching and image loading in parallel

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.preprocess_fn = preprocess_fn

        self.img_list = []
        self.label_list = []

    def get_img_list(self, img_dir):
        """
        Recursively load all images from a directory
        """
        img_file_list = []
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if file.endswith(".png"):
                    img_file_list.append(os.path.join(root, file))
        return img_file_list

    def __len__(self):
        """
        :return: Number of batches per epoch
        """
        return int(np.floor(len(self.img_list) / self.batch_size))

    def __getitem__(self, index):
        """
        :param index: Index of batch
        :return: Batch of images and labels
        """
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Get list of images and labels
        img_list_batch = [self.img_list[k] for k in indexes]
        # label_list_batch = [self.label_list[
