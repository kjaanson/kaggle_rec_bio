import random

from retry import retry

import numpy as np

from PIL import Image, ImageFilter

from tensorflow.keras.utils import Sequence


@retry(tries=3)
def get_input(
    experiment, plate, well, site, channel, train=True, path="./input/recbio"
):

    if train == True:
        base_path = f"{path}/train"
    else:
        base_path = f"{path}/test"

    try:
        path = f"{base_path}/{experiment}/Plate{plate}/{well}_s{str(site)}_w{str(channel)}.png"
        img = Image.open(path)
    except FileNotFoundError as err:
        print(f"Error loading input - {err}")
        print("Will default to other site")

        # hack mis aitab kahe puuduva pildi puhul
        # pm kui puudub pilt siis proovib lihtsalt teist saiti võtta
        if site == 2:
            path = (
                f"{base_path}/{experiment}/Plate{plate}/{well}_s1_w{str(channel)}.png"
            )
            img = Image.open(path)
        else:
            path = (
                f"{base_path}/{experiment}/Plate{plate}/{well}_s2_w{str(channel)}.png"
            )
            img = Image.open(path)

    return img


def augment(image):

    random_transform = random.randint(-1, 4)
    if random_transform == 0:
        image = image.rotate(random.randint(-5, 5))
    if random_transform == 1:
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
    if random_transform == 2:
        image = image.filter(ImageFilter.RankFilter(size=3, rank=1))
    if random_transform == 3:
        image = image.filter(ImageFilter.MedianFilter(size=3))
    if random_transform == 4:
        image = image.filter(ImageFilter.MaxFilter(size=3))
    return image


def get_center_box(image, shape_w_h=(224, 224)):

    cropbox = (144, 144, 144 + shape_w_h[0], 144 + shape_w_h[1])

    return image.crop(cropbox)


def get_random_subbox(image, shape_w_h=(224, 224)):

    w = image.width
    h = image.height

    print(f"Image w, h: {w},{h}")

    random_w = random.randint(0, w - shape_w_h[0])
    random_h = random.randint(0, h - shape_w_h[1])

    cropbox = (random_w, random_h, random_w + shape_w_h[0], random_h + shape_w_h[1])

    print(f"Random cropbox: {cropbox}")

    cropped = image.crop(cropbox)

    print(f"Cropped image w, h: {cropped.width}, {cropped.height}")

    return cropped


class ImgGen(Sequence):
    def __init__(
        self,
        label_data,
        label_encoder,
        path,
        batch_size=32,
        preprocess=(lambda x: x),
        shuffle=False,
    ):

        if shuffle:
            self.label_data = label_data.sample(frac=1).reset_index(drop=True)
        else:
            self.label_data = label_data

        self.batch_size = batch_size
        self.preprocess = preprocess
        self.path = path
        self.label_encoder = label_encoder

        self._batches = dict()

    def __len__(self):
        return int(np.ceil(len(self.label_data)) / float(self.batch_size))

    def __getitem__(self, i):

        if i in self._batches:
            return self._batches[i]

        batch_x = self.label_data.loc[
            i * self.batch_size : (i + 1) * self.batch_size,
            ("experiment", "plate", "well"),
        ]
        batch_y = self.label_data.loc[
            i * self.batch_size : (i + 1) * self.batch_size, ("sirna")
        ]

        x_s1_c1 = [
            np.array(
                self.preprocess(get_input(e, p, w, site=1, channel=1, path=self.path))
            )
            / 255
            for e, p, w in batch_x.values.tolist()
        ]
        x_s1_c2 = [
            np.array(
                self.preprocess(get_input(e, p, w, site=1, channel=2, path=self.path))
            )
            / 255
            for e, p, w in batch_x.values.tolist()
        ]
        x_s1_c3 = [
            np.array(
                self.preprocess(get_input(e, p, w, site=1, channel=3, path=self.path))
            )
            / 255
            for e, p, w in batch_x.values.tolist()
        ]

        x_s1_c4 = [
            np.array(
                self.preprocess(get_input(e, p, w, site=1, channel=4, path=self.path))
            )
            / 255
            for e, p, w in batch_x.values.tolist()
        ]
        x_s1_c5 = [
            np.array(
                self.preprocess(get_input(e, p, w, site=1, channel=5, path=self.path))
            )
            / 255
            for e, p, w in batch_x.values.tolist()
        ]
        x_s1_c6 = [
            np.array(
                self.preprocess(get_input(e, p, w, site=1, channel=6, path=self.path))
            )
            / 255
            for e, p, w in batch_x.values.tolist()
        ]

        x1 = np.array([x_s1_c1, x_s1_c2, x_s1_c3]).transpose((1, 2, 3, 0))
        x2 = np.array([x_s1_c4, x_s1_c5, x_s1_c6]).transpose((1, 2, 3, 0))

        y = self.label_encoder.transform(batch_y)

        batch = [np.array(x1), np.array(x2)], y

        self._batches[i] = batch

        return batch
