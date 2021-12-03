import logging
import random
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageFilter
from tensorflow.keras.utils import Sequence

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)

logger = logging.getLogger("imgloader")
# logger.setLevel(logging.INFO)


def get_image_zip(
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


def get_image(
    experiment, plate, well, site=1, channels=(1, 2, 3, 4, 5, 6), path="../data/train/"
):
    """
    Loads image from folder or zip file using given parameters
    """

    def load_image(channel):
        try:
            _path = f"{path}{experiment}/Plate{plate}/{well}_s{str(site)}_w{str(channel)}.png"
            img = Image.open(_path)
        except FileNotFoundError as err:
            logger.error(f"Error loading input - {err}")
            logger.info("Will default to other site")
            # hack mis aitab kahe puuduva pildi puhul
            # pm kui puudub pilt siis proovib lihtsalt teist saiti võtta
            if site == 2:
                _path = f"{path}{experiment}/Plate{plate}/{well}_s1_w{str(channel)}.png"
                img = Image.open(_path)

            else:
                _path = f"{path}{experiment}/Plate{plate}/{well}_s2_w{str(channel)}.png"
                img = Image.open(_path)

        return img

    img_channels = [load_image(c) for c in channels]

    return img_channels


def augment(image, seed=None):

    random.seed(seed)

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


def get_random_subbox(image, shape_w_h=(224, 224), seed=42):

    random.seed(seed)

    w = image.width
    h = image.height

    logger.debug(f"Image w, h: {w},{h}")

    random_w = random.randint(0, w - shape_w_h[0])
    random_h = random.randint(0, h - shape_w_h[1])

    cropbox = (random_w, random_h, random_w + shape_w_h[0], random_h + shape_w_h[1])

    logger.debug(f"Random cropbox: {cropbox}")

    cropped = image.crop(cropbox)

    logger.debug(f"Cropped image w, h: {cropped.width}, {cropped.height}")

    return cropped


def random_subbox_and_augment(image, seed=None):
    """
    Randomly crops image to 224x224 and augments it
    """
    cropped = get_random_subbox(image, seed=seed)
    return augment(cropped, seed=seed)


def preprocess_and_scale(img_channels, preprocess, seed=42):

    logger.info("Preprocessing images")
    img_prepro = [preprocess(img, seed=seed) for img in img_channels]

    logger.info("Scaling channels")
    x_s1_c1 = np.array(img_prepro[0]) / 255
    x_s1_c2 = np.array(img_prepro[1]) / 255
    x_s1_c3 = np.array(img_prepro[2]) / 255

    x_s1_c4 = np.array(img_prepro[3]) / 255
    x_s1_c5 = np.array(img_prepro[4]) / 255
    x_s1_c6 = np.array(img_prepro[5]) / 255

    logger.info("Scaling channels complete")

    c1_c3_img = np.array([x_s1_c1, x_s1_c2, x_s1_c3])
    c4_c6_img = np.array([x_s1_c4, x_s1_c5, x_s1_c6])

    c1_c6_img_array = np.append(c1_c3_img, c4_c6_img, axis=0)
    logger.info(f"Image array shape: {c1_c6_img_array.shape}")

    logger.info("Loading channels compete")

    return c1_c6_img_array


class ImgGen(Sequence):
    def __init__(
        self,
        label_data,
        label_encoder,
        path,
        batch_size=32,
        cache=False,
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
        self.cache = cache

        self._batches = dict()

    def __len__(self):

        logger.info(f"Batch size {self.batch_size}")
        logger.info(f"Label data {len(self.label_data)}")
        batches = int(np.ceil(len(self.label_data) / float(self.batch_size)))
        logger.info(f"Nr of batches {batches}")

        return batches

    def __getitem__(self, i):

        logger.info(f"Getting batch {i}")
        batch_x = self.label_data.loc[
            i * self.batch_size : (i + 1) * self.batch_size - 1,
            ("experiment", "plate", "well", "site"),
        ]

        # logger.debug(batch_x)

        batch_y = self.label_data.loc[
            i * self.batch_size : (i + 1) * self.batch_size - 1, ("sirna")
        ]

        # logger.debug(batch_y)

        x = list()

        if self.cache and i in self._batches:
            logger.info("Getting images from memory cache")
            x_images = self._batches[i]
        else:
            logger.info("Loading images from disk")
            x_images = list()
            for e, p, w, s in batch_x.itertuples(index=False):
                logger.info(f"Loading image for {e}, {p}, {w}, {s}")
                img_channels = get_image(e, p, w, site=s, path=self.path)

                logger.info(f"Got channels: {len(img_channels)}")

                logger.debug(f"Channels: {img_channels}")

                x_images.append(img_channels)

            if self.cache:
                self._batches[i] = x_images

        logger.info("Done loading/getting images")

        logger.info("Starting preprocessing")

        # get seed for consistent random augmentation
        seed = random.randint(0, 2 ** 32 - 1)

        x = [preprocess_and_scale(img, self.preprocess, seed=seed) for img in x_images]

        logger.info("Preprocessing complete")

        logger.info("Finished batch loading")

        y = self.label_encoder.transform(batch_y)

        logger.info(f"Labels size: {len(y)}")

        X = np.array(x)

        logger.info(f"X sixe: {len(X)}, x size {len(x)}, X shape: {X.shape}")

        batch = X, y

        return batch
