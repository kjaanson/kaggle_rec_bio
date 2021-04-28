
import random

from retry import retry

import numpy as np

from PIL import Image, ImageFilter

from tensorflow.keras.utils import Sequence
from joblib import Parallel, delayed

import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)

logger = logging.getLogger("imgloader")
logger.setLevel(logging.INFO)

def get_input(experiment, plate, well, site=1, channels=(1,2,3,4,5,6), train=True, path='./input/recbio'):
    
    if train==True:
        base_path = f'{path}/train'
    else:
        base_path = f'{path}/test'
    
    def load_image(channel):

        try:
            path = f"{base_path}/{experiment}/Plate{plate}/{well}_s{str(site)}_w{str(channel)}.png"
            
            logger.info(f"Loading image {path}")
            img = Image.open(path)
            
        except FileNotFoundError as err:
            print(f"Error loading input - {err}")
            print("Will default to other site")
            
            # hack mis aitab kahe puuduva pildi puhul
            # pm kui puudub pilt siis proovib lihtsalt teist saiti võtta
            if site==2:
                path = f"{base_path}/{experiment}/Plate{plate}/{well}_s1_w{str(channel)}.png"
                img = Image.open(path)
            else:
                path = f"{base_path}/{experiment}/Plate{plate}/{well}_s2_w{str(channel)}.png"
                img = Image.open(path)

        return img

    #img_channels = Parallel(n_jobs=6)(delayed(load_image)(c) for c in channels)
    
    img_channels = [load_image(c) for c in channels]


    return img_channels


def augment(image):

    random_transform = random.randint(-1,4)
    if random_transform==0:
        image = image.rotate(random.randint(-5,5))
    if random_transform==1:
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
    if random_transform==2:
        image = image.filter(ImageFilter.RankFilter(size=3, rank=1))
    if random_transform==3:
        image = image.filter(ImageFilter.MedianFilter(size=3))
    if random_transform==4:
        image = image.filter(ImageFilter.MaxFilter(size=3))
    return image



def get_center_box(image, shape_w_h=(224,224)):
    
    cropbox = (144, 144, 144 + shape_w_h[0], 144 + shape_w_h[1])
    
    return image.crop(cropbox)


def get_random_subbox(image, shape_w_h=(224,224)):
    
    w = image.width
    h = image.height
    
    print(f"Image w, h: {w},{h}")
    
    random_w = random.randint(0, w - shape_w_h[0])
    random_h = random.randint(0, h - shape_w_h[1])
    
    cropbox = (random_w, random_h, random_w + shape_w_h[0], random_h + shape_w_h[1])
    
    print(f"Random cropbox: {cropbox}")
    
    cropped = image.crop( cropbox )
    
    print(f"Cropped image w, h: {cropped.width}, {cropped.height}")
    
    return cropped




def get_channels(e, p, w, site, path, preprocess):

    logger.info(f"Experiment {e}, plate {p}, well {w}")
    img_chs = get_input(e, p, w, site=site, path=path)

    logger.info(f"Got channels: {len(img_chs)}")

    logger.info("Starting preprocessing")
    img_prepro = [preprocess(img) for img in img_chs]
    logger.info("Preprocessing complete")

    logger.info("Scaling channels")
    x_s1_c1 = np.array(img_prepro[0])/255
    x_s1_c2 = np.array(img_prepro[1])/255
    x_s1_c3 = np.array(img_prepro[2])/255

    x_s1_c4 = np.array(img_prepro[3])/255
    x_s1_c5 = np.array(img_prepro[4])/255
    x_s1_c6 = np.array(img_prepro[5])/255

    logger.info("Scaling channels complete")

    c1_c3_img = np.array([x_s1_c1,x_s1_c2,x_s1_c3])

    c4_c6_img = np.array([x_s1_c4,x_s1_c5,x_s1_c6])

    c1_c6_img_array = np.append(c1_c3_img, c4_c6_img, axis=0)
    logger.info(f"Image array shape: {c1_c6_img_array.shape}")

    logger.info("Loading channels compete")

    return c1_c6_img_array


class ImgGen(Sequence):
    def __init__(self, label_data, label_encoder, path, batch_size = 32, preprocess=(lambda x: x), shuffle=False):
    
        if shuffle:
            self.label_data=label_data.sample(frac=1).reset_index(drop=True)
        else:
            self.label_data=label_data
        
        self.batch_size=batch_size
        self.preprocess=preprocess
        self.path=path
        self.label_encoder=label_encoder
        
        self._batches=dict()

        
    def __len__(self):

        logger.info(f"Batch size {self.batch_size}")
        logger.info(f"Label data {len(self.label_data)}")
        batches = np.ceil(len(self.label_data)/float(self.batch_size))
        logger.info(f"Nr of batches {batches}")

        return int(batches)

    def __getitem__(self, i):
        
        if i in self._batches:
            return self._batches[i]

        logger.info("Taking batches")
        batch_x = self.label_data.loc[i*self.batch_size:(i+1)*self.batch_size-1,("experiment","plate","well")]
        
        logger.info(batch_x)
        
        batch_y = self.label_data.loc[i*self.batch_size:(i+1)*self.batch_size-1,("sirna")]
        
        logger.info(batch_y)

        logger.info("Starting batch loading")        
        x = list()

        x = Parallel(n_jobs=-1)(delayed(get_channels)(e, p, w, 1, self.path, self.preprocess) for e, p, w in batch_x.values.tolist())
        
        logger.info("Finished batch loading")        

        y = self.label_encoder.transform(batch_y)

        logger.info(f"Labels size: {len(y)}")

        X = np.array(x)

        logger.info(f"X sixe: {len(X)}, x size {len(x)}, X shape: {X.shape}")

        batch = X, y
        
        self._batches[i]=batch
        
        return batch

