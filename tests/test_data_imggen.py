
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.data_v2 import ImgGen, random_subbox_and_augment




@pytest.fixture(scope="module")
def train_dataset():
    """
    Use the train dataset to test the data loading
    TODO - replace with sythetic data
    """

    train_data_all = pd.read_csv("./data/train.csv")
    sirnas=[
        "sirna_706","sirna_1046","sirna_1045","sirna_586","sirna_747",
        "sirna_705","sirna_1044","sirna_1043","sirna_1042","sirna_1041",
    ]
    train_data_all=train_data_all.loc[train_data_all.sirna.isin(sirnas),:]
    train_data_all.loc[:,["site"]]="1"
    train_data_second_site=train_data_all.copy()
    train_data_second_site.loc[:,["site"]]="2"
    train_data_all=pd.concat([train_data_all,train_data_second_site],axis=0)

    train_data_strat=train_data_all

    train_data_sample_1 = train_data_strat.sample(frac=1,random_state=42).reset_index(drop=True)
    #print("Shape of train_data_sample_1:", train_data_sample_1.shape)
    sirna_label_encoder_sample_1 = LabelEncoder().fit(train_data_sample_1.sirna)
    #print(f"Sirna classes in train_data_sample_1 {len(sirna_label_encoder_sample_1.classes_)}")

    return train_data_sample_1, sirna_label_encoder_sample_1

def test_data_loading_default_no_cache(train_dataset):
    """
    Test the data loading with the default parameters
    """
    
    train, sirna_label_encoder_sample_1 = train_dataset

    train_gen = ImgGen(train,batch_size=16,preprocess=random_subbox_and_augment,shuffle=True,label_encoder=sirna_label_encoder_sample_1, path='./data/train/')

    #print(f"Training set batched size {len(train_gen)}")
    assert len(train_gen) == 42
    
    first_batch = train_gen[0]

    # check that the first batch has the right shape (batch_size, channels, height, width)
    #print(first_batch[0].shape)
    assert first_batch[0].shape == (16, 6, 224, 224)

    # check that the first batch has the right number of classes
    #print(first_batch[1].shape)
    assert first_batch[1].shape == (16,)



def test_data_loading_cached(train_dataset):
    """
    Test the data loading using cached batches
    """
    
    train, sirna_label_encoder_sample_1 = train_dataset

    train_gen = ImgGen(train,batch_size=16,cache=True,preprocess=random_subbox_and_augment,shuffle=True,label_encoder=sirna_label_encoder_sample_1, path='./data/train/')

    #print(f"Training set batched size {len(train_gen)}")
    assert len(train_gen) == 42
    
    first_batch = train_gen[0]

    # check that the first batch has the right shape (batch_size, channels, height, width)
    #print(first_batch[0].shape)
    assert first_batch[0].shape == (16, 6, 224, 224)

    # check that the first batch has the right number of classes
    #print(first_batch[1].shape)
    assert first_batch[1].shape == (16,)

    # check that the batch has been cached
    assert len(train_gen._batches) == 1, "Batch has not been cached"

def test_data_loading_cached_many_accesses(train_dataset):
    """
    Test the data loading using cached batches and accessing batches many times
    """

    train, sirna_label_encoder_sample_1 = train_dataset

    train_gen = ImgGen(train,batch_size=16,cache=True,preprocess=random_subbox_and_augment,shuffle=True,label_encoder=sirna_label_encoder_sample_1, path='./data/train/')

    first_batch = train_gen[0]
    second_batch = train_gen[1]

    assert len(train_gen._batches) == 2, "Batches have not been cached"

    # accessing first batch again should not change the cache
    first_batch_again = train_gen[0]
    second_batch_again = train_gen[1]

    assert first_batch_again[0].shape == (16, 6, 224, 224)
    assert second_batch_again[0].shape == (16, 6, 224, 224)







