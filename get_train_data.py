import sqlite3
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from utils import ColorPredictionDataset

RANDOM_SEED=42
np.random.seed(RANDOM_SEED)


def scale_rgb(rgb_val: int) -> float:
    return rgb_val / 255

def get_df() -> pd.DataFrame:

    connection = sqlite3.connect('./colorsurvey/users.db')
    # get distinct colornames with average rgb values
    answers = pd.read_sql_query('select colorname, avg(r) as r, avg(g) as g, avg(b) as b from answers group by colorname', connection)

    for col in 'rgb':
        answers[col] = answers[col].apply(scale_rgb)
    
    return answers

def get_train_val_test_splits(
    df: pd.DataFrame, 
    val_size: float=0.1, 
    test_size: float=0.1, 
    num_samples: Optional[int]=None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if num_samples is not None:
        df = df.sample(n=num_samples, random_state=RANDOM_SEED)
    else:
        df = df.sample(frac=1, random_state=RANDOM_SEED)
    
    train_size = 1 - (val_size + test_size)
    # create splits based on end index of train dataset and end index of validation dataset
    train, val, test = \
            np.split(df, [int(train_size*len(df)), int((train_size + val_size)*len(df))])
    
    return train, val, test

# TODO: decouple dataset creation from type of dataset ie take dataset class as param
def create_dataset(df: pd.DataFrame, tokenize_fn) -> torch.utils.data.Dataset:
    X, y = df['colorname'], df[['r', 'g', 'b']].to_numpy(np.float64)
    return ColorPredictionDataset(texts=list(X), labels=y, tokenize_fn=tokenize_fn)

