"""
dataset loader.
@author Caleb Geniesse
"""
from __future__ import print_function, division
from __future__ import unicode_literals

import os
import re
import glob
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd 

from sklearn.datasets.base import Bunch
#from sklearn.preprocessing import LabelEncoder
#from nilearn.input_data import NiftiLabelsMasker
#from nilearn.image import load_img
#from nilearn.signal import clean
#from nilearn.datasets import fetch_atlas_msdl

class config:
    # define some paths
    data_dir = 'data/base/'
    data_scrubbed_dir = os.path.join(data_dir, 'combined_data_scrubbed')
    data_behavior_dir = os.path.join(data_dir, 'behavior')

  

def fetch_data(**kwargs):
    """ Fetch my connectome data.
        
        Just print command to download data for now... too long to run in-script.
    """
    url = "http://web.stanford.edu/group/poldracklab/myconnectome-data/base/combined_data_scrubbed"
    cmd = "wget -N -r -l inf --no-remove-listing -nH --cut-dirs=3 {}".format(url)
    print("Run the following in a new Jupyter cell to fetch data:\n")
    print("%%bash")
    print("mkdir -p data")
    print("cd data")
    print("{}".format(cmd))
    # import subprocess as sp
    # o = sp.check_output(cmd, shell=True)
    # 
    return True




def clean_meta(df, columns=None, zscore=False, **kwargs):
    """ Clean meta DataFrame, zscore.
    """
    # copy input df
    df_clean = df.copy()
    
    # select columns
    if columns is not None:
        df_clean = df_clean[columns]
    
    # replace non-numerics
    df_clean = df_clean.replace('.', 0.0)
    df_numeric = df_clean.astype(str).applymap(str.isnumeric)
    df_clean[df_numeric==False] = np.nan
    df_clean = df_clean.fillna(0.0)

    # only keep numeric data, convert to float
    df_clean = df_clean.astype(float)
    
    # z-score values
    if zscore is True:
        from scipy.stats import zscore
        good_rows = df_clean.any(axis=1)
        good_cols = df_clean.any(axis=0)
        df_nonzero = df_clean.loc[good_rows, good_cols]
        df_clean.loc[good_rows, good_cols] = zscore(df_nonzero, axis=0)
    
    # fill nans
    df_clean = df_clean.fillna(0.0)
    
    # return cleaned
    return df_clean



def load_scrubbed(**kwargs):
    """ Loads scrubbed data
    """
    logger = logging.getLogger(__name__)
    logger.info('load_scrubbed(**{})'.format(kwargs))
    
    # file path (avmovie only, for now)
    glob_str = os.path.join(config.data_scrubbed_dir, "sub???.txt")
    data_paths = sorted(glob.glob(glob_str))

    glob_str = os.path.join(
        config.data_behavior_dir, 'trackingdata_goodscans.txt')
    meta_paths = sorted(glob.glob(glob_str))

    # hacky, but just print command to fetch data, if not already
    if len(data_paths) < 1:
        fetch_data()
        return None
    
    # how many sessions to load?
    n_sessions = kwargs.get('n_sessions', -1)
    if n_sessions == -1:
        n_sessions = len(data_paths)

    
    # check sizes
    logger.debug('found {} data files'.format(len(data_paths)))
    logger.debug('found {} meta files'.format(len(meta_paths)))
    logger.debug('using {} sessions'.format(n_sessions))

    # load data ?
    logger.info("Loading data...")
    dataset = []
    for i, data_path in enumerate(data_paths):

        if i >= n_sessions:
            break

        logging.info("  [+] session: {}, path: {}".format(i+1, data_path))
        
        # load dataframe
        df_data = pd.read_csv(data_path, header=None, delim_whitespace=True)

        # load meta as tr_id (for now...)
        df_meta = df_data.assign(tr_id = df_data.index.values)[['tr_id']]
        
        # parse session, session_id from file
        session = os.path.basename(data_path).split('.txt')[0]
        session_id = int(''.join([__ for __ in session if __.isdigit()]))
        df_meta = df_meta.assign(session=session, session_id=session_id)
        
        # join with other meta files
        df_meta = df_meta.join(
            pd.concat(pd.read_table(_,index_col='subcode') for _ in meta_paths)
            , how='left', on='session'
            )
        df_meta = df_meta.set_index(['session', 'session_id', 'tr_id'])
        
        # clean meta
        #df_meta = clean_meta(df_meta, **kwargs)

        #masker = NiftiLabelsMasker(
        #    labels_img=atlas_paths[0],
        #    memory="nilearn_cache"
        #    )
        #masker = masker.fit()

        # low pas filter
        #cleaned_ = clean(df_data.values,
        #    low_pass=0.09, high_pass=0.008
        #    )
        #df_data.iloc[:, :] = cleaned_


        # save masker, x
        dataset.append(Bunch(
            data=df_data.copy().fillna(0.0),
            meta=df_meta.copy().fillna(0),

            #masker=masker,
            X=df_data.values.copy(),
            y=df_meta.values.copy()
            ))
    
    
    # return dataset as Bunch
    if kwargs.get('merge') is False:
        for i, session in enumerate(dataset):
            session.meta = (clean_meta(session.meta, **kwargs)
                            .reset_index(drop=False)
                            )
            dataset[i].meta = session.meta.copy()
            dataset[i].y = session.meta.set_index('session').values
        return dataset

    # nerge data into single dataframe, array, etc
    dataset = Bunch(
        data=pd.concat((_.data for _ in dataset), ignore_index=True, sort=False).fillna(0.0),
        meta=pd.concat((_.meta for _ in dataset), ignore_index=False, sort=False),
        #atlas=[_.atlas for _ in dataset][0],
        )
    dataset.meta = (clean_meta(dataset.meta, **kwargs)
                    .reset_index(drop=False)
                    .set_index('session')
                    )

    dataset.X = dataset.data.values.reshape(-1, dataset.data.shape[-1])
    dataset.y = dataset.meta.values.reshape(-1, dataset.meta.shape[-1])
    return dataset



   
