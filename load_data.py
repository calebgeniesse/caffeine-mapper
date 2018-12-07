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
import scipy as sp 

import scipy.stats 
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
    data_tmask_dir = os.path.join(data_dir, 'rsfmri/tmasks')
    data_behavior_dir = os.path.join(data_dir, 'behavior')
    data_parcel_dir = os.path.join(data_dir, 'parcellation')



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



##############################################################################
### mask helpers
##############################################################################
def get_RSN_rmask(atlas, n=None, minor=False, ignore=['Zero', 'na'], **kwargs):
    """ Return region mask based on RSNs
    
    Inputs
    ------
        :atlas = Bunch
        :n = int, return rmask for first n networks
        :minor = bool, return rmask for minor networks (major by default)
    """
    rmask = atlas.data.copy()

    # get list of RSNs sorted by size
    RSNs = rmask[~rmask.network.isin(ignore)].groupby('network')
    RSNs = RSNs.network.count().sort_values(ascending=minor)
    RSNs = RSNs.index.tolist()[:n]
    
    # define rmask based on RSNs
    rmask = rmask.assign(data_id = rmask.index)
    rmask = rmask.assign(rmask = rmask.network.isin(RSNs))
    rmask = rmask.loc[rmask.rmask, ['data_id', 'region', 'network', 'rmask']]
    return rmask



def get_session_tmask(meta, session=None, **kwargs):
    """ Return temporal mask for subject(s)

    Inputs
    ------
        :data = data to mask 
        :session = subcode (load all by default, assumes data for all sessions)
    """
    def glob_tmask(subcode=None):
        subcode = 'sub???' if subcode is None else subcode.split('.txt')[0]
        glob_str = os.path.join(config.data_tmask_dir, subcode + '.txt')
        found = sorted(glob.glob(glob_str))
        return found
    
    def load_tmask(filename):
        tmask = pd.read_csv(filename, header=None, names=['tmask'], dtype=bool)
        tmask = tmask.assign(session=os.path.basename(filename).split('.txt')[0]) 
        tmask = tmask.assign(tr_id=tmask.index)
        return tmask
    
    # glob tmask paths (subject specific)
    tmask_paths = sorted(__ for _ in np.ravel(session) for __ in glob_tmask(_))
    
    # load tmasks
    tmask = pd.concat(map(load_tmask, tmask_paths), ignore_index=True, sort=False)
    
    # joint to meta
    tmask = meta.copy().reset_index(drop=False).join(
        tmask.set_index(['session', 'tr_id']), 
        how='inner', on=['session', 'tr_id'],
        )
    
    # assign data_id, return 
    tmask = tmask.fillna(False).assign(data_id = tmask.index)
    tmask = tmask.loc[tmask.tmask, ['data_id', 'session', 'tr_id', 'tmask']]
    return tmask



##############################################################################
### meta data helper functions
##############################################################################
def load_atlas(atlas_file=None):
    """ Load parcellation / atlas data.
    """ 
    if atlas_file is None:
        atlas_file = os.path.join(config.data_parcel_dir, "parcel_data.txt")
    
    # load parcellation into DataFrame
    df_parcel = pd.read_table(atlas_file, header=None)

    # relabel columns
    df_parcel = df_parcel.rename(
        columns={
            0:'target',
            1:'hemisphere',
            2:'x',
            3:'y',
            4:'z',
            5:'region',
            6:'subregion',
            7:'network'
        })

    # region_coords: (x, y, z)
    df_coords = df_parcel[['x', 'y', 'z']].copy()

    # regions: string list of region labels
    df_regions = df_parcel[['region']].copy()

    # networks: names of the networks
    df_networks = df_parcel[['network']].copy()
    
    # Bunch
    atlas = Bunch(
        data=df_parcel,
        regions=df_regions,
        region_coords=df_coords,
        networks=df_networks,
        )
    return atlas



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
        good_rows = df_clean.any(axis=1) & df_clean.std(axis=1).gt(0)
        good_cols = df_clean.any(axis=0) & df_clean.std(axis=0).gt(0)
        df_nonzero = df_clean.loc[good_rows, good_cols]
        df_clean.loc[good_rows, good_cols] = scipy.stats.zscore(df_nonzero, axis=0)
    
    # fill nans
    df_clean = df_clean.fillna(0.0)
    
    # return cleaned
    return df_clean



##############################################################################
### main loaders
##############################################################################
def load_scrubbed(**kwargs):
    """ Loads scrubbed data
    """
    logger = logging.getLogger(__name__)
    logger.info('load_scrubbed(**{})'.format(kwargs))
    
    # file path (avmovie only, for now)
    glob_str = os.path.join(config.data_scrubbed_dir, "sub???.txt")
    data_paths = sorted(glob.glob(glob_str))
    tmask_paths = [os.path.join(config.data_tmask_dir, os.path.basename(_)) for _ in data_paths]

    glob_str = os.path.join(
        config.data_behavior_dir, 'trackingdata_goodscans.txt')
    meta_paths = sorted(glob.glob(glob_str))

    # hacky, but just print command to fetch data, if not already
    if len(data_paths) < 1:
        fetch_data()
        return None

    # load atlas
    atlas = load_atlas()
    
    # how many sessions to load?
    n_sessions = kwargs.get('n_sessions', -1)
    if n_sessions == -1:
        n_sessions = len(data_paths)

    # check sizes
    logger.debug('found {} data files'.format(len(data_paths)))
    logger.debug('found {} tmask files'.format(len(tmask_paths)))
    logger.debug('found {} meta files'.format(len(meta_paths)))
    logger.debug('using {} sessions'.format(n_sessions))

    # load data ?
    logger.info("Loading data...")
    dataset = []
    for i, data_path in enumerate(data_paths):

        if i >= n_sessions:
            break

        logging.info("  [+] session: {}, file: {}".format(i, os.path.basename(data_path)))
        
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
           
         # load tmask (subject specific)
        df_tmask = get_session_tmask(df_meta, session=session, **kwargs.get('tmask_kwds', {}))
        if kwargs.get('apply_tmask'):
            df_data = df_data.loc[df_tmask.data_id, :]
            df_meta = df_meta.loc[df_tmask.data_id, :]
            logging.info("      keeping: {} (time points)".format(df_data.shape[0]))

        # load rmask (region specific)
        df_rmask = get_RSN_rmask(atlas, **kwargs.get('rmask_kwds', {}))
        if kwargs.get('apply_rmask'):
            df_data = df_data.loc[:, df_rmask.data_id]
            logging.info("      keeping: {} (regions)".format(df_data.shape[-1]))


         # clean data, meta
        #df_data = clean_data(data=df_data, meta=df_meta, **kwargs)
        #df_meta = clean_meta(df_meta, **kwargs)
      
        # z score (?)
        if kwargs.get('zscore'):
            logging.info("      zscore: {}".format(kwargs.get('zscore')))
            df_data.loc[:] = scipy.stats.zscore(df_data.values, axis=0)


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

        # reset meta index 
        df_meta = df_meta.set_index(['session', 'session_id', 'tr_id'])

      
        # save masker, x
        dataset.append(Bunch(
            data=df_data.copy().fillna(0.0),
            meta=df_meta.copy().fillna(0),

            #masker=masker,
            X=df_data.values.copy(),
            y=df_meta.values.copy(),
            
            # masks
            tmask=df_tmask.copy(),
            rmask=df_rmask.copy(),

            # atlas
            atlas=atlas,
            ))
    
    
    # return dataset as Bunch
    if kwargs.get('merge') is False:
        for i, session in enumerate(dataset):
            meta = (clean_meta(session.meta, **kwargs)
                            .reset_index(drop=False)
                            )
            dataset[i].meta = meta.copy()
            dataset[i].y = meta.set_index('session').values.copy()
        return dataset

    # nerge data into single dataframe, array, etc
    dataset = Bunch(
        data=pd.concat((_.data for _ in dataset), ignore_index=True, sort=False).fillna(0.0),
        meta=pd.concat((_.meta for _ in dataset), ignore_index=False, sort=False),
        tmask=pd.concat((_.tmask for _ in dataset)),
        rmask=pd.concat((_.rmask for _ in dataset)),
        atlas=dataset[0].atlas,
        )
    dataset.meta = (clean_meta(dataset.meta, **kwargs)
                    .reset_index(drop=False)
                    .set_index('session')
                    )


    dataset.X = dataset.data.values.reshape(-1, dataset.data.shape[-1])
    dataset.y = dataset.meta.values.reshape(-1, dataset.meta.shape[-1])
    return dataset



   
