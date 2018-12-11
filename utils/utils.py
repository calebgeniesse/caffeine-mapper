"""
Some utils for caffeine-mapper...
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals


import dyneusr as ds
import kmapper as km
import networkx as nx
import bct

import numpy as np
import pandas as pd
import scipy.stats

from sklearn.datasets.base import Bunch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE

from umap.umap_ import UMAP
from hdbscan import HDBSCAN

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter, OrderedDict
from functools import partial
from itertools import product



##############################################################################
### helper functions
##############################################################################
def get_data_splits(data_, zscore=True, groupby='day_of_week', groups=None, **kwargs):
    """ Return data splits.
    
    Inputs
    ------
        :data_ = Bunch,  merged sessions 
        :groups = dict, {group: name} mapping of splits
        
    """
    x_ = data_.X.copy()
    
    # z-score
    if zscore is True:
        x_ = scipy.stats.zscore(x_, axis=0)

    # group by fed / fasted
    grouped = data_.meta.groupby(groupby)
    
    # get splits    
    splits = dict()
    for g_i, (group, df_group) in enumerate(grouped):
        
        # extract data for fed / fasted
        x_group = x_[df_group.index, :]

        # extract RSNs data for fed / fasted
        df_rsn_group = get_RSN_act(x_group, data_.rmask, **kwargs)

        # name split
        try:
            name = '{:}_{:}'.format(groupby, str(g_i).zfill(len(str(grouped.ngroups))))
            name = groups.get(group) or groups.get(int(group)) or name
        except Exception as e:
            pass
            
        # save split
        splits[name] = split = Bunch(
            meta=df_group.copy(),
            X=x_group.copy(),
            RSN=df_rsn_group.copy(),
            group=group,
            name=name,
            )
    
        # print shapes
        print("{:15} => {:15}  x.shape: {}  RSN.shape: {}".format(
            group, name, split.X.shape, split.RSN.shape
        ))
    
    # return as Bunch
    splits = Bunch(**splits)
    return splits



def get_RSN_act(x, rsn, zscore=True, density=None, threshold=0.05, binary=True):
    """ Compute mean activity for RSN at each TR.
    
    Inputs
    ------
        :x = np.ndarray (TR, ROI)
        :rsn = pd.DataFrame (ROI, RSN)
        :zscore = bool, whether or not to zscore x
        :density = float, set everything below 1-density to 0
        :threshold = float, set everything above threshold to 1 (set density=1.0)
        :binary = bool, whether or not to binarize results
        
    """
    x_ = x.copy() 
    rsn_ = rsn.copy()
    if 'networks' in rsn_:
        rsn_ = dict(rsn_).get('networks')
    if isinstance(rsn_, pd.Series):
        rsn_ = rsn_.to_frame()

    # reset indices (i.e. if rmask was applied to data...)
    if rsn_.shape[0] > x_.shape[-1]:
        rsn_ = rsn_.reset_index(drop=True)
    elif rsn_.index.max() > x_.shape[-1]:
        rsn_ = rsn_.reset_index(drop=True)

    # z-score (?)
    if zscore is True:
        x_ = scipy.stats.zscore(x_, axis=0)

    # get average RSN activity for each network
    #       sorted by -len(ROIs), -sum(ROIs) 
    # TODO: this could be its own function
    rsn_rois = rsn_.groupby('network').indices.items()
    rsn_rois = sorted(rsn_rois, key=lambda _: [-len(_[-1]), -sum(_[-1])])
    rsn_act = {rsn: x_[:, rois].mean(axis=1) for (rsn, rois) in rsn_rois}

    # save as DataFrame
    df_rsn_act = pd.DataFrame(rsn_act)
    
    # threshold / density (?)
    if density is not None:
        threshold = df_rsn_act.quantile(1.0-density, axis=0)
        df_rsn_act[df_rsn_act.lt(threshold)] = 0

    # threshold (?)
    if threshold is None:
        threshold = df_rsn_act.mean(axis=0)

    # binary (?)
    if binary is True:
        df_rsn_act = (df_rsn_act >= threshold).astype(int)
 
    return df_rsn_act




def get_PC(df, columns=None):
    """ Find PC with max weight assigned to column.
    """
    # extract X
    X = df.values
    
    # fit transform PCA
    from sklearn.decomposition import PCA
    pca = PCA()
    U = pca.fit_transform(X)
    V = pca.components_
    
    # create new DataFrame of components (V)
    df_V = pd.DataFrame(V, columns=df.columns)
    df_V.index = df_V.index.map('PC_{}'.format)
   
    # create new DataFrame of embedding (U)
    df_U = pd.DataFrame(U, index=df.index, columns=df_V.index)
    
    # set column to 0,1 or idxmax   
    if columns is None:
        columns = pd.Series(df_U.columns[:2])
    else:
        columns = [_ for _ in np.ravel(columns)]
        columns = df_V[columns].idxmax()   

    # return PC for col
    df_PC = df_U[columns]
    df_PC.columns = ["{} [{}]".format(columns[_],_) for _ in columns.index]
    return df_PC




def run_mapper(X=None, y=None, X_inverse=True, lens=None, verbose=0, **params):
    """ Wrap KeplerMapper calls
    
    Notes
    -----
    - See PCA_metadata.ipynb

    """
    # init MAPPER params
    projection = params.get('projection', TSNE(perplexity=50, init='random', random_state=0))
    clusterer = params.get('clusterer', HDBSCAN(allow_single_cluster=True))
    cover = params.get('cover', km.Cover(10, 0.67))
    X_inverse = X if X_inverse is True else X_inverse
    
    # fit
    if lens is None:
        mapper = km.KeplerMapper(verbose=verbose-1)
        lens = mapper.fit_transform(X, projection=projection)
       
    # map
    mapper = km.KeplerMapper(verbose=verbose)
    graph = mapper.map(lens, X_inverse, clusterer=clusterer, coverer=cover)
  
    # dG
    dG = ds.DyNeuGraph()
    dG.fit(graph, y=y)
    
    # save results
    results = Bunch(
        X=X.copy(), X_inverse=X, 
        lens=lens.copy(), graph=graph, 
        params=params, cover=cover,
        dG=dG, G=dG.G_, TCM=dG.tcm_.copy()
    )
    return results
    

    

def draw_G(G, y=None, pos=None, ax=None, **kwargs):
    """ Draw networkx graph.

    Notes
    -----
    - See PCA_metadata.ipynb
    
    """
    
    # size by number of members
    node_size = kwargs.get('node_size')
    if node_size is None:
        node_size = [1 * len(G.nodes[n]['members'])**1.5 for n in G]
        kwargs.update(node_size=node_size)

    
    # color nodes by mode
    node_color = kwargs.get('node_color')
    if node_color is None and y is not None:
        node_color = [Counter(y[_]).most_common()[0][0] for n,_ in G.nodes('members')]
        kwargs.update(node_color=node_color)
    
    # color nodes by mode
    layout = kwargs.get('layout')
    if pos is None and layout is not None:
        pos = layout(G)
    
    # plot
    _ = nx.draw_networkx(
        G, pos=pos,
        with_labels=False, 
        ax=ax,
        **kwargs
        )
    
    # remove spines
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    return ax
    
