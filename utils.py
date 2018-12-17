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

from load_data import *

##############################################################################
### classes
##############################################################################
class Config(Bunch):
    pass



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
        	atlas=Bunch(**dict(data_.atlas)),
        	rmask=data_.rmask.copy(),
        	data=data_.data.iloc[df_group.index, :].copy(),
            meta=df_group.copy(),
            X=x_group.copy(), y=data_.y[df_group.index].copy(),
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


 
##############################################################################
### helper functions
##############################################################################
RSN_LABELS = np.ravel([
    'Cingulo_opercular',
    'Frontoparietal_1',
    'Somatomotor',
    'Visual_2',
    'DMN',
    'Dorsal_Attention', 
    'Ventral_Attention', 
    'Salience', 
    'Visual_1', 
    'Medial_Parietal',
    'Parieto_occipital', 
    'Frontoparietal 2'
    ])

RSN_LABELS_PRETTY = np.ravel([
    'Cingulo-opercular',
    'Fronto-parietal 1',
    'Somatomotor',
    'Visual 2',
    'DMN',
    'Dorsal Attention', 
    'Ventral Attention', 
    'Salience', 
    'Visual 1',
    'Medial Parietal',
    'Parieto Occipital', 
    'Fronto-parietal 2'
    ])


def get_majorRSN(rmask, sort=True, encode=True, n=5):
    """
    Usage
    -----
        majorRSN, majorRSNs = get_majorRSN(combined.rmask, n=5)
        majorRSN.groupby('network').first()
    """
    rmask_ = rmask.copy().reset_index(drop=True)
    rsn_, _ = get_RSN_map(rmask_, n=n)
    
    # roi2rsn 
    roi2rsn_ = rmask_[[]].assign(
        data_id=rsn_.reset_index(drop=True).index,
        region=rmask_.region,
        network=rsn_.idxmax(axis=1)
    )
    # sort by network
    roi2rsn_ = roi2rsn_.reset_index(drop=False)
    if sort is True:
        roi2rsn_ = roi2rsn_.sort_values(['network','region'])
    roi2rsn_ = roi2rsn_.set_index('index')
    # encode network => target, label
    target, label = pd.factorize(roi2rsn_['network'])
    if encode is True:
        roi2rsn_ = roi2rsn_.assign(target=target, label=label[target])
    return roi2rsn_, label


def get_majorROI(rmask, sort=True, encode=True, n=5):
    """
    Usage
    -----
        majorROI, majorROIs = get_majorROI(combined.rmask, n=5)
        majorROI.groupby('region').first()
    """
    rmask_ = rmask.copy().reset_index(drop=True)
    rsn_, _ = get_RSN_map(rmask_, n=n)
    
    # roi2rsn 
    roi2rsn_ = rmask_[[]].assign(
        data_id=rsn_.reset_index(drop=True).index,
        region=rmask_.region,
        network=rsn_.idxmax(axis=1)
    )
    # sort by network
    roi2rsn_ = roi2rsn_.reset_index(drop=False)
    if sort is True:
        roi2rsn_ = roi2rsn_.sort_values(['network','region'])
    roi2rsn_ = roi2rsn_.set_index('index')
    # encode network => target, label
    target, label = pd.factorize(roi2rsn_['region'])
    if encode is True:
        roi2rsn_ = roi2rsn_.assign(target=target, label=label[target])
    return roi2rsn_, label



def get_RSN_map(rsn, n=None, sort=False):

    """ Return mapping of ROIs => RSNs (pandas.DataFrame)
    """
    allRSNs = np.ravel(RSN_LABELS)
    majorRSNs = np.ravel(allRSNs)

    # match to rmask
    if n and len(allRSNs) > n:
        majorRSNs = majorRSNs[:n]
    
    # sort (?)
    if sort is True:
        majorRSNs = np.sort(majorRSNs)

    # Extract ROI, RSN labels from parcellation
    if 'data' in rsn:
        rsn = rsn['data']
    roi2rsn = rsn.network.copy()

    # Store encoded RSN labels in df_roi_rsn (pandas.DataFrame)
    roi_index = range(roi2rsn.index.min(), roi2rsn.index.max()+1)

    df_rsn_onehot = pd.DataFrame(0, index=roi_index, columns=majorRSNs)
    df_rsn_labels = pd.DataFrame('zero', index=roi_index, columns=['network', 'order'])

    # Mask RSN -> major RSN only
    roi2rsn = roi2rsn[roi2rsn.isin(majorRSNs)].reset_index(drop=False)

    # Get mapping of RSN -> ROI indices 
    rsn2roi = roi2rsn.groupby('network').indices
    
    # assign 1 for every set of  rois, rsn
    for rsn, rois in rsn2roi.items():
        print("RSN: {}  (size={})".format(rsn, len(rois)))
        df_rsn_onehot.loc[rois, rsn] = 1
        df_rsn_labels.loc[rois, 'network'] = rsn
        df_rsn_labels.loc[rois, 'order'] = list(majorRSNs).index(rsn)

    # return data frame
    df_rsn_labels = df_rsn_labels.reset_index(drop=True)
    reorder_index = df_rsn_labels.astype('str').sort_values(by='order', ascending=True).index
    df_rsn_labels = df_rsn_labels.loc[reorder_index, ['network']].astype(str)
    return df_rsn_onehot, df_rsn_labels




def get_RSN_act(x, rsn, zscore=True, density=None, threshold=0.5, binary=True):
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




def run_mapper(X=None, y=None, X_inverse=True, lens=None, zscore=False, verbose=0, **params):
    """ Wrap KeplerMapper calls
    
    Notes
    -----
    - See PCA_metadata.ipynb

    """
    X_ = np.copy(X)
    if zscore is True:
        X_ = scipy.stats.zscore(X_, axis=0)

    # init MAPPER params
    projection = params.get('projection', TSNE(perplexity=50, init='pca', random_state=0))
    clusterer = params.get('clusterer', HDBSCAN(allow_single_cluster=True))
    cover = params.get('cover', km.Cover(10, 0.67))
    X_inverse = X_ if X_inverse is True else X_inverse
    
    # fit
    if lens is None:
        mapper = km.KeplerMapper(verbose=verbose-1)
        lens = mapper.fit_transform(X_, projection=projection)
       
    # map
    mapper = km.KeplerMapper(verbose=verbose)
    graph = mapper.map(lens, X_inverse, clusterer=clusterer, coverer=cover)
  
    # dG
    dG = ds.DyNeuGraph(G=graph, y=y)
    
    # save results
    results = Bunch(
        X=X_, y=y, X_inverse=X_, 
        lens=lens.copy(), graph=dict(graph), 
        projection=projection, clusterer=clusterer, cover=cover,
        dG=dG, 
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
    


##############################################################################
### Network analysis
##############################################################################
import community
from bct import participation_coef
import collections


def get_mod(G, return_partition=False):
    partition = dict()
    # Get groups - node attribute that has a 1 for the RSN the node should belong to
    groups = nx.get_node_attributes(G,'group')
    # Iteration through each node (dictionary)
    for n in groups.keys():
        # Loop through each RSN and pull out the only one that has a 1 - that's our label
        # Implemented a try catch for rare cases where a node has a tie between groups
        try:
            nwlabel = [nw for nw in groups[n] if groups[n][nw] == 1][0]
        # Basically go through proportions which has the number of members in each RSN,
        # ... find the max, and choose network that is alphabetically first
        except IndexError:
            props = nx.get_node_attributes(G,'proportions')
            values = [dic['value'] for dic in props[n]['multiclass']]
            highestnws = np.argwhere(values == np.max(values))
            highestnws = [sublist[0] for sublist in highestnws]
            nwlabel = props[n]['multiclass'][highestnws[0]]['label']
        partition[n] = nwlabel
    Q = community.modularity(partition,G)
    if return_partition:
        return Q,partition
    else:
        return Q
    
    
def within_module_degree(G, partition, normalize = False):
    # If we want to normalize by community size
    if normalize:
        # Get size of each RSN community
        nodecount = collections.Counter(partition.values())
    inmod_deg = collections.defaultdict(list)
    # Loop through nodes, calculate degree within module, append to list by RSN
    for node in G.nodes():
        # Get neighbors of node and only count the ones that are in same RSN
        wmod = len([1 for nbr in G[node] if partition[nbr] == partition[node]])
        if normalize:
            # Normalize by community size
            wmod /= float(nodecount[partition[node]])
        inmod_deg[partition[node]].append(wmod)
    # This is a dictionary keyed by RSN, values are lists of within module degree of every node in RSN
    return inmod_deg 


def betweenness(G, partition):
    BC = nx.betweenness_centrality(G)
    btw = collections.defaultdict(list)
    # Loop through nodes with betweenness calculated, and append to appropriate RSN
    for node in BC:
        btw[partition[node]].append(BC[node])

    return btw # This is a dictionary keyed by RSN, values are lists of betweenness of every node in RSN


def calc_particip(G,partition,A,C):
    # Calculate participation coefficient for each node
    P = participation_coef(A,C)
    # Create a dictionary keyed by RSN, values are lists of particip coef of every node in RSN
    particip = collections.defaultdict(list)
    for ind,p in enumerate(P):
        particip[partition[list(G.nodes())[ind]]].append(p)
        
    return particip



def plot_network_measure(network_measure1,network_measure2,y_axlabel,plot_title,y_lim,null_measure=None):
    """
    Take a dictionary keyed by RSN, values are lists of a network measure value for every node in RSN
    Plot bars comparing fed and fasted states for each RSN
    """
    from matplotlib import cm

    # define majorRSNs
    majorRSNs = sorted(network_measure1.keys())

    # Bar plot for fed - this is a list of tuples (RSN,mean,std)
    bars_fed = [
        (np.mean(network_measure1[rsn]), np.std(network_measure1[rsn])) 
        for rsn in sorted(network_measure1.keys())]
    barh_fed,barerr_fed = list(zip(*bars_fed))

    # Bar plot for fast - this is a list of tuples (RSN,mean,std)
    bars_fast = [
        (np.mean(network_measure2[rsn]), np.std(network_measure2[rsn])) 
        for rsn in sorted(network_measure2.keys())]
    barh_fast,barerr_fast = list(zip(*bars_fast))

    cmap = cm.get_cmap('viridis', 20)
    colors = cmap(np.linspace(0,1,10))

    index = np.arange(len(bars_fed))
    
    if null_measure is None:
        bar_width = 0.35
        error_config = {'ecolor': '0.3'}

        plt.close('all')
        fig, ax = plt.subplots(figsize=(12,4))

        rects1 = ax.bar(index, barh_fed, bar_width,
                        color=colors[2],
                        yerr=barerr_fed, error_kw=error_config,
                        label='Fed')
        rects2 = ax.bar(index + bar_width, barh_fast, bar_width,
                        color=colors[7],
                        yerr=barerr_fast, error_kw=error_config,
                        label='Fasted')

        ax.set_ylabel(y_axlabel)
        ax.set_title(plot_title)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(majorRSNs,fontsize=7)
        ax.set_ylim([0,y_lim])
        ax.legend()

        fig.tight_layout()
        plt.show()
    else:
        bars_null = [(np.mean(null_measure[rsn]),np.std(null_measure[rsn])) for rsn in sorted(null_measure.keys())]
        barh_null,barerr_null = list(zip(*bars_null))
        
        bar_width = 0.25
        error_config = {'ecolor': '0.3'}

        plt.close('all')
        fig, ax = plt.subplots(figsize=(12,4))

        rects1 = ax.bar(index - bar_width, barh_fed, bar_width,
                        color=colors[2],
                        yerr=barerr_fed, error_kw=error_config,
                        label='Fed')
        rects2 = ax.bar(index, barh_fast, bar_width,
                        color=colors[7],
                        yerr=barerr_fast, error_kw=error_config,
                        label='Fasted')
        rects3 = ax.bar(index + bar_width, barh_null, bar_width,
                        color='gray',
                        yerr=barerr_null, error_kw=error_config,
                        label='SBM')

        ax.set_ylabel(y_axlabel)
        ax.set_title(plot_title)
        ax.set_xticks(index)
        ax.set_xticklabels(majorRSNs,fontsize=7)
        ax.set_ylim([0,y_lim])
        ax.legend()

        fig.tight_layout()
        plt.show()
    return fig, ax 




##############################################################################
### ROI x ROI adjacency
##############################################################################
def rsn_index_change(sorted_rsns):
    """
    Process sorted array of network labels, return an array of indexes where the label changes
    """
    index_array = []
    current_label = 'batman'
    for ind,label in enumerate(sorted_rsns):
        if label != current_label:
            current_label = label
            index_array.append(ind)
    # Add ending index
    index_array.append(len(sorted_rsns)-1)
    return index_array


def add_rsn_patches(ax, lenx, leny, boundaries, color_array, alpha_param = 1.0, offset = 1.0):
    """
    Plots rectangular patches vertically and horizontally for each network bounded by the indices
    provided using the colors provided
    """
    from matplotlib import patches

    # Loop through boundaries
    for num,ind in enumerate(boundaries[:-1]):
        nextind = boundaries[num+1]
        hrect = patches.Rectangle((ind,ind), nextind-ind, nextind-ind,
            edgecolor=color_array[num],facecolor='none',
            linewidth=1.5, alpha=alpha_param)
        ax.add_patch(hrect)
        vrect = patches.Rectangle((ind,ind), nextind-ind, nextind-ind,
            edgecolor=color_array[num], facecolor='none',
            linewidth=1.5, alpha=alpha_param)
        ax.add_patch(vrect)
        
        

def plot_network_adj(TCM, plot_title, add_cbar=True, labels=None, cmap='binary_r', **kwargs):
    
    import matplotlib as mpl
    
    #elif 'values' in dir(labels):
    #    labels = labels.values

    # labels should have entry for every entry in TCM
    #labels_idx = np.arange(len(labels)) 
    #if sort:
    #labels = np.sort(RSN_LABELS)
    labels_idx = np.argsort(labels)
    #labels_idx = np.arange(TCM.shape[0])
    #if 'index' in dir(labels):
    #    labels_idx = labels.to_frame().reset_index().index.values
    #    labels = labels.values
    #labels_set = np.ravel(sorted(set(labels), key=lambda _: list(labels).index(_)))
   


    A = TCM.copy()
    A = A[labels_idx, :]
    A = A[:, labels_idx]
    rect_inds = rsn_index_change(labels[labels_idx])
    print(rect_inds)
    
    # get fig, axes
    fig = kwargs.get('figure')
    ax = kwargs.get('ax') 
    if fig is None and ax is None:
        plt.close('all')
        fig, ax = plt.subplots(1, 1)
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.add_subplot(111)
    else:
        fig, ax = plt.subplots(1, 1)
       
    plt.rc('font', family='serif')
    ax.imshow(A, cmap=cmap)
    ax.axis('off')
    ax.set_title(plot_title)
    cmap = mpl.cm.get_cmap('tab20', 20)
    colors = cmap(np.linspace(0,1, len(np.unique(labels))))

    add_rsn_patches(
        ax,A.shape[1], A.shape[0], rect_inds,
        colors, alpha_param=1.0, offset=2)
    
    # Create colorbar to label the RSN rectangles
    if add_cbar:
        # Use rect_inds to define which ticks are what colors
        # If our colormap is size of the matrix (554), then all indices 
        # ... within a RSN should be same color
        bar_colors = np.zeros((A.shape[0], 4))
        # Holds the midpoint value of each RSN region to use for tick labels
        midticks = []
        for num,ind in enumerate(rect_inds[:-1]):
            nextind = rect_inds[num+1]
            bclr = np.matlib.repmat(colors[num],nextind-ind,1)
            bar_colors[ind:nextind,:] = bclr
            midticks.append((nextind+ind)/2.0)
        # Flip since matrix goes top to bottom
        bar_colors = bar_colors[::-1]
        midticks = A.shape[0] - np.array(midticks)
        
        cmap_bar = mpl.colors.ListedColormap(bar_colors)
        norm = mpl.colors.Normalize(vmin=0,vmax=A.shape[0])
        sm = plt.cm.ScalarMappable(cmap=cmap_bar, norm=norm)
        sm.set_array([])
        
        # Left colorbar
        fig.tight_layout(rect=[0.21, 0.11, 0.98, 0.88])
        cbaxes = fig.add_axes([0.18, 0.11, 0.02, 0.77]) 
        cbar = plt.colorbar(sm, ticks=midticks, cax = cbaxes)
        cbaxes.yaxis.set_ticks_position('left')
        labels_pretty = np.sort(RSN_LABELS_PRETTY)
        labels_pretty = labels_pretty[:len(np.unique(labels))]
        cbar.ax.set_yticklabels(labels_pretty)
        cbar.ax.tick_params(labelsize=7) 

    plt.show()
    return fig, ax







##############################################################################
### Null Models
##############################################################################
import operator
from matplotlib import patches
import collections

class nullSBM(object):
    def __init__(self):
        pass
    
    def __rsn_index_change(self,sorted_rsns):
        """
        Process sorted array of network labels, 
        ... return an array of indexes where the label changes
        """
        index_array = []
        current_label = 'batman'
        for ind,label in enumerate(sorted_rsns):
            if label != current_label:
                current_label = label
                index_array.append(ind)
        # Add ending index
        index_array.append(len(sorted_rsns))
        return index_array
    
    def __get_mod(self,G):
        """
        Partition graph G into communities
        """
        partition = dict()
        # Get groups - node attribute that has a 1 for the RSN the node 
        # ... should belong to
        groups = nx.get_node_attributes(G,'group')
        # Iteration through each node (dictionary)
        for n in groups.keys():
            # Loop through each RSN and pull out the only one that has 
            # ... a 1 - that's our label
            # Implemented a try catch for rare cases where a node has 
            # ... a tie between groups
            try:
                nwlabel = [nw for nw in groups[n] if groups[n][nw] == 1][0]
            # Basically go through proportions which has the number of
            # ... members in each RSN, find the max, and choose network
            # ... that is alphabetically first
            except IndexError:
                props = nx.get_node_attributes(G,'proportions')
                values = [dic['value'] for dic in props[n]['multiclass']]
                highestnws = np.argwhere(values == np.max(values))
                highestnws = [sublist[0] for sublist in highestnws]
                nwlabel = props[n]['multiclass'][highestnws[0]]['label']
            partition[n] = nwlabel
        return partition
    
    def __sort_adjacency(self,G):
        """
        Take the adjacency matrix from the Mapper graph and sort it by RSN
        """
        partition = self.__get_mod(G)
        sorted_p = sorted(partition.items(), key=operator.itemgetter(1))
        # Creat new adjacency matrix from the partition
        nodes,comms = list(zip(*sorted_p))
        A = np.zeros((len(nodes),len(nodes)))
        for node in nodes:
            # Find neighbors of node
            for nbr in G[node]:
                # Create an edge
                A[nodes.index(node),nodes.index(nbr)] = 1
        # Return
        return A,comms  

    def __calc_SBM_param(self,G):
        """
        Take an ROI x ROI matrix from a DyNeuSR and estimates probabilities 
        ... of connections between communities
        """
        A,labels = self.__sort_adjacency(G)
        # Get number of nodes in each community
        nodecount = collections.Counter(labels)
        sizes = [nodecount[key] for key in sorted(nodecount.keys())]
        # Get the indices of the bounds between RSN regions in the matrix
        bounds = self.__rsn_index_change(labels)
        # Create array to hold probabilities
        probs = np.zeros((len(np.unique(labels)),len(np.unique(labels))))
        # Go through bounds twice, once for each dimension
        for comm1,b1 in enumerate(bounds[:-1]):
            for comm2,b2 in enumerate(bounds[:-1]):
                # End index of the region in matrix that belongs to this RSN
                nextb1 = bounds[comm1+1]
                nextb2 = bounds[comm2+1]
                # Sum the values in this region
                num_edges = np.sum(A[b1:nextb1,b2:nextb2])
                # Divide by 2 if same community
                if comm1 == comm2:
                    num_edges /= 2
                # Divide by number of nodes in community 1 * nodes 
                # ... in community 2
                p = num_edges / float(sizes[comm1] * sizes[comm2])
                # Add to appropriate locations in probs
                probs[comm1,comm2] = p

        self.labels = labels
        
        return sizes, probs
    
    def __add_rsn_patches(self, ax, lenx, leny, boundaries, color_array, 
                          alpha_param=1.0, offset=1.0):
        """
        Plots rectangular patches vertically and horizontally for each
        ... network bounded by the indices provided using the colors provided
        """
        # Loop through boundaries
        for num,ind in enumerate(boundaries[:-1]):
            nextind = boundaries[num+1]
            hrect = patches.Rectangle(
                (ind,ind), nextind-ind, nextind-ind,
                edgecolor=color_array[num],facecolor='none',
                linewidth=1.5, alpha=alpha_param)
            ax.add_patch(hrect)
            vrect = patches.Rectangle(
                (ind,ind), nextind-ind, nextind-ind,
                edgecolor=color_array[num],facecolor='none',
                linewidth=1.5, alpha=alpha_param)
            ax.add_patch(vrect)
        
    def configure(self, sessions_, y=None, labels=None, **kwargs):
        """
        Calculate probabilities for each scan, use average probability to 
        ... create one SBM model
        """
        if labels is None:
            labels = np.ravel(RSN_LABELS)

        P = []

        for sess_num,session in enumerate(sessions_):
            print('Processing session %d out of %d...'
                  % ((sess_num+1), len(sessions_)))
            # Run Mapper
            y = session.y if y is None else y
            labels = y.columns

            result_ = run_mapper(
                session.X.T, **dict(dict(kwargs), y=y, verbose=0))
            # Get probs
            self.sizes,p = self.__calc_SBM_param(result_['dG'].G_)
            if p.shape[0] != len(np.unique(labels)):
                print('Threw out session %d' % sess_num)
                print('  *  p.shape:', p.shape)
                print('  *  n_labels:', len(np.unique(labels)))
                continue
            # Append so we can get the mean
            P.append(p)
        print(P)
        self.probs = np.mean(P,axis=0)
        
    def run(self,num_itera=30):
        if num_itera == 1:
            self.G_ = nx.stochastic_block_model(self.sizes,self.probs)
            self.A = nx.to_numpy_array(self.G_)
        else:
            # Run SBM
            Anull = [nx.to_numpy_array(
                        nx.stochastic_block_model(self.sizes,self.probs)) 
                     for i in range(num_itera)]
            Anull_mean = np.mean(Anull,axis=0)

            self.A = Anull_mean
            self.G_ = nx.stochastic_block_model(self.sizes,self.probs)
        
        # Get partition/communities - loop through every node and find 
        # ... its RSN community based on the index bounds for each RSN
        bounds = np.array(self.__rsn_index_change(self.labels)[1:])
        bounds[-1] += 1
        self.partition = {node: np.where(node < bounds)[0][0] 
                          for node in list(self.G_.nodes())}
            

    def plot(self, ax=None, figure=None, labels=None, add_cbar=True, show=True):

        if labels is None:
            labels = self.labels
        labels_set = np.unique(labels)

        rect_inds = self.__rsn_index_change(self.labels)

        if ax is None and figure is None:
            plt.close('all')
        plt.rc('font', family='serif')
        plt.imshow(self.A,cmap='binary_r')
        plt.axis('off')
        plt.title('ROI x ROI matrix, SBM')
        ax = plt.gca() if ax is None else ax
        fig = plt.gcf() if figure is None else figure
        cmap = mpl.cm.get_cmap('tab20', len(np.unique(labels)))
        colors = cmap(np.linspace(0,1,len(np.unique(labels))))

        self.__add_rsn_patches(
            ax, self.A.shape[1], self.A.shape[0], rect_inds, 
            colors, alpha_param=1.0, offset=2)

        # Create colorbar to label the RSN rectangles
        if add_cbar:
            # Use rect_inds to define which ticks are what colors
            # If our colormap is size of the matrix (554), then all indices 
            # ... within a RSN should be same color
            bar_colors = np.zeros((self.A.shape[0],4))
            # Holds the midpoint value of each RSN region to use for
            # ... tick labels
            midticks = []
            for num,ind in enumerate(rect_inds[:-1]):
                nextind = rect_inds[num+1]
                bclr = np.matlib.repmat(colors[num],nextind-ind,1)
                bar_colors[ind:nextind,:] = bclr
                midticks.append((nextind+ind)/2.0)
            # Flip since matrix goes top to bottom
            bar_colors = bar_colors[::-1]
            midticks = self.A.shape[0] - np.array(midticks)

            cmap_bar = mpl.colors.ListedColormap(bar_colors)
            norm = mpl.colors.Normalize(vmin=0,vmax=self.A.shape[0])
            sm = plt.cm.ScalarMappable(cmap=cmap_bar, norm=norm)
            sm.set_array([])

            # Left colorbar
            cbaxes = fig.add_axes([0.18, 0.11, 0.02, 0.77]) 
            cbar = plt.colorbar(sm, ticks=midticks, cax = cbaxes)
            cbaxes.yaxis.set_ticks_position('left')
            rsns_pretty = np.ravel(RSN_LABELS_PRETTY)
            cbar.ax.set_yticklabels(rsns_pretty[:len(np.unique(labels))])
            cbar.ax.tick_params(labelsize=7) 
        
        if show:
            plt.show()
        return fig, ax
        
    def set_params(self,G):
        # Get probs
        self.sizes,self.probs = self.__calc_SBM_param(G)
        
    def draw(self, ax=None, figure=None, layout='kamada_kawai'):
        if ax is None and figure is None:
            plt.close('all')
        
        # Generate colormap
        cmap = mpl.cm.get_cmap('tab20', 20)
        colors = cmap(np.linspace(0,1,20))

        #pos = nx.spring_layout(self.G_,scale=0.5)
        layout = layout if callable(layout) else getattr(nx, layout+'_layout')
        pos = layout(self.G_)
        for com in set(self.partition.values()) :
            list_nodes = [nodes for nodes in self.partition.keys() 
                          if self.partition[nodes] == com]
            nx.draw_networkx_nodes(
                self.G_, pos, list_nodes, 
                node_size=50, node_color=colors[com], 
                edgecolors='none', ax=ax)

        nx.draw_networkx_edges(self.G_, pos, alpha=0.5, ax=ax)
        ax = plt.gca() if ax is None else ax
        fig = plt.gcf() if figure is None else figure
        ax.axis('off')
        return fig, ax