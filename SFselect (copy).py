#!/usr/bin/env python

import cPickle as pickle
import sys

import numpy as np
#  SFselect.py
#
#  This program takes
#  (1) a trained model (SVM) of the site frequency spectrum.
#  (2) variant frequencies along a gemomic segment.
#  
#  It applies the given model to the frequency data and outputs class probabilities. Classes are
#  genmic segments evolving either neutrally, or under a 'hard sweep' model of positive selection.
#  The model is applied to a sliding window of constant size along the given segment. 

###############################################################################
############################ Parameter Defaults ###############################
###############################################################################

# defaults for internal parameters

# nbins       = 10     # single-population non-fixed bins
# nbins_case  = 7      # cross population non-fixed bins for control 
# nbins_cont  = 7      # cross population non-fixed bins for case
# GEN_SVM     = True   # internal indicator for general/specific
# XP_SFS      = False  # internal indicator for SP/XP
# bins        = []     # bins for scaled SFS
# 
# # optional parameters, overwritten from cmd
# 
# specific_s = None # selection  pressure  of specific SVM
# specific_t = None # time under selection of specific SVM  

###############################################################################
################################### GO ########################################
###############################################################################
import pandas as pd
def sfselect(sweep_pop_freqs, neut_pop_freqs=None, svm=None, svm_path='/home/arya/bin/sfselect/SVMs/',
             GENERAL=True, specific_s=None, specific_t=None, removeFixedSites=False):

    if svm is None: svm = read_pck_SVM(svm_path,GENERAL,specific_s=specific_s, specific_t= specific_t)
    sfs_v = bin_scale_norm(sweep_pop_freqs, neut_pop_freqs)
    if not len(sfs_v):  return pd.Series(( None, None), index=['score','prob'])
    if(GENERAL):
        w_prob, svm2prob = svm.predict_proba( np.array([sfs_v]) )
        score_w = -np.log(1 - w_prob[0,1])
        near_p, post_p = svm2prob[0][0], svm2prob[1][0]
        return pd.Series( (score_w, near_p, post_p), index=['score','nearp','postp'])
    else:
        w_prob          = svm.predict_proba( np.array([sfs_v]) )
        score_w = -np.log(1 - w_prob[0,1])
        prob = w_prob[0,1]
        return pd.Series(( score_w, prob), index=['score','prob'])

########################### read_pck_SVM ############################
def read_pck_SVM(svm_path,GENERAL, specific_s=None, specific_t=None):
    from SFSelect import metaSVM
    sys.modules['metaSVM']= metaSVM
    svm_file='{}{}_SVM{}_{}p.pck'.format(svm_path,('specific','general')[GENERAL],('s','')[GENERAL],('s','x')[specific_s is not None])
    with open(svm_file,mode='rb') as fh:
        svm = pickle.load(fh) # general SVM
        
        # if needed, overwrite with specific SVM
        if(isinstance(svm, dict)): 
            if(specific_s != None and specific_t != None):
                svm = svm[specific_s, specific_t]
            else:
                print "\n\t" + "Error::specific SVM input using general argument switch. Quitting...\n"
                sys.exit(1)
    return svm

######################## bin_scale_norm_xp ##########################
def bin_scale_norm(freqs_case, freqs_cont):
    ''' returns the binned, scales, and normalized XP-SFS
        note: in linearized form 
    '''
    x,y =freqs_case[(freqs_case!=0) & (freqs_case!=1)],None
    if freqs_cont is not None:
        xx= freqs_case + freqs_cont
        i=(xx!=0) & (xx!=2)
        x, y=pd.Series(freqs_case[i]).to_dict(), pd.Series(freqs_case[i]).to_dict()


    nbins,nbins_cont,nbins_case=10,7,7
    bins_case  = lrange( 0.0, 1.0, 1.0/float(nbins) )
    bins_cont = lrange( 0.0, 1.0, 1.0/float(nbins_cont) )
    if y is not None:
        bins_case = lrange( 0.0, 1.0, 1.0/float(nbins_case) )
    if y is None:
        v, binEdges = np.histogram(x, bins_case)
        centers = 0.5*(binEdges[1:]+binEdges[:-1])
        v = v*centers

        return v / np.linalg.norm(v)
    else:
        union_case, union_cont =  union_snp_freq_dicts(x, y)
        v, x_edges, y_edges = np.histogram2d(union_case.values(), union_cont.values(), np.array([bins_case, bins_cont]) )

        # bin centers
        x_centers = 0.5*(x_edges[1:]+x_edges[:-1])
        y_centers = 0.5*(y_edges[1:]+y_edges[:-1])

        # scaling
        xpscale = np.outer(x_centers, y_centers.T)
        v = v*xpscale

        lin_v = v.reshape(-1)

        return lin_v / np.linalg.norm(lin_v)

####################### union_snp_freq_dicts ########################
def union_snp_freq_dicts(freq_sweep, freq_neut):
    ''' unite dict keys to be the union of sweep & neutral '''
    union_sweep, union_neut = {}, {}
    
    for pos,f_case in freq_sweep.iteritems():
        # add all case frequencies to sweep-union
        union_sweep[pos] = f_case
        
        # add case-specific frequencies to neutral-union
        if(not pos in freq_neut):
            union_neut[pos] = 0.0
        
    for pos,f_cont in freq_neut.iteritems():
        # add all cont frequencies to neutral-union 
        union_neut[pos] = f_cont
        
        # add cont-specific frequencies to sweep-union
        if(not pos in freq_sweep):
            union_sweep[pos] = 0.0
    
    # sanity check
    assert (len(union_sweep.keys()) == len(union_neut.keys())), "Error::union_snp_freq_dicts::key union error"
    return union_sweep, union_neut

#####################################################################
def remove_xp_fixed(freqs1, freqs2):
    ''' removes variants fixed in both sweep & neutral populations
        returns number of variants removed
    '''
    removed_c = 0
    for pos in freqs1.keys():
        if(freqs1[pos] == 1.0 and pos in freqs2 and freqs2[pos] == 1.0):
            del(freqs1[pos])
            del(freqs2[pos])
            removed_c += 1
    
    return removed_c
        

#####################################################################
def lrange(start, stop, step):
    ''' returns bin edges including a last bin exclusive to fixed variants '''
    
    # basic bins in [0,1]
    l = list( np.arange(start, stop, step) )
        
    # special bin for fixed variants
    l.append(np.nextafter(1,0)) 
    
    l.append(stop)
    
    return l
