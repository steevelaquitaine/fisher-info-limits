"""MODULE WITH RELEVANT PROCEDURES: NOISE MODELS, SSI CALCULATIONS FROM TCS, FIT CURVES, ...
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os
import math
import scipy
from itertools import product
from scipy.signal import find_peaks
from scipy.signal import argrelextrema as arex
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.sparse as sp
from scipy.interpolate import UnivariateSpline
import urllib
import json
import os
import ipykernel
from shutil import copy2
import sys

from scipy.stats import poisson as poi

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def SUM_LOG_LIST(position):
    '''Given an integer n it recursively calculates log(n!)'''
    if position == 0:
        return np.array([0])
    if position == 1:
        return np.append(SUM_LOG_LIST(0), 0)
    new_list = SUM_LOG_LIST(position-1)
    return np.append(new_list, new_list[-1]+np.around(np.log(float(position)), 8))


def ADD_ARRAY_TO_NPZ(dat_dir, filename, new_array_name, new_array):
    '''Adds an array to an existing .npz file as dictionary entry. \n Input variables: directory of file, filename, new array name, new array'''

    dictionary = dict(np.load(dat_dir+filename))
    dictionary[new_array_name] = new_array
    np.savez(dat_dir + filename, **dictionary)
    return


def GET_SSI_LOOP(all_tcs):
    '''Input a list of TCs, \n Output a list of associated SSIs. Output in NATS.'''
    aux = np.array([GET_SSI_FROM_TC(el) for el in all_tcs])
    return aux
    

def GET_SSI_FROM_TC(input_tc_):           #!!! NB: ASSUME UNIFORM PRIOR !!!
    '''Input 1 Tuning Curve, Outputs Associated SSI, supposing Poisson noise model. Outpu in NATS.'''
    single_tc = np.maximum(np.squeeze(input_tc_), 1e-10)
    n_th = len(single_tc)
    theta_entropy_ = np.full(n_th, np.log(n_th))
    log_likelihood__, likelihood__ = POISSON(single_tc)
    normalization_ = np.sum(likelihood__, axis = 1)
    entropy_theta_given_m_ = np.log(normalization_) - np.sum(likelihood__*log_likelihood__, axis = 1)/normalization_
    ssi = theta_entropy_ - np.sum(likelihood__*entropy_theta_given_m_[:,None], axis = 0)

    return ssi#, log_likelihood__, likelihood__, normalization_, mask__


def POISSON(input_tc_):     # lambda has to be poissonian, dependent on tc(sigma)
    tc_ = np.squeeze(input_tc_)
    n_th = len(tc_)
    lambdas_ = np.copy(tc_)
    log_lambdas_ = np.log(np.maximum(lambdas_, 1e-10))
    
    m_max_ = np.around(lambdas_+6*np.sqrt(lambdas_)+1)       #dims = dim(stim) = n_th = 36
    
    peak_resp = np.max(m_max_)
    response_array_ = np.arange(peak_resp+1)            #dims = highest response integer
    response_matrix__ = np.tile(response_array_[:,None], (1,n_th))
#     print('r',response_matrix__)
    mask_condition__ = np.tile(m_max_[None,:], (len(response_array_),1))
#     print('m',mask_condition__.shape)
    mask__ = response_matrix__ <= mask_condition__
#     mask__ = np.ones_like(response_matrix__)

    #calculate the sums of the logarithms of the reponses 0,1,2,3,4,...,m  -> [0,0,log2,log3,log4,...,logm]
    somme_log_ = SUM_LOG_LIST(peak_resp)
    log_likelihood__ = response_matrix__*log_lambdas_[None,:] - lambdas_[None,:] - somme_log_[:,None]
    likelihood__ = np.exp(log_likelihood__)
    likelihood__ = likelihood__ / np.sum(likelihood__, axis = 0)[None,:]
    log_likelihood = np.log(np.maximum(likelihood__, 1e-10))
    
    log_likelihood__[mask__== False]= np.min(log_likelihood__)*10e5
    likelihood__[mask__== False] = 0.
    
    return log_likelihood__, likelihood__        #return log_likelihood and likelihood


def VON_MISES(th, a, th_0, s, baseline):           #indexed with 'ftvm'
    f = lambda stimulus, amp, x_peak, width: \
            amp* np.exp((np.cos((stimulus-x_peak)*(np.pi/180)))/width)
    vm_f = f(th, a, th_0, s)
    return (vm_f / np.sum(vm_f/a) + baseline)*(1 +any(vm_f / np.sum(vm_f/a) + baseline < 0)*1e10)

def von_mises_tuning(theta, amplitude, preferred_angle, kappa, baseline=0):
    """
    Von Mises tuning function for circular data
    
    Args:
        theta: stimulus angles (radians)
        amplitude: peak response
        preferred_angle: preferred angle (radians) 
        kappa: concentration (larger = sharper tuning)
        baseline: baseline firing rate
    """
    # Von Mises: exp(kappa * cos(theta - mu))
    response = amplitude * np.exp(kappa * np.cos(theta - preferred_angle))
    return np.maximum(response + baseline, 0)  # Ensure non-negative

def FLAT_TOPPED_VON_MISES(th, a, th_0, s, baseline, g):           #indexed with 'ftvm'
    '''Modified Von Mises function\n Parameters Area, mean, concentration, baseline, flatness\n g = 0 gives Von Mises function'''
    delta_th = th[1]-th[0]
    ftvm_f = np.exp(np.float64((np.cos(np.deg2rad(th-th_0-g*np.sin(np.deg2rad(th-th_0))))-1)/s))
    
    res = (a*ftvm_f / np.sum(ftvm_f/(len(th)/36)) + baseline) 
    return res+np.any(res<0.)*1e10


def FTVM(th, a, th_0, s, baseline, g):           #indexed with 'ftvm'
    return FLAT_TOPPED_VON_MISES(th, a, th_0, s, baseline, g)


def BIG_SSI(tcs):  
    '''Takes an array of 1,2,3 or 4 tuning curves and returns the combined SSI\n
    SUPPOSE NO CORRELATIONS -> p(n1,n2,...,nm|th) = p(n1|th)*p(n2|th)*...*p(nm|th)\n
    Output in BITS.'''
    if tcs.ndim==1:
#         print('s',np.squeeze(tcs))
        return GET_SSI_FROM_TC(np.squeeze(tcs))

    log_likelihoods = []
    likelihoods = []
    resp_dims = []
    output_shape = []
    for i,el in enumerate(tcs):
        new_log_likelihood__, new_likelihood__ = POISSON(el)
#         log_likelihoods.append(new_log_likelihood__) 
        likelihoods.append(new_likelihood__) 
        output_shape.append(len(likelihoods[i]))
    output_shape.append(len(tcs[0]))
#     print('os',output_shape, end = '\r')
    
#    print('sparse: ',np.count_nonzero(likelihoods)/np.prod(likelihoods.shape) > 2./3)
#    print('count1',np.count_nonzero(log_likelihoods>0))
#    print('count2',np.count_nonzero(likelihoods<0))
#    print('count3',np.count_nonzero(likelihoods>1))#

    if len(tcs)==2:
        multidim_likelihood = np.float32(np.einsum('im,jm->ijm', *likelihoods))          #p(n1|th)*p(n2|th)
        multidim_norm = np.float32(np.sum(multidim_likelihood, axis=-1))
        multidim_norm[multidim_norm==0.] = 1e-20  # CAREFUL !! IF NORM==0 THEN EVERYTHING EXPLODES -> SSI = nan
        likelihood_entropy = np.float32(-1.*np.sum(multidim_likelihood * np.log(np.maximum(multidim_likelihood, 1e-10)), axis=-1))       # this step takes a long time !!!
        multidim_entropy = np.float32((likelihood_entropy/multidim_norm) + np.log(multidim_norm))
        ssi_theta = np.float32(np.log(len(tcs[0]))) - np.float32(np.sum(multidim_likelihood*multidim_entropy[:,:,None], axis = (0,1)))
        

    elif len(tcs)==3:
        multidim_likelihood = np.array(likelihoods[0])[:, np.newaxis, np.newaxis, np.newaxis, :] \
                            * np.array(likelihoods[1])[np.newaxis, :, np.newaxis, np.newaxis, :] \
                            * np.array(likelihoods[2])[np.newaxis,np.newaxis,:, np.newaxis, :]
        multidim_norm = np.sum(multidim_likelihood, axis=-1)
        multidim_norm[multidim_norm == 0.] = 1e-20
        likelihood_entropy = -1. * np.sum(multidim_likelihood * np.log(np.maximum(multidim_likelihood, 1e-10)), axis=-1)
        multidim_entropy = (likelihood_entropy / multidim_norm) + np.log(multidim_norm)
        del likelihood_entropy
        del multidim_norm
        ssi_theta = np.log(np.array(likelihoods).shape[-1]) - np.sum(multidim_likelihood * multidim_entropy[:, :, :, None], axis=(0, 1, 2))
        del multidim_likelihood, multidim_entropy
         
    elif len(tcs) == 4:
        
        multidim_likelihood = np.array(likelihoods[0])[:, np.newaxis, np.newaxis, np.newaxis, :] \
                            * np.array(likelihoods[1])[np.newaxis, :, np.newaxis, np.newaxis, :] \
                            * np.array(likelihoods[2])[np.newaxis,np.newaxis,:, np.newaxis, :]   \
                            * np.array(likelihoods[3])[np.newaxis, np.newaxis, np.newaxis, :, :]
#         print('4', end = '\033[K\r')
        multidim_norm = np.sum(multidim_likelihood, axis=-1)
        multidim_norm[multidim_norm == 0.] = 1
#         print('3', end = '\033[K\r')

        likelihood_entropy = -1. * np.sum(multidim_likelihood * np.log(np.maximum(multidim_likelihood, 1e-30)), axis=-1)
#         print('2', end = '\033[K\r')
        multidim_entropy = (likelihood_entropy / multidim_norm) + np.log(np.maximum(multidim_norm, 1e-30))
#         print('1', end = '\033[K\r')
        del likelihood_entropy
        del multidim_norm
        ssi_theta = np.log(tcs.shape[-1]) - np.sum(multidim_likelihood * multidim_entropy[:, :, :, :, None], axis=(0, 1, 2, 3))
#         print('Done', end = '\033[K\r')
        del multidim_likelihood
        del multidim_entropy
    del new_log_likelihood__
    del new_likelihood__
    del log_likelihoods
    del output_shape
    
    return np.squeeze(ssi_theta)


def FISHER_POISSON(stim, tcs):
    lik_2d_spl = np.array([UnivariateSpline(stim, el, s = 0,k = 2).derivative(n=1)(stim) for el in tcs])
    fisher = np.sum(np.array([lik_2d_spl[i]**2./tcs[i] for i in range(len(tcs))]),  axis =0)
    return fisher


def MEAN_RES_LENGTH(stim, tcs, means, p):   #population resultant length            0 < MRL < 1
    """Input: stim, tcs, means, p,  Calculates the pth moment of the tcs \n p=1 gives population resultant length \n np.sum(tc) serves as normalization \n input means are the means of the fitted ftvm"""
    if tcs.ndim==1:
        res = np.sum(tcs*np.cos(p*np.deg2rad(stim-means)))/np.sum(tcs)
    else:
        if np.isscalar(means):
            res = np.array([np.sum(tc*np.cos(p*np.deg2rad(stim-means)))/np.sum(tc) for i, tc in enumerate(tcs)])   
        else:
            res = np.array([np.sum(tc*np.cos(p*np.deg2rad(stim-means[i])))/np.sum(tc) for i, tc in enumerate(tcs)])
    return res


def CSTD(stim, tcs, means):    #calculated the circular STD of a (flat-topped) vonMises distribution
                               #contained between [0, infinity] for r in [1,0]=[completely centered,completely dispersed]                     
    """ Calculates the circular STD \n It uses the POP_RES_LENGTH internally """
#     b0 = np.sum(tc, axis = 1)                                #zeroth moment of tc
    ro1 = MEAN_RES_LENGTH(stim, tcs, means, 1)                  #first moment of tc 
    cstd_rad = np.sqrt(-2*np.log(ro1))           #circular std in radians
    return np.rad2deg(cstd_rad)                   #circular std in degrees  


def SQRT_CVAR(stim, tcs, means):
    return np.rad2deg(np.sqrt((1.-MEAN_RES_LENGTH(stim, tcs, means, 1)))*np.pi)     #between [0,180]


def CDISP(stim, tcs, means):
    ro = MEAN_RES_LENGTH(stim, tcs, means, 1)
    ro2 = MEAN_RES_LENGTH(stim, tcs, means, 2)
    return np.rad2deg((1.-ro2)/(2*ro**2))                #contained between [0, infinity]


def ARC_DISP(stim, tcs, means):
    '''Input: stim, tcs, means\n gives a measure of the arcs subtracted by the tcs (circular distributions on [0,2pi])'''
    r = MEAN_RES_LENGTH(stim, tcs, means, 1)
    if np.isscalar(r):
        return 2*np.rad2deg(np.arccos(r))
    else:
        arcs = 2*np.array([np.rad2deg(np.arccos(el)) for el in r] )
        return 2*arcs #contained in [0,360]
    

def MC_SSI(tcs_input):         #dim tcs = 4x36
    ''' Calculates SSI by Monte Carlo method. \n Up to 2023-11-12 gives result in nats, from 2023-11-12 (included) gives result in bits'''
    #if input is multiple cells
#     K = 10000
    
    if tcs_input.ndim==1:
        tcs = np.atleast_2d(tcs_input)
    else:
        tcs = np.copy(tcs_input)
    posterior = np.array([])
    stim_len = tcs.shape[-1]
    K = 100
    err = 1e4
    epsilon = 1        #errors in percents

    previous_ssi = np.zeros(stim_len)
    if np.max(tcs_input)<20:
        return BIG_SSI(tcs_input)*np.log2(math.e)
    
    while (K<10000 and err>epsilon):
        #get your K samples -> to be summed over at the end

        sample_r = np.random.poisson(tcs, size = (K, *tcs.shape))     #dim = Kx4x36 for quadruplet
    #     norm = np.sum(np.prod(np.maximum(poi.pmf(np.tile(sample_r[:,:,:,None], (1,1,1,stim_len)), \
    #                           np.tile(tcs[None,:,None, :], (K,1, stim_len,1))), 1e-10), axis = 1), axis = -1)
        post_aux = np.maximum(1e-40, \
                              np.prod(np.maximum(poi.pmf(np.tile(sample_r[:,:,:,None], (1,1,1,stim_len)), \
                              np.tile(tcs[None,:,None, :], (K,1, stim_len,1))), 1e-100), axis = 1))
        posterior = post_aux / np.tile(np.sum(post_aux, axis = -1)[:,:,None], (1,1,stim_len))


        i_sp = np.log(stim_len)+np.sum(posterior*np.maximum(np.log(posterior),-50), axis = 2)
        ssi = np.mean(i_sp, axis = 0)
        del sample_r, post_aux, posterior
        err = np.sqrt(np.mean((ssi-previous_ssi)**2))/np.mean(ssi)*100
        previous_ssi = np.copy(ssi)
        K *=2
        
#     print(K)
    del previous_ssi
    

    # if the number of samples has gotten bigger than 5000 and the error is still larger than epsilon
    # then we're probably in the high noise regime, and it's best to compute the SSI exactly - use BIG_SSI     
    #However if the TCs have a very high maximum average firing rate, then do not compute exactly in any case.

    if K>=10000 and err>epsilon and np.max(tcs_input)<30:
        ssi = BIG_SSI(tcs_input)
    return ssi*np.log2(math.e)


def FWHM(pars):
    
    """Return the Full Width at Half Maximum of a FTVM function.
       The TC is obtained by subtracting the baseline and dividing by the area.
       In other words, the measure is area and baseline independent.
        NOTE: baseline parameter must be non-negative
        """
    x = np.arange(pars[1]-180,pars[1]+180,0.5)
    ftvm = FLAT_TOPPED_VON_MISES(x, *pars)

    fcn = ftvm - np.max(ftvm)/2
#     print(np.max(ftvm)/2)
#     plt.figure()
#     plt.plot(x, fcn)
    
    if np.all(fcn>0):
        return np.max(ftvm)/2, 360
    
    idx = np.argmin(fcn**2)
#     print(ftvm[idx])
    return ftvm[idx], 2*np.abs(x[idx]-pars[1])%360