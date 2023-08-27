# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 07:54:51 2021

@author: Robert James
"""
import numpy as np
import pandas as pd
import math

from numba import jit
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.optimize import minimize

#MLE
@jit(nopython = True)
def MLE_estimator(params, data): 
    sum_component = np.log(np.where(1 + params[0]*data/params[1] > 0, 1 + params[0]*data/params[1], 1e-6))
    ll = -(-len(data)*np.log(params[1]) - (1 + (1/params[0]))*np.sum(sum_component))
    return ll

#@jit(nopython = True)
def LMoments(data):
    try:
        n = len(data)
        #First L-Moment
        b_0 = np.mean(data)
        L1 = b_0            
        #Second L-Moment
        to_sum = []
        for j in range(1, n):
            to_sum.append(data[j]*((j+1-1)/(n+1-1)))
        beta_1 = np.mean(np.array(to_sum))
        L2 = 2*beta_1 - b_0
    except:
        L1 = np.nan
        L2 = np.nan
    return L1,L2

#@jit(nopython = True)
def PWM_initial_parameters(excess_loss, sample_mean_excess_loss):
    gamma = -0.35
    delta = 0
    pvect = (np.array([x for x in range(1, len(excess_loss)+1)]) + gamma)/ (len(excess_loss) + delta)
    a1 = np.mean(np.sort(excess_loss) * (1-pvect))
    scale = (2 * sample_mean_excess_loss * a1)/(sample_mean_excess_loss - 2 * a1)
    shape = 2 - sample_mean_excess_loss/(sample_mean_excess_loss-2*a1)
    return shape, scale

#WNLS
@jit(nopython = True)
def weights(nu, n):
    weights = []
    variances = []
    denominator = ((n+1)**2)*(n+2)
    variances = [(i*(n - i+1))/denominator for i in range(1, nu+1, 1)]
    weights = [1/((i*(n - i+1))/denominator) for i in range(1, nu+1, 1)]
    return variances, weights

def ECDF_data_ParkKim2016(test_threshold, random_sample, tail_sample):   
    n = len(random_sample)
    ECDF_fit = ECDF(random_sample)
    ECDF_array = np.column_stack([ECDF_fit.x, ECDF_fit.y])[1:,:]
    ECDF_component = ECDF_array[:,1][np.isin(ECDF_array[:,0], tail_sample)]
    
    ECDF_component_11 = np.log(1 - ECDF_component+(1/(10*n)))
    ECDF_component_12 = np.log(1 - (test_threshold/100))
    ECDF_component_1 = ECDF_component_11 - ECDF_component_12
        
    ECDF_component_2 = (ECDF_component - (test_threshold/100))/(1 - (test_threshold/100))
    return ECDF_component_1, ECDF_component_2

@jit(nopython = True)
def stage_1_POT_NLS_estimator_ParkKim2016(params, X, y):
    to_log = 1 + ((params[0]*X)/params[1])
    to_log = np.where(to_log > 0, to_log, 1e-10)
    y_hat = (-1/params[0]) * (np.log(to_log))
    resids = y - y_hat
    return (1/len(X))*(np.sum(np.square(resids)))

@jit(nopython = True)
def stage_2_POT_WNLS_estimator_ParkKim2016(params, X, y, weights_vector):
    y_hat = 1 - (1 + (params[0]*X)/(params[1]))**(-1/params[0])
    resids = y - y_hat
    return (1/len(X))*(np.sum(weights_vector*(np.square(resids))))
    
#@jit(nopython = True)
def revised_NEW_lx(b, x):
    k = -np.mean(np.log(1-b*x))
    if b == 0:
        return k - 1 - np.log(np.mean(x))
    else:
        return k - 1 + np.log(b/k)

#@jit(nopython = True)
def revised_NEW(sorted_excess_loss_sample):
    try:
        _n_ = len(sorted_excess_loss_sample)
        p = np.array(list(range(3,10)))/10
        xp = sorted_excess_loss_sample[(np.round(_n_*(1-p)+0.5).astype(int))-1]
        m = int(20 + np.round(_n_**0.5))
        xq = sorted_excess_loss_sample[(np.round(_n_*(1-p*p)+0.5).astype(int))-1]
        k = np.log(xq/xp - 1)/np.log(p)
        a = k*xp/(1-p**k)
        a[k==0] = (-xp/np.log(p))[k==0] 
        k = -1
        b = (_n_-1)/(_n_+1)/sorted_excess_loss_sample[-1]-(1-((np.array(list(range(1, m+1)))-.5)/m)**k)/k/np.median(a)/2
        w = (_n_-1)/(_n_+1)/sorted_excess_loss_sample[-1]-(1-((np.array(list(range(1, m+1)))-.5)/m)**k)/k/np.median(a)/2
        L = (_n_-1)/(_n_+1)/sorted_excess_loss_sample[-1]-(1-((np.array(list(range(1, m+1)))-.5)/m)**k)/k/np.median(a)/2
        for i in range(m):
            L[i] = _n_*revised_NEW_lx(b[i], sorted_excess_loss_sample)
        for i in range(m):
            w[i] = 1/np.sum(np.exp(L-L[i]))
        b = np.sum(b*w)
        k = -np.mean(np.log(1-b*sorted_excess_loss_sample))
        zhang_scale = k/b
        zhang_shape = k*-1
    except:
        zhang_shape = np.nan
        zhang_scale = np.nan
    return zhang_shape, zhang_scale


@jit(nopython = True)
def NEW_lx(b, x):
    k = -np.mean(np.log(1-b*x))
    return np.log(b/k) + k - 1

@jit(nopython = True)
def NEW(sorted_excess_loss_sample):
    _n_ = len(sorted_excess_loss_sample)
    m = 20 + math.floor(np.sqrt(_n_))
    b = 1/sorted_excess_loss_sample[-1] + (1 - np.sqrt(m/np.arange(0.5,m+1-0.5)))/3/(sorted_excess_loss_sample[math.floor(_n_/4+0.5)] + (1/(10*_n_)))
    w = np.zeros(len(b))
    L = np.zeros(len(b))
    for i in range(m):
        L[i] = _n_*NEW_lx(b[i], sorted_excess_loss_sample)
    for j in range(m):
        w[j] = 1/(np.sum(np.exp(L - L[j])))
    b = np.sum(b*w)
    shape = -np.mean(np.log(1-b*sorted_excess_loss_sample))    
    zhang_scale = shape/b
    zhang_shape = shape *-1
    return zhang_shape, zhang_scale

#LME
#@jit(nopython = True)
def LME_estimation(x, xn, b, r):
    try:
        for i in range(1,100):
            B = 1 - b*x
            K = np.log(1 - b*x)
            k = -np.mean(K)
            gb = np.mean(B**(-r/k)) - 1/(1-r)
            gd = np.mean(B**(-r/k) * (x/B*k + K*np.mean(x/B)))*r/k/k
            b = np.min([b-gb/gd, (1-1/2**i)/xn])
            if np.abs(gb/gd/b) < 1e-6:
                break
    #        else:
    #            r = copy.deepcopy(k)
        LME_sigma = k/b
        LME_shape = k*-1
    except:
        LME_shape = np.nan
        LME_sigma = np.nan
    return LME_shape, LME_sigma

#Maximum Goodness of Fit estimator
def MGF_AD_objective(params, sorted_excess_loss):
    n = len(sorted_excess_loss)
    theop = np.sort(stats.genpareto.cdf(sorted_excess_loss, c = params[0], scale = params[1]))
    AD2 = - n - np.mean( (2 * np.arange(1,n+1) - 1) * (np.log(theop) + np.log(1 - theop[::-1])))
    return AD2

#Maximum Goodness of Fit estimator
def MGF_AD2R_objective(params, sorted_excess_loss):
    n = len(sorted_excess_loss)
    theop = np.sort(stats.genpareto.cdf(sorted_excess_loss, c = params[0], scale = params[1]))
    theop = np.where(theop > 0.9999, 0.9999, theop) #For numerical stability
    AD2R = 2 * np.sum(np.log(1 - theop)) + np.mean ((2 * np.arange(1,n+1) - 1) / (1 - theop[::-1]))
    return AD2R

def MGF_ADR_objective(params, sorted_excess_loss):
    n = len(sorted_excess_loss)
    theop = np.sort(stats.genpareto.cdf(sorted_excess_loss, c = params[0], scale = params[1]))
    theop = np.where(theop > 0.9999, 0.9999, theop) #For numerical stability
    ADR = n/2 - 2 * np.sum(theop) - np.mean((2 * np.arange(1,n+1) - 1) * np.log(1 - theop[::-1]))
    return ADR

def Tukey_biweight(u, c):
    if (np.abs(u) > c):
        return (c**2)/6
    else:
        return (u**2)/2*(1-(u**2)/((c**2) + (u**4)/3/(c**4)))
    
def Mestimation_objective(params, data):
    n = len(data)
    if params[1] > 0:
        u = (np.arange(1,n+1)-0.5)/n - stats.genpareto.cdf(data, c = params[0], scale = params[1])
        return np.sum(np.array([Tukey_biweight(ui, c=4.6851) for ui in u]))
    else:
        return 1e10 #extended value extension to ensure scale is positive

def weighted_Mestimation_objective(params, data):
    n = len(data)
    if params[1] > 0:
        true_cdf = stats.genpareto.cdf(data, c = params[0], scale = params[1])
        u = ((np.arange(1,n+1)-0.5)/n - true_cdf)/(true_cdf*(1-true_cdf))**0.5
        return np.sum(np.array([Tukey_biweight(ui, c=4.6851) for ui in u]))
    else:
        return 1e10 #extended value extension to ensure scale is positive

def initial_params(excess_loss, initial_param_type):
    excess_loss = np.sort(excess_loss)
    if initial_param_type == None:
        initial_parameters = {x:{} for x in [1,2,3]}
    else:
        initial_parameters = {x:{} for x in initial_param_type}
    m = np.mean(excess_loss)
    v = np.var(excess_loss)
    
    if initial_param_type == None:
        #Type 1: PWM
        pwm_initial_params = PWM_initial_parameters(excess_loss = excess_loss, sample_mean_excess_loss = m)
        initial_parameters[1]['scale'] = pwm_initial_params[1]
        initial_parameters[1]['shape'] = pwm_initial_params[0]
    
        #Type2: Sample moments estimators
        initial_parameters[2]['scale'] = np.sqrt(v)
        initial_parameters[2]['shape'] = 1e-6
        
        #Type3: L-Moments 
        L1,L2 = LMoments(data = excess_loss)
        tau = L2/L1
        initial_parameters[3]['scale'] = L1 * (1/tau-1)
        initial_parameters[3]['shape'] = -(1/tau-2)
    else:
        if 1 in initial_param_type:
            #Type 1: PWM
            pwm_initial_params = PWM_initial_parameters(excess_loss = excess_loss, sample_mean_excess_loss = m)
            initial_parameters[1]['scale'] = pwm_initial_params[1]
            initial_parameters[1]['shape'] = pwm_initial_params[0]
        if 2 in initial_param_type:
            #Type2: Sample moments estimators
            initial_parameters[2]['scale'] = np.sqrt(v)
            initial_parameters[2]['shape'] = 1e-6
        if 3 in initial_param_type:
            #Type3: L-Moments 
            L1,L2 = LMoments(data = excess_loss)
            tau = L2/L1
            initial_parameters[3]['scale'] = L1 * (1/tau-1)
            initial_parameters[3]['shape'] = -(1/tau-2)   
    return initial_parameters

def parameter_estimation(parameter_estimation_dict, full_sample, sorted_excess_loss, excess_loss, tail_sample, initial_threshold, weights_vector, models, optimization_bounds):
    #L-Moments 
    if 'L-Moments' in models:
        L1,L2 = LMoments(data = sorted_excess_loss)
        if (np.isnan(L1) or np.isinf(L1)) or (np.isnan(L2) or np.isinf(L2)):
            parameter_estimation_dict['L-Moments']['scale'] = np.nan
            parameter_estimation_dict['L-Moments']['shape'] = np.nan
        else:
            tau = L2/L1
            parameter_estimation_dict['L-Moments']['scale'] = L1 * (1/tau-1)
            parameter_estimation_dict['L-Moments']['shape'] = -(1/tau-2)
    
    #PWM
    if 'PWM' in models: 
        PWM_initial_params = PWM_initial_parameters(excess_loss = sorted_excess_loss, sample_mean_excess_loss = np.mean(sorted_excess_loss))        
        parameter_estimation_dict['PWM']['scale'] = PWM_initial_params[0]
        parameter_estimation_dict['PWM']['shape'] = PWM_initial_params[1]
    
    #MLE 
    if 'MLE' in models:
        MLE_estimate_results = []
        MLE_initial_values = initial_params(excess_loss = excess_loss.ravel(), initial_param_type = [1,3])
        for initial_values in MLE_initial_values.values():
            try:
                MLE_estimate_results.append(minimize(fun = MLE_estimator, x0 = [initial_values['shape'], initial_values['scale']], method = 'nelder-mead', args = (excess_loss), options = {'maxiter':10000}))
            except:
                pass
            try:
                MLE_estimate_results.append(minimize(fun = MLE_estimator, x0 = [initial_values['shape'], initial_values['scale']], method = 'L-BFGS-B', bounds = optimization_bounds, args = (excess_loss), options = {'maxiter':10000, 'maxfun':100000, 'maxcor':500}))
            except:
                pass
        if MLE_estimate_results == [] or np.all([x.success == False for x in MLE_estimate_results]):
            best_model_index = np.argmin([x.fun for x in MLE_estimate_results])
            parameter_estimation_dict['MLE']['scale'] = np.nan
            parameter_estimation_dict['MLE']['shape'] = np.nan
        else:            
            best_model_index = np.argmin([x.fun for x in MLE_estimate_results if not np.isnan(x.fun)])
            parameter_estimation_dict['MLE']['scale'] = MLE_estimate_results[best_model_index]['x'][1]
            parameter_estimation_dict['MLE']['shape'] = MLE_estimate_results[best_model_index]['x'][0]
    
    #LME (Zhang 2009)
    if 'LME' in models:
        x = sorted_excess_loss
        b = -1
        xn = np.max(x)
        initial_r = parameter_estimation_dict['L-Moments']['shape']
        LME_shape, LME_scale = LME_estimation(x = x, xn = xn, b = b, r = initial_r)
        if (np.isnan(LME_shape) or np.isinf(LME_shape)) or (np.isnan(LME_scale) or np.isinf(LME_scale)):
            parameter_estimation_dict['LME']['scale'] = np.nan
            parameter_estimation_dict['LME']['shape'] = np.nan
        else:
            parameter_estimation_dict['LME']['scale'] = LME_scale
            parameter_estimation_dict['LME']['shape'] = LME_shape     
            
    #Zhang 2010 NEW 
    if 'Revised_NEW' in models:
        revised_NEW_shape, revised_NEW_scale = revised_NEW(sorted_excess_loss_sample = sorted_excess_loss)
        if (np.isnan(revised_NEW_shape) or np.isinf(revised_NEW_shape)) or (np.isnan(revised_NEW_scale) or np.isinf(revised_NEW_scale)):
            parameter_estimation_dict['Revised_NEW']['scale'] = np.nan
            parameter_estimation_dict['Revised_NEW']['shape'] = np.nan
        else:
            parameter_estimation_dict['Revised_NEW']['scale'] = revised_NEW_scale
            parameter_estimation_dict['Revised_NEW']['shape'] = revised_NEW_shape   
            
    #Zhang 2009 NEW 
    if 'NEW' in models:
        NEW_shape, NEW_scale = NEW(sorted_excess_loss_sample = sorted_excess_loss)
        if (np.isnan(NEW_shape) or np.isinf(NEW_shape)) or (np.isnan(NEW_scale) or np.isinf(NEW_scale)):
            parameter_estimation_dict['NEW']['scale'] = np.nan
            parameter_estimation_dict['NEW']['shape'] = np.nan
        else:
            parameter_estimation_dict['NEW']['scale'] = NEW_scale
            parameter_estimation_dict['NEW']['shape'] = NEW_shape  
            
    #Maximum Goodness of Fit
    if 'MGF_AD2R' in models:
        MGF_estimate_results = []
        for initial_values in MLE_initial_values.values():
            try:
                MGF_estimate_results.append(minimize(fun = MGF_AD2R_objective, x0 = [initial_values['shape'], initial_values['scale']], method = 'nelder-mead', args = (sorted_excess_loss), options = {'maxiter':10000}))
            except:
                pass
            try:
                MGF_estimate_results.append(minimize(fun = MGF_AD2R_objective, x0 = [initial_values['shape'], initial_values['scale']], method = 'L-BFGS-B', bounds = optimization_bounds, args = (sorted_excess_loss), options = {'maxiter':10000, 'maxfun':100000, 'maxcor':500}))
            except:
                pass
        if MGF_estimate_results == [] or np.all([x.success == False for x in MGF_estimate_results]):
            parameter_estimation_dict['MGF_AD2R']['scale'] = np.nan
            parameter_estimation_dict['MGF_AD2R']['shape'] = np.nan
        else:               
            best_model_index = np.argmin([x.fun for x in MGF_estimate_results if not np.isnan(x.fun)])
            MGF_shape = MGF_estimate_results[best_model_index]['x'][0]
            MGF_scale = MGF_estimate_results[best_model_index]['x'][1] 
            parameter_estimation_dict['MGF_AD2R']['scale'] = MGF_scale
            parameter_estimation_dict['MGF_AD2R']['shape'] = MGF_shape
        
    #Maximum Goodness of Fit
    if 'MGF_AD' in models:
        MGF_AD_estimate_results = []
        for initial_values in MLE_initial_values.values():
            try:
                MGF_AD_estimate_results.append(minimize(fun = MGF_AD_objective, x0 = [initial_values['shape'], initial_values['scale']], method = 'nelder-mead', args = (sorted_excess_loss), options = {'maxiter':10000}))
            except:
                pass
            try:
                MGF_AD_estimate_results.append(minimize(fun = MGF_AD_objective, x0 = [initial_values['shape'], initial_values['scale']], method = 'L-BFGS-B', bounds = optimization_bounds, args = (sorted_excess_loss), options = {'maxiter':10000, 'maxfun':100000, 'maxcor':500}))
            except:
                pass
        if MGF_AD_estimate_results == [] or np.all([x.success == False for x in MGF_AD_estimate_results]):
            parameter_estimation_dict['MGF_AD']['scale'] = np.nan
            parameter_estimation_dict['MGF_AD']['shape'] = np.nan
        else:               
            best_model_index = np.argmin([x.fun for x in MGF_AD_estimate_results if not np.isnan(x.fun)])
            MGF_AD_shape = MGF_AD_estimate_results[best_model_index]['x'][0]
            MGF_AD_scale = MGF_AD_estimate_results[best_model_index]['x'][1] 
            parameter_estimation_dict['MGF_AD']['scale'] = MGF_AD_scale
            parameter_estimation_dict['MGF_AD']['shape'] = MGF_AD_shape

    #POT-WNLS
    if 'WNLS' in models:
        ECDF_component_1, ECDF_component_2 = ECDF_data_ParkKim2016(test_threshold = initial_threshold*100, random_sample = full_sample.ravel(), tail_sample = tail_sample)
        o1_models = []
        try:
            o1_models.append(minimize(fun = stage_1_POT_NLS_estimator_ParkKim2016, x0 = [0.01, 0.1], method = 'nelder-mead', args = (sorted_excess_loss, ECDF_component_1,), options = {'maxiter':10000}))
        except:
            pass
        try:
            o1_models.append(minimize(fun = stage_1_POT_NLS_estimator_ParkKim2016, x0 = [0.01, 0.1], method = 'L-BFGS-B', bounds = optimization_bounds, args = (sorted_excess_loss, ECDF_component_1,), options = {'maxiter':10000, 'maxfun':10000, 'maxcor':500}))
        except:
            pass
        best_model_index = np.argmin([x.fun for x in o1_models])
        stage_1_params = o1_models[best_model_index]['x']
           
        POT_WNLS_optimization_results = []
        try: 
            POT_WNLS_optimization_results.append(minimize(fun = stage_2_POT_WNLS_estimator_ParkKim2016, x0 = stage_1_params,  method = 'nelder-mead', args = (sorted_excess_loss, ECDF_component_2, np.array(weights_vector)), options = {'maxiter':10000}))
        except:
            pass
        try:             
            POT_WNLS_optimization_results.append(minimize(fun = stage_2_POT_WNLS_estimator_ParkKim2016, x0 = stage_1_params,  method = 'L-BFGS-B', bounds = optimization_bounds, args = (sorted_excess_loss, ECDF_component_2, np.array(weights_vector)), options = {'maxiter':10000, 'maxfun':10000, 'maxcor':500}))
        except:
            pass
        if POT_WNLS_optimization_results == [] or np.all([x.success == False for x in POT_WNLS_optimization_results]):
            parameter_estimation_dict['WNLS']['scale'] = np.nan
            parameter_estimation_dict['WNLS']['shape'] = np.nan
        else:
            best_model_index = np.argmin([x.fun for x in POT_WNLS_optimization_results if not np.isnan(x.fun)])
            POT_WNLS_shape = POT_WNLS_optimization_results[best_model_index]['x'][0]
            POT_WNLS_scale = POT_WNLS_optimization_results[best_model_index]['x'][1]
            parameter_estimation_dict['WNLS']['scale'] = POT_WNLS_scale
            parameter_estimation_dict['WNLS']['shape'] = POT_WNLS_shape
            
    #Maximum Goodness of Fit
    if 'MDE' in models:
        MDE_estimate_results = []
        try:
            MDE_estimate_results.append(minimize(fun = Mestimation_objective, x0 = [parameter_estimation_dict['Revised_NEW']['shape'], parameter_estimation_dict['Revised_NEW']['scale']], method = 'nelder-mead', args = (sorted_excess_loss), options = {'maxiter':10000}))
        except:
            pass
        try:
            MDE_estimate_results.append(minimize(fun = Mestimation_objective, x0 = [parameter_estimation_dict['Revised_NEW']['shape'], parameter_estimation_dict['Revised_NEW']['scale']], method = 'L-BFGS-B', bounds = optimization_bounds, args = (sorted_excess_loss), options = {'maxiter':10000, 'maxfun':100000, 'maxcor':500}))
        except:
            pass
        if MDE_estimate_results == [] or np.all([x.success == False for x in MDE_estimate_results]):
            parameter_estimation_dict['MDE']['scale'] = np.nan
            parameter_estimation_dict['MDE']['shape'] = np.nan
        else:               
            best_model_index = np.argmin([x.fun for x in MDE_estimate_results if not np.isnan(x.fun)])
            MDE_shape = MDE_estimate_results[best_model_index]['x'][0]
            MDE_scale = MDE_estimate_results[best_model_index]['x'][1] 
            parameter_estimation_dict['MDE']['scale'] = MDE_scale
            parameter_estimation_dict['MDE']['shape'] = MDE_shape
    
    #Maximum Goodness of Fit
    if 'Weighted_MDE' in models:
        weighted_MDE_estimate_results = []
        try:
            weighted_MDE_estimate_results.append(minimize(fun = weighted_Mestimation_objective, x0 = [parameter_estimation_dict['MDE']['shape'], parameter_estimation_dict['Revised_NEW']['scale']], method = 'nelder-mead', args = (sorted_excess_loss), options = {'maxiter':10000}))
        except:
            pass
        try:
            weighted_MDE_estimate_results.append(minimize(fun = weighted_Mestimation_objective, x0 = [parameter_estimation_dict['MDE']['shape'], parameter_estimation_dict['Revised_NEW']['scale']], method = 'L-BFGS-B', bounds = optimization_bounds, args = (sorted_excess_loss), options = {'maxiter':10000, 'maxfun':100000, 'maxcor':500}))
        except:
            pass
        if weighted_MDE_estimate_results == [] or np.all([x.success == False for x in weighted_MDE_estimate_results]):
            parameter_estimation_dict['Weighted_MDE']['scale'] = np.nan
            parameter_estimation_dict['Weighted_MDE']['shape'] = np.nan
        else:               
            best_model_index = np.argmin([x.fun for x in weighted_MDE_estimate_results if not np.isnan(x.fun)])
            weighted_MDE_shape = weighted_MDE_estimate_results[best_model_index]['x'][0]
            weighted_MDE_scale = weighted_MDE_estimate_results[best_model_index]['x'][1] 
            parameter_estimation_dict['Weighted_MDE']['scale'] = weighted_MDE_scale
            parameter_estimation_dict['Weighted_MDE']['shape'] = weighted_MDE_shape
    return parameter_estimation_dict
