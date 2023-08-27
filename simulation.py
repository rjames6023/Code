# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:38:18 2021

@author: Robert James
"""
global file_path
file_path = r'/project/RDS-FOB-GPD_Param_Est-RW'
#file_path = r'C:\Users\Robert\Dropbox (Sydney Uni)\GPD_Paramaeter_Estimation_Comparison_Project'

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os
import multiprocessing
import warnings
import sys
sys.path.append(r'{}/Code'.format(file_path))

from tqdm import tqdm
from scipy import stats
from scipy import optimize
from scipy.linalg import expm
from joblib import Parallel, delayed
import scipy.special as sc
from sympy import uppergamma, csc

import param_est_methods
import loss_funcs

#utility functions 
# =============================================================================
def assign_model_ranks(functional, results_frame):
    rankings_list = []
    if functional == 'VaR':
        risk_levels = VaR_levels
    elif functional == 'CTE':
        risk_levels = CTE_levels
    for simulation_params in results_frame['Simulation Parameters'].unique():
        rankings_data = results_frame[(results_frame['Simulation Parameters'] == simulation_params)]
        for VaR_level in risk_levels:
            rankings_data['{} {} Rank'.format(functional, VaR_level)] = rankings_data['{} {}'.format(functional, VaR_level)].rank().values
        rankings_list.append(rankings_data)
    final_results_frame = pd.concat(rankings_list, axis = 0)
    return final_results_frame

def logPH_CDF(x, alpha, T):
    CDF = 1.0-(alpha@expm(T*np.log(x)))@np.ones((len(T),1))
    return CDF[0]    

def logPH_inverse_function(x, alpha, T, p):
    return (1.0-(alpha@expm(T*np.log(x)))@np.ones((len(T),1))) - p

def PH_inverse_function(x, alpha, T, p):
    return (1.0-(alpha@expm(T*x))@np.ones((len(T),1))) - p

def logPHroot_function(uniform_RV, alpha, T):
    return optimize.root_scalar(logPH_inverse_function, args = (alpha, T, uniform_RV), bracket = [1, 1e6]).root  
    
def log_PH_sample_generator(sample, n, alpha, T):
    #uniform RV's
    uniform_sample = stats.uniform.rvs(size = n, random_state=sample)
    inverse_transform_sample = []
    for i in range(n):
        inverse_transform_sample.append(logPHroot_function(uniform_RV=uniform_sample[i],
                                                           alpha=alpha, T=T))
    return inverse_transform_sample

from scipy.optimize import minimize

def convex_combination_objective(theta, shape_estimates, scale_estimates, random_sample, initial_threshold, excess_loss_sample, functional_level):

    shape = theta[:4]@shape_estimates
    scale = theta[4:]@scale_estimates

    r1 = initial_threshold + (scale / shape) * (
                               ((len(random_sample) / len(excess_loss_sample)) * (1 - functional_level)) ** (
                           -shape) - 1)

    r2 = (r1 + scale - (
                shape * initial_threshold)) / (1 - shape)

    # Invert all components of the loss function
    r1 = r1 * -1
    r2 = r2 * -1
    x = random_sample * -1
    alpha = 1 - functional_level

    return (1 / (1 - alpha)) * np.mean(-(1 / (alpha * r2)) * (x <= r1) * (r1 - x) + (r1 / r2) + np.log(-r2) - 1)

def weight_sum_constraint(theta):
    result1 = np.sum(theta[:4]) - 1
    results2 = np.sum(theta[4:]) - 1
    if (result1 + results2) == 0:
        return 0
    else:
        return 1

def convex_combination(parameter_estimation_dict, random_sample_draw, initial_threshold, excess_loss_sample, functional_ranges):

    results_dict = {}
    initial_values = np.array([1/len(parameter_estimation_dict) for i in range(len(parameter_estimation_dict)*2)])
    shape_estimates = np.array([parameter_estimation_dict[key]['shape'] for key in parameter_estimation_dict.keys()])
    scale_estimates = np.array([parameter_estimation_dict[key]['scale'] for key in parameter_estimation_dict.keys()])

    for alpha in functional_ranges:
        theta_hat = minimize(convex_combination_objective,
                             x0=initial_values,
                             args=(shape_estimates, scale_estimates, random_sample_draw, initial_threshold, excess_loss_sample, alpha),
                             method = 'SLSQP',
                             bounds=[(0,1) for x in range(len(initial_values))],
                             constraints=[{'type':'eq', 'fun':weight_sum_constraint}])
        results_dict[alpha] = theta_hat.x

    return results_dict

def min_patton_loss_method_objective(theta, random_sample, initial_threshold, excess_loss_sample, functional_level):

    r1 = initial_threshold + (theta[1] / theta[0]) * (
                               ((len(random_sample) / len(excess_loss_sample)) * (1 - functional_level)) ** (
                           -theta[0]) - 1)

    r2 = (r1 + theta[1] - (
                theta[0] * initial_threshold)) / (1 - theta[0])

    # Invert all components of the loss function
    r1 = r1 * -1
    r2 = r2 * -1
    x = random_sample * -1
    alpha = 1 - functional_level

    return (1 / (1 - alpha)) * np.nanmean(-(1 / (alpha * r2)) * (x <= r1) * (r1 - x) + (r1 / r2) + np.log(-r2) - 1)

def min_patton_loss(random_sample_draw, initial_threshold, excess_loss_sample, functional_ranges):

    results_dict = {}
    MLE_initial_values = param_est_methods.initial_params(excess_loss=excess_loss_sample.ravel(), initial_param_type=[3])
    MLE_initial_values = np.array(list(MLE_initial_values[3].values()))[::-1]
    for alpha in functional_ranges:

        theta_hat = minimize(min_patton_loss_method_objective,
                             x0=MLE_initial_values,
                             args=(random_sample_draw, initial_threshold, excess_loss_sample, alpha),
                             method = 'SLSQP',
                             bounds=optimization_bounds)
        results_dict[alpha] = theta_hat.x

    return results_dict


# =============================================================================
def parameter_estimation_simulation_compute_functionals(sample, random_sample_size, initial_threshold_choice,
                                                        functional, functional_ranges, simulation_distribution,
                                                        shape_param, scale_param, logPH_alpha, logPH_T):

    if simulation_distribution == 'GPD':
        random_sample_draw = stats.genpareto.rvs(loc=0, c=shape_param, scale=scale_param, size=random_sample_size, random_state=sample)
    elif simulation_distribution == 'Pareto':
        random_sample_draw = stats.pareto.rvs(loc=0, b=shape_param, scale=scale_param, size=random_sample_size, random_state=sample)
    elif simulation_distribution == 'Students-t':
        random_sample_draw = stats.t.rvs(loc=0, df=scale_param, size=random_sample_size, random_state=sample)
    elif simulation_distribution == 'Weibull':
        uniform_RVs = stats.uniform.rvs(size=random_sample_size, random_state=sample)
        random_sample_draw = scale_param * (-np.log(1 - uniform_RVs)) ** (1 / shape_param)
    elif simulation_distribution == 'Log-Normal':
        random_sample_draw = stats.lognorm.rvs(size=random_sample_size, s=scale_param, random_state=sample)
    elif simulation_distribution == 'Log-Logistic':
        uniform_RVs = stats.uniform.rvs(size=random_sample_size, random_state=sample)
        random_sample_draw = uniform_RVs * (uniform_RVs / (1 - uniform_RVs)) ** (1 / shape_param)
    elif simulation_distribution == 'LogPH':
        random_sample_draw = np.array(
            log_PH_sample_generator(sample=sample,
                                    n=random_sample_size,
                                    alpha=logPH_alpha,
                                    T=logPH_T))

    #Construct the excess loss sample
    initial_threshold_choice_percentile = 0.9
    initial_threshold_choice = int((1-initial_threshold_choice_percentile)*len(random_sample_draw))
    sorted_random_sample_draw = np.sort(random_sample_draw)
    initial_threshold = sorted_random_sample_draw[-(initial_threshold_choice-1)]

    exceedences = random_sample_draw[random_sample_draw > initial_threshold]

    excess_loss_sample = exceedences - initial_threshold
    sorted_excess_loss_sample = np.sort(excess_loss_sample)

    #Construct weights vector for the WNLS estimator
    variances, weights_vector = param_est_methods.weights(nu = len(sorted_excess_loss_sample), n = random_sample_size)
    weights_vector = weights_vector[::-1]

    #Fit GPD using different methods
    parameter_estimation_dict = {model:{} for model in models}
    parameter_estimation_dict = param_est_methods.parameter_estimation(parameter_estimation_dict,
                                                                       full_sample = random_sample_draw,
                                                                       sorted_excess_loss = sorted_excess_loss_sample,
                                                                       excess_loss = excess_loss_sample,
                                                                       tail_sample = exceedences,
                                                                       initial_threshold = initial_threshold_choice_percentile,
                                                                       weights_vector = weights_vector,
                                                                       models = models,
                                                                       optimization_bounds = optimization_bounds)

    convex_com_results = convex_combination(parameter_estimation_dict={key:parameter_estimation_dict[key] for key in ['MLE', 'LME', 'Revised_NEW', 'WNLS']},
                                                random_sample_draw=random_sample_draw,
                                            initial_threshold=initial_threshold,
                                            excess_loss_sample=excess_loss_sample,
                                            functional_ranges=functional_ranges)
    min_patton_loss_results = min_patton_loss(random_sample_draw=random_sample_draw,
                                               initial_threshold=initial_threshold,
                                               excess_loss_sample=excess_loss_sample,
                                               functional_ranges=functional_ranges)


    RMSE_results_dict = {functional_level:{model:np.nan for model in models + ['Truth', 'convex_combination', 'min_patton_loss']} for functional_level in functional_ranges}
    quantile_loss_results_dict = {functional_level:{model:np.nan for model in models + ['Truth', 'convex_combination', 'min_patton_loss']} for functional_level in functional_ranges}
    loss_0_homogenous_results_dict = {functional_level:{model:np.nan for model in models + ['Truth','convex_combination', 'min_patton_loss']} for functional_level in functional_ranges}

    #Estimate the risk measures
    for functional_level in functional_ranges:
        if functional_level > initial_threshold_choice_percentile:
            if functional == 'VaR':
                #True VaR
                if simulation_distribution == 'GPD':
                    true_functional = stats.genpareto.ppf(functional_level, c = shape_param, loc = 0, scale = scale_param)
                    RV_draw = stats.genpareto.rvs(size = random_sample_size, c = shape_param, loc = 0, scale = scale_param, random_state=sample+1)
                elif simulation_distribution == 'Pareto':
                    true_functional = stats.pareto.ppf(functional_level, b = shape_param, loc = 0, scale = scale_param)
                    RV_draw = stats.pareto.rvs(size = random_sample_size, b = shape_param, loc = 0, scale = scale_param, random_state=sample+1)
                elif simulation_distribution == 'Students-t':
                    true_functional = stats.t.ppf(functional_level, loc = 0, df = scale_param)
                    RV_draw = stats.t.rvs(size = random_sample_size, loc = 0, df = scale_param, random_state=sample+1)
                elif simulation_distribution == 'Weibull':
                    true_functional = scale_param*(-np.log(1-functional_level))**(1/shape_param)
                    uniform_RVs = stats.uniform.rvs(size = random_sample_size, random_state=sample+1)
                    RV_draw = scale_param*(-np.log(1-uniform_RVs))**(1/shape_param)
                elif simulation_distribution == 'Log-Normal':
                    true_functional = stats.lognorm.ppf(functional_level, s = scale_param)
                    RV_draw = stats.lognorm.rvs(size = random_sample_size, s = scale_param, random_state=sample+1)
                elif simulation_distribution == 'Log-Logistic':
                    true_functional = functional_level*(functional_level/(1- functional_level))**(1/shape_param)
                    uniform_RVs = stats.uniform.rvs(size = random_sample_size, random_state=sample+1)
                    RV_draw = uniform_RVs*(uniform_RVs/(1- uniform_RVs))**(1/shape_param)
                elif simulation_distribution == 'LogPH':
                    true_functional = optimize.root_scalar(logPH_inverse_function, args = (logPH_alpha, logPH_T, functional_level), bracket = [1, 1e6]).root
                    RV_draw = np.array(log_PH_sample_generator(sample=sample+1,
                                                               n = random_sample_size, alpha = logPH_alpha, T = logPH_T))
            else:
                #True CTE
                if simulation_distribution == 'GPD':
                    true_VaR = stats.genpareto.ppf(functional_level, c = shape_param, loc = 0, scale = scale_param)
                    true_functional = scale_param*((((1 - functional_level)**(-shape_param)))/(1-shape_param) + (((1 - functional_level)**(-shape_param)) - 1)/(shape_param))
                    RV_draw = stats.genpareto.rvs(size = random_sample_size, c = shape_param, loc = 0, scale = scale_param, random_state=sample+1)
                elif simulation_distribution == 'Pareto':
                    true_VaR = stats.pareto.ppf(functional_level, b = shape_param, loc = 0, scale = scale_param)
                    true_functional = (scale_param*(1-functional_level)**(-1/shape_param))*(shape_param/(shape_param-1))
                    RV_draw = stats.pareto.rvs(size = random_sample_size, b = shape_param, loc = 0, scale = scale_param, random_state=sample+1)
                elif simulation_distribution == 'Students-t':
                    true_VaR = stats.t.ppf(functional_level, loc = 0, df = scale_param)
                    true_functional = (stats.t.pdf(true_VaR, df = scale_param)/(1-functional_level))*((scale_param + true_VaR**2)/(scale_param - 1))
                    RV_draw = stats.t.rvs(size = random_sample_size, loc = 0, df = scale_param, random_state=sample+1)
                elif simulation_distribution == 'Weibull':
                    true_VaR = scale_param*(-np.log(1-functional_level))**(1/shape_param)
                    true_functional = (scale_param/(1-functional_level))*float(uppergamma(1 + 1/shape_param, -np.log(1-functional_level)))
                    uniform_RVs = stats.uniform.rvs(size = random_sample_size, random_state=sample+1)
                    RV_draw = scale_param*(-np.log(1-uniform_RVs))**(1/shape_param)
                elif simulation_distribution == 'Log-Normal':
                    true_VaR = stats.lognorm.ppf(functional_level, s = scale_param)
                    erf_term = (1 + sc.erf(scale_param/np.sqrt(2) - sc.erfinv(2*functional_level -1)))/(1- functional_level)
                    true_functional = 0.5*np.exp((scale_param**2)/2)*erf_term
                    RV_draw = stats.lognorm.rvs(size = random_sample_size, s = scale_param, random_state=sample+1)
                elif simulation_distribution == 'Log-Logistic':
                    true_VaR = functional_level*(functional_level/(1- functional_level))**(1/shape_param)
                    true_functional = (functional_level/(1-functional_level))*( np.pi/shape_param*float(csc(np.pi/shape_param)) - sc.betainc(1/shape_param + 1, 1- 1/shape_param, functional_level)*sc.beta(1/shape_param + 1, 1- 1/shape_param))
                    uniform_RVs = stats.uniform.rvs(size = random_sample_size, random_state=sample+1)
                    RV_draw = uniform_RVs*(uniform_RVs/(1- uniform_RVs))**(1/shape_param)
                elif simulation_distribution == 'LogPH':
                    true_VaR = optimize.root_scalar(logPH_inverse_function, args = (logPH_alpha, logPH_T, functional_level), bracket = [1, 1e6]).root
                    true_VaR_PH = optimize.root_scalar(PH_inverse_function, args = (logPH_alpha, logPH_T, functional_level), bracket = [1, 1e6]).root
                    alpha_d = (logPH_alpha@expm(true_VaR_PH*logPH_T))/((logPH_alpha@expm(true_VaR_PH*logPH_T))@np.ones((len(logPH_T), 1))) #eq.7 Ahn et al. 2012
                    true_functional = (-true_VaR*alpha_d@np.linalg.inv((np.eye(2) + logPH_T))@(-logPH_T@np.ones((len(logPH_T), 1))))[0]
                    RV_draw = np.array(log_PH_sample_generator(sample=sample+1,
                                                               n = random_sample_size, alpha = logPH_alpha, T = logPH_T))

            model_VaRs = {}
            model_functionals = {}
            for model in models:
                VaR_estimate = initial_threshold + (parameter_estimation_dict[model]['scale']/parameter_estimation_dict[model]['shape'])*(((random_sample_size/len(excess_loss_sample))*(1-functional_level))**(-parameter_estimation_dict[model]['shape']) - 1)
                model_VaRs[model] = VaR_estimate
                if functional == 'VaR':
                    functional_estimate = VaR_estimate
                else:
                    if parameter_estimation_dict[model]['shape'] >= 1:
                        parameter_estimation_dict[model]['shape'] = 0.99
                    functional_estimate = (VaR_estimate + parameter_estimation_dict[model]['scale'] - (parameter_estimation_dict[model]['shape'] * initial_threshold))/(1 - parameter_estimation_dict[model]['shape'])
                model_functionals[model] = functional_estimate
                if functional == 'VaR':
                    loss_0_homogenous = loss_funcs.tick_loss_0_homogenous(r=functional_estimate,
                                                                          x=RV_draw,
                                                                          alpha=functional_level)
                    loss_standard_quantile = loss_funcs.quantile_loss(r=functional_estimate,
                                                                      x=RV_draw,
                                                                      alpha=functional_level)/random_sample_size
                else:
                    loss_0_homogenous = loss_funcs.joint_VaR_CTE_patton_0_homogenous(r1 = VaR_estimate,
                                                                                     r2 = functional_estimate,
                                                                                     x = RV_draw,
                                                                                     alpha = functional_level)
                    loss_standard_quantile = None

                RMSE_results_dict[functional_level][model] = functional_estimate - true_functional
                quantile_loss_results_dict[functional_level][model] = loss_standard_quantile
                loss_0_homogenous_results_dict[functional_level][model] = loss_0_homogenous
                
                #Add scoring function results for the true functional
                if functional == 'VaR':
                    truth_loss_0_homogenous = loss_funcs.tick_loss_0_homogenous(r = true_functional, x = RV_draw, alpha = functional_level)
                    truth_loss_standard_quantile = loss_funcs.quantile_loss(r = true_functional, x = RV_draw, alpha = functional_level)/random_sample_size                                                                                             
                else:
                    truth_loss_0_homogenous = loss_funcs.joint_VaR_CTE_patton_0_homogenous(r1 = true_VaR, r2 = true_functional, x = RV_draw, alpha = functional_level)
                    truth_loss_standard_quantile = None
                RMSE_results_dict[functional_level]['Truth'] = functional_estimate - true_functional
                quantile_loss_results_dict[functional_level]['Truth'] = truth_loss_standard_quantile
                loss_0_homogenous_results_dict[functional_level]['Truth'] = truth_loss_0_homogenous

            # Add losses for convex combination and patton loss methods
            shape_estimates = np.array(
                [parameter_estimation_dict[key]['shape'] for key in ['MLE', 'LME', 'Revised_NEW', 'WNLS']])
            scale_estimates = np.array(
                [parameter_estimation_dict[key]['scale'] for key in ['MLE', 'LME', 'Revised_NEW', 'WNLS']])
            convex_comb_shape = convex_com_results[functional_level][:4]@shape_estimates
            convex_comb_scale = convex_com_results[functional_level][4:]@scale_estimates
            convex_comb_VaR_estimate = initial_threshold + (
                        convex_comb_scale / convex_comb_shape) * (
                                       ((random_sample_size / len(excess_loss_sample)) * (1 - functional_level)) ** (
                                   -convex_comb_shape) - 1)
            if convex_comb_shape >= 1:
                convex_comb_shape = 0.99
            functional_estimate = (convex_comb_VaR_estimate + convex_comb_scale - (
                        convex_comb_shape * initial_threshold)) / (
                        1 - convex_comb_shape)

            loss_0_homogenous = loss_funcs.joint_VaR_CTE_patton_0_homogenous(r1=convex_comb_VaR_estimate,
                                                                             r2=functional_estimate,
                                                                             x=RV_draw,
                                                                             alpha=functional_level)
            loss_standard_quantile = None
            RMSE_results_dict[functional_level]['convex_combination'] = functional_estimate - true_functional
            quantile_loss_results_dict[functional_level]['convex_combination'] = loss_standard_quantile
            loss_0_homogenous_results_dict[functional_level]['convex_combination'] = loss_0_homogenous

            patton_loss_scale = min_patton_loss_results[functional_level][1]
            patton_loss_shape = min_patton_loss_results[functional_level][0]
            min_patton_loss_VaR_estimate = initial_threshold + (
                    patton_loss_scale / patton_loss_shape) * (
                                               ((random_sample_size / len(excess_loss_sample)) * (
                                                           1 - functional_level)) ** (
                                                   -patton_loss_shape) - 1)
            if patton_loss_shape >= 1:
                patton_loss_shape = 0.99
            functional_estimate = (min_patton_loss_VaR_estimate + patton_loss_scale - (
                    patton_loss_shape * initial_threshold)) / (
                                          1 - patton_loss_shape)

            loss_0_homogenous = loss_funcs.joint_VaR_CTE_patton_0_homogenous(r1=min_patton_loss_VaR_estimate,
                                                                             r2=functional_estimate,
                                                                             x=RV_draw,
                                                                             alpha=functional_level)
            loss_standard_quantile = None
            RMSE_results_dict[functional_level]['min_patton_loss'] = functional_estimate - true_functional
            quantile_loss_results_dict[functional_level]['min_patton_loss'] = loss_standard_quantile
            loss_0_homogenous_results_dict[functional_level]['min_patton_loss'] = loss_0_homogenous

    return RMSE_results_dict, loss_0_homogenous_results_dict, quantile_loss_results_dict, parameter_estimation_dict

def global_variable_initializer():
    global n_replications, initial_threshold_choice, VaR_levels, models, CTE_levels, empirical_application_VaR_levels, \
        penalized_MLE_lambda, penalized_MLE_alpha, empirical_application_initial_threshold_choices, \
        VaR_initial_threshold_choices, CTE_initial_threshold_choices, optimization_bounds, VaR_murphy_ranges

    n_replications = 10000
    VaR_levels = np.array([0.925, 0.95, 0.975, 0.99, 0.999, 0.9999])
    CTE_levels = np.array([0.925, 0.95, 0.975, 0.99, 0.999])
    models = ['L-Moments', 'MLE', 'LME', 'NEW', 'Revised_NEW', 'MGF_AD', 'MGF_AD2R', 'WNLS', 'MDE']
    penalized_MLE_lambda = 1
    penalized_MLE_alpha = 1
    initial_threshold_choice = 100
    optimization_bounds = [(0.001, 1), (0.001, np.inf)]
    
def main(distribution, shape, scale, functional, n):
    global n_replications, initial_threshold_choice, VaR_levels, models, CTE_levels, empirical_application_VaR_levels, \
        penalized_MLE_lambda, penalized_MLE_alpha, empirical_application_initial_threshold_choices, \
        VaR_initial_threshold_choices, CTE_initial_threshold_choices, optimization_bounds
    # =============================================================================
    #Initialize global variables
    global_variable_initializer()    
    # if not os.path.exists(r'{}/Results/Simulation'.format(file_path)):
    #     os.makedirs(r'{}/Results/Simulation'.format(file_path))
    
    # =============================================================================
    # =============================================================================
    print('Starting {} {} {} {} n = {}'.format(distribution, functional, shape, scale,
                                               n))

    logPH_alpha = np.array([0.622, 0.378])
    logPH_T = np.array([[-4, 3.564],
                        [0.267, -1.813]])

    if functional == 'VaR':
        functional_ranges = VaR_levels
        functional_columns = ['{} {}'.format(functional, x) for x in functional_ranges]
    else:
        functional_ranges = CTE_levels
        functional_columns = ['{} {}'.format(functional, x) for x in functional_ranges]

    RMSE_error_frame = pd.DataFrame(columns=['u'] + functional_columns,
                                    index=models + ['Truth', 'convex_combination', 'min_patton_loss'])
    loss_error_frame_0_homogenous = pd.DataFrame(columns=['u'] + functional_columns,
                                                 index=models + ['Truth', 'convex_combination', 'min_patton_loss'])
    loss_error_frame_standard_quantile = pd.DataFrame(columns=['u', ] + functional_columns,
                                                      index=models + ['Truth', 'convex_combination', 'min_patton_loss'])
    RMSE_error_frame['u'] = initial_threshold_choice
    loss_error_frame_0_homogenous['u'] = initial_threshold_choice
    loss_error_frame_standard_quantile['u'] = initial_threshold_choice

    # Error frames for the RMSE replication
    RMSE_frame_dict = {model: pd.DataFrame(columns=functional_columns, index=range(n_replications)) for model in
                       models + ['Truth', 'convex_combination', 'min_patton_loss']}
    # Prediction frames for the consistent scoring functions
    loss_frame_0_homogenous = {model: pd.DataFrame(columns=functional_columns, index=range(n_replications)) for model in
                               models + ['Truth', 'convex_combination', 'min_patton_loss']}
    loss_frame_standard_quantile = {model: pd.DataFrame(columns=functional_columns, index=range(n_replications)) for
                                    model in models + ['Truth', 'convex_combination', 'min_patton_loss']}

    simulation_results = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(
        delayed(parameter_estimation_simulation_compute_functionals)(sample,
                                                                     n,
                                                                     initial_threshold_choice,
                                                                     functional,
                                                                     functional_ranges,
                                                                     distribution,
                                                                     shape  ,
                                                                     scale,
                                                                     logPH_alpha,
                                                                     logPH_T)
        for sample in range(n_replications))

    mean_shape_parameter_estimates = pd.DataFrame(columns=models, index=range(n_replications))
    mean_scale_parameter_estimates = pd.DataFrame(columns=models, index=range(n_replications))
    for i, result in enumerate(simulation_results):
        for functional_level in functional_ranges:
            for model in models + ['Truth', 'convex_combination', 'min_patton_loss']:
                RMSE_frame_dict[model].at[i, '{} {}'.format(functional, functional_level)] = \
                result[0][functional_level][model]
                loss_frame_0_homogenous[model].at[i, '{} {}'.format(functional, functional_level)] = \
                result[1][functional_level][model]
                if functional == 'VaR':
                    loss_frame_standard_quantile[model].at[i, '{} {}'.format(functional, functional_level)] = \
                    result[2][functional_level][model]
    for i, result in enumerate(simulation_results):
        for model in models:
            mean_shape_parameter_estimates.at[i, model] = simulation_results[i][3][model]['shape']
            mean_scale_parameter_estimates.at[i, model] = simulation_results[i][3][model]['scale']

    for model in models + ['Truth', 'convex_combination', 'min_patton_loss']:
        RMSE_error_frame.loc[[model], RMSE_error_frame.columns[1:]] = np.sqrt(
            np.mean(np.square(RMSE_frame_dict[model]))).values.T
        loss_error_frame_0_homogenous.loc[[model], loss_error_frame_0_homogenous.columns[1:]] = np.mean(
            loss_frame_0_homogenous[model]).values.T
        if functional == 'VaR':
            loss_error_frame_standard_quantile.loc[[model], RMSE_error_frame.columns[1:]] = np.mean(loss_frame_standard_quantile[model]).values.T

    RMSE_error_frame['Simulation Parameters'] = '{}({}, {})'.format(distribution, shape, scale)
    RMSE_error_frame = RMSE_error_frame[['Simulation Parameters'] + RMSE_error_frame.columns.tolist()[:-1]]

    loss_error_frame_0_homogenous['Simulation Parameters'] = '{}({}, {})'.format(distribution, shape, scale)
    loss_error_frame_0_homogenous = loss_error_frame_0_homogenous[
        ['Simulation Parameters'] + loss_error_frame_0_homogenous.columns.tolist()[:-1]]

    if functional == 'VaR':
        loss_error_frame_standard_quantile['Simulation Parameters'] = '{}({}, {})'.format(distribution, shape, scale)
        loss_error_frame_standard_quantile = loss_error_frame_standard_quantile[
            ['Simulation Parameters'] + loss_error_frame_standard_quantile.columns.tolist()[:-1]]
    print('Completed')

    if functional == 'VaR':
        loss_error_frame_standard_quantile = loss_error_frame_standard_quantile
        loss_error_frame_standard_quantile = assign_model_ranks(functional = functional, results_frame = loss_error_frame_standard_quantile)
    
    #Assign Rankings
    RMSE_error_frame = assign_model_ranks(functional = functional, results_frame = RMSE_error_frame)
    loss_error_frame_0_homogenous = assign_model_ranks(functional = functional, results_frame = loss_error_frame_0_homogenous)
    if distribution in ['GPD', 'Students-t', 'Log-Normal', 'Log-Logistic', 'Pareto', 'Weibull']:
        if functional == 'VaR':
            loss_error_frame_standard_quantile.to_csv(r'{}/Results/Simulation/{}_{}_quantile_loss_frame_n={}_shape={}_scale={}.csv'.format(file_path, distribution, functional, n, shape, scale), index = True)
            mean_shape_parameter_estimates.to_csv(r'{}/Results/Simulation/{}_mean_shape_param_n={}_shape={}_scale={}.csv'.format(file_path, distribution, n, shape, scale), index = False)
            mean_scale_parameter_estimates.to_csv(r'{}/Results/Simulation/{}_mean_scale_param_n={}_shape={}_scale={}.csv'.format(file_path, distribution, n, shape, scale), index = False)
        RMSE_error_frame.to_csv(r'{}/Results/Simulation/{}_{}_RMSE_error_frame_n={}_shape={}_scale={}.csv'.format(file_path, distribution, functional, n, shape, scale), index = True)
        loss_error_frame_0_homogenous.to_csv(r'{}/Results/Simulation/{}_{}_0_homogenous_loss_frame_n={}_shape={}_scale={}.csv'.format(file_path, distribution, functional, n, shape, scale), index = True)
    elif distribution == 'LogPH':
        if functional == 'VaR':
            loss_error_frame_standard_quantile.to_csv(r'{}/Results/Simulation/{}_{}_quantile_loss_frame_n={}.csv'.format(file_path, distribution, functional, n), index = True)
            mean_shape_parameter_estimates.to_csv(r'{}/Results/Simulation/{}_mean_shape_param_n={}.csv'.format(file_path, distribution, n), index = False)
            mean_scale_parameter_estimates.to_csv(r'{}/Results/Simulation/{}_mean_scale_param_n={}.csv'.format(file_path, distribution, n), index = False)
        RMSE_error_frame.to_csv(r'{}/Results/Simulation/{}_{}_RMSE_error_frame_n={}.csv'.format(file_path, distribution, functional, n), index = True)
        loss_error_frame_0_homogenous.to_csv(r'{}/Results/Simulation/{}_{}_0_homogenous_loss_frame_n={}.csv'.format(file_path, distribution, functional, n), index = True)

if __name__ == '__main__':  
    warnings.filterwarnings('ignore')
    np.seterr(all = 'ignore')
    
    distribution = str(sys.argv[1]).strip()
    shape = str(sys.argv[2]).strip()
    if shape == 'None':
        shape = None
    else:       
        shape = float(shape)
    scale = str(sys.argv[3]).strip()
    if scale == 'None':
        scale = None
    else:       
        scale = float(scale)
    functional = str(sys.argv[4]).strip()
    n = int(sys.argv[5])
    main(distribution = distribution,
         shape = shape,
         scale = scale,
         functional = functional,
         n = n)
