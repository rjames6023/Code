# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 11:38:18 2021

@author: Robert James
"""
global file_path
file_path = r'/project/RDS-FOB-GPD_Param_Est-RW'
#file_path = r'C:\Users\Robert James\Dropbox (Sydney Uni)\GPD_Paramaeter_Estimation_Comparison_Project'

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
        for threshold in initial_threshold_choices:
            rankings_data = results_frame[(results_frame['Simulation Parameters'] == simulation_params) & (results_frame['u'] == threshold)]
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
    
def log_PH_sample_generator(n, alpha, T, processing_pool):
    #uniform RV's
    uniform_sample = stats.uniform.rvs(size = n)
    mapping_list = []
    for i in range(n):
        mapping_list.append([uniform_sample[i], alpha, T])
    if processing_pool == None:
        inverse_transform_sample = []
        for i in range(len(mapping_list)):
            inverse_transform_sample.append(logPHroot_function(uniform_RV = mapping_list[i][0], alpha = mapping_list[i][1], T = mapping_list[i][2]))
    else:
        inverse_transform_sample = processing_pool.starmap(logPHroot_function, mapping_list)
    return inverse_transform_sample

# =============================================================================
def parameter_estimation_simulation_compute_functionals(random_sample_draw, random_sample_size, initial_threshold_choice, functional, functional_ranges, simulation_distribution, shape_param, scale_param):
    #Construct the excess loss sample
    initial_threshold_choice_percentile = 1-(initial_threshold_choice/random_sample_size)
#    initial_threshold = np.percentile(random_sample_draw, 100*initial_threshold_choice_percentile)
    sorted_random_sample_draw = np.sort(random_sample_draw)
    initial_threshold = sorted_random_sample_draw[-initial_threshold_choice-1]
    
    exceedences = random_sample_draw[random_sample_draw > initial_threshold]
    
    excess_loss_sample = exceedences - initial_threshold
    sorted_excess_loss_sample = np.sort(excess_loss_sample)
    
    #Construct weights vector for the WNLS estimator
    variances, weights_vector = param_est_methods.weights(nu = len(sorted_excess_loss_sample), n = random_sample_size)
    weights_vector = weights_vector[::-1]
    
    #Fit GPD using different methods 
    parameter_estimation_dict = {model:{} for model in models}
    parameter_estimation_dict = param_est_methods.parameter_estimation(parameter_estimation_dict, full_sample = random_sample_draw, sorted_excess_loss = sorted_excess_loss_sample, excess_loss = excess_loss_sample, tail_sample = exceedences, 
                                                                       initial_threshold = initial_threshold_choice_percentile, weights_vector = weights_vector, models = models, optimization_bounds = optimization_bounds)
    
    RMSE_results_dict = {functional_level:{model:np.nan for model in models + ['Truth']} for functional_level in functional_ranges}
    quantile_loss_results_dict = {functional_level:{model:np.nan for model in models + ['Truth']} for functional_level in functional_ranges}
    loss_1_homogenous_results_dict = {functional_level:{model:np.nan for model in models + ['Truth']} for functional_level in functional_ranges}
    loss_0_homogenous_results_dict = {functional_level:{model:np.nan for model in models + ['Truth']} for functional_level in functional_ranges}
#    Murphy_diagram_scores_dict = {functional_level:{model:np.nan for model in models} for functional_level in functional_ranges}

    #Estimate the VaRs
    if simulation_distribution == 'LogPH':
        alpha = np.array([0.622, 0.378])
        T = np.array([[-4, 3.564], 
                       [0.267, -1.813]])    
    for functional_level in functional_ranges:
        if functional_level > initial_threshold_choice_percentile:
            if functional == 'VaR':
                #True VaR
                if simulation_distribution == 'GPD':
                    true_functional = stats.genpareto.ppf(functional_level, c = shape_param, loc = 0, scale = scale_param)
                    RV_draw = stats.genpareto.rvs(size = random_sample_size, c = shape_param, loc = 0, scale = scale_param)
                elif simulation_distribution == 'Pareto':
                    true_functional = stats.pareto.ppf(functional_level, b = shape_param, loc = 0, scale = scale_param)       
                    RV_draw = stats.pareto.rvs(size = random_sample_size, b = shape_param, loc = 0, scale = scale_param)
                elif simulation_distribution == 'Students-t':
                    true_functional = stats.t.ppf(functional_level, loc = 0, df = scale_param)    
                    RV_draw = stats.t.rvs(size = random_sample_size, loc = 0, df = scale_param)
                    random_sample_draw = np.exp(stats.gamma.rvs(loc = 0, a = 2, scale = scale_param, size = random_sample_size))
                elif simulation_distribution == 'Weibull': 
                    true_functional = scale_param*(-np.log(1-functional_level))**(1/shape_param)
                    uniform_RVs = stats.uniform.rvs(size = random_sample_size)
                    RV_draw = scale_param*(-np.log(1-uniform_RVs))**(1/shape_param)
                elif simulation_distribution == 'Log-Normal':
                    true_functional = stats.lognorm.ppf(functional_level, s = scale_param)
                    RV_draw = stats.lognorm.rvs(size = random_sample_size, s = scale_param) 
                elif simulation_distribution == 'Log-Logistic':
                    true_functional = functional_level*(functional_level/(1- functional_level))**(1/shape_param)
                    uniform_RVs = stats.uniform.rvs(size = random_sample_size)
                    RV_draw = uniform_RVs*(uniform_RVs/(1- uniform_RVs))**(1/shape_param)
                elif simulation_distribution == 'LogPH':
                    true_functional = optimize.root_scalar(logPH_inverse_function, args = (alpha, T, functional_level), bracket = [1, 1e6]).root
                    true_VaR_PH = optimize.root_scalar(PH_inverse_function, args = (alpha, T, functional_level), bracket = [1, 1e6]).root       
                    RV_draw = np.array(log_PH_sample_generator(n = random_sample_size, alpha = alpha, T = T, processing_pool = None))
            else:                   
                #True CTE
                if simulation_distribution == 'GPD':
                    true_VaR = stats.genpareto.ppf(functional_level, c = shape_param, loc = 0, scale = scale_param)
                    true_functional = scale_param*((((1 - functional_level)**(-shape_param)))/(1-shape_param) + (((1 - functional_level)**(-shape_param)) - 1)/(shape_param))
                    RV_draw = stats.genpareto.rvs(size = random_sample_size, c = shape_param, loc = 0, scale = scale_param)
                elif simulation_distribution == 'Pareto':
                    true_VaR = stats.pareto.ppf(functional_level, b = shape_param, loc = 0, scale = scale_param)       
                    true_functional = (scale_param*(1-functional_level)**(-1/shape_param))*(shape_param/(shape_param-1))  
                    RV_draw = stats.pareto.rvs(size = random_sample_size, b = shape_param, loc = 0, scale = scale_param)
                elif simulation_distribution == 'Students-t':
                    true_VaR = stats.t.ppf(functional_level, loc = 0, df = scale_param)       
                    true_functional = (stats.t.pdf(true_VaR, df = scale_param)/(1-functional_level))*((scale_param + true_VaR**2)/(scale_param - 1))
                    RV_draw = stats.t.rvs(size = random_sample_size, loc = 0, df = scale_param)
                elif simulation_distribution == 'Weibull':
                    true_VaR = scale_param*(-np.log(1-functional_level))**(1/shape_param)
                    true_functional = (scale_param/(1-functional_level))*float(uppergamma(1 + 1/shape_param, -np.log(1-functional_level)))
                    uniform_RVs = stats.uniform.rvs(size = random_sample_size)
                    RV_draw = scale_param*(-np.log(1-uniform_RVs))**(1/shape_param)
                elif simulation_distribution == 'Log-Normal':
                    true_VaR = stats.lognorm.ppf(functional_level, s = scale_param)
                    erf_term = (1 + sc.erf(scale_param/np.sqrt(2) - sc.erfinv(2*functional_level -1)))/(1- functional_level)
                    true_functional = 0.5*np.exp((scale_param**2)/2)*erf_term
                    RV_draw = stats.lognorm.rvs(size = random_sample_size, s = scale_param) 
                elif simulation_distribution == 'Log-Logistic':     
                    true_VaR = functional_level*(functional_level/(1- functional_level))**(1/shape_param)
                    true_functional = (functional_level/(1-functional_level))*( np.pi/shape_param*float(csc(np.pi/shape_param)) - sc.betainc(1/shape_param + 1, 1- 1/shape_param, functional_level)*sc.beta(1/shape_param + 1, 1- 1/shape_param))
                    uniform_RVs = stats.uniform.rvs(size = random_sample_size)
                    RV_draw = uniform_RVs*(uniform_RVs/(1- uniform_RVs))**(1/shape_param)
                elif simulation_distribution == 'LogPH':
                    true_VaR = optimize.root_scalar(logPH_inverse_function, args = (alpha, T, functional_level), bracket = [1, 1e6]).root
                    true_VaR_PH = optimize.root_scalar(PH_inverse_function, args = (alpha, T, functional_level), bracket = [1, 1e6]).root       
                    alpha_d = (alpha@expm(true_VaR_PH*T))/((alpha@expm(true_VaR_PH*T))@np.ones((len(T), 1))) #eq.7 Ahn et al. 2012
                    true_functional = (-true_VaR*alpha_d@np.linalg.inv((np.eye(2) + T))@(-T@np.ones((len(T), 1))))[0]
                    RV_draw = np.array(log_PH_sample_generator(n = random_sample_size, alpha = alpha, T = T, processing_pool = None))

            for model in models:
                VaR_estimate = initial_threshold + (parameter_estimation_dict[model]['scale']/parameter_estimation_dict[model]['shape'])*(((random_sample_size/len(excess_loss_sample))*(1-functional_level))**(-parameter_estimation_dict[model]['shape']) - 1)
                if functional == 'VaR':
                    functional_estimate = VaR_estimate
                else:
                    if parameter_estimation_dict[model]['shape'] >= 1:
                        parameter_estimation_dict[model]['shape'] = 0.99
                    functional_estimate = (VaR_estimate + parameter_estimation_dict[model]['scale'] - (parameter_estimation_dict[model]['shape'] * initial_threshold))/(1 - parameter_estimation_dict[model]['shape'])
                if functional == 'VaR':
                    loss_1_homogenous = loss_funcs.tick_loss_1_homogenous(r = functional_estimate, x = RV_draw, alpha = functional_level)/random_sample_size
                    loss_0_homogenous = loss_funcs.tick_loss_0_homogenous(r = functional_estimate, x = RV_draw, alpha = functional_level)
                    loss_standard_quantile = loss_funcs.quantile_loss(r = functional_estimate, x = RV_draw, alpha = functional_level)/random_sample_size                                                                                             
                else:
                    loss_1_homogenous = loss_funcs.joint_VaR_CTE_loss(r1 = VaR_estimate, r2 = functional_estimate, x = RV_draw, alpha = functional_level)
                    loss_0_homogenous = loss_funcs.joint_VaR_CTE_loss_0_homogenous(r1 = VaR_estimate, r2 = functional_estimate, x = RV_draw, alpha = functional_level)
                    loss_standard_quantile = None
                RMSE_results_dict[functional_level][model] = functional_estimate - true_functional
                quantile_loss_results_dict[functional_level][model] = loss_standard_quantile
                loss_1_homogenous_results_dict[functional_level][model] = loss_1_homogenous
                loss_0_homogenous_results_dict[functional_level][model] = loss_0_homogenous
                
                #Add scoring function results for the true functional
                if functional == 'VaR':
                    truth_loss_1_homogenous = loss_funcs.tick_loss_1_homogenous(r = true_functional, x = RV_draw, alpha = functional_level)/random_sample_size
                    truth_loss_0_homogenous = loss_funcs.tick_loss_0_homogenous(r = true_functional, x = RV_draw, alpha = functional_level)
                    truth_loss_standard_quantile = loss_funcs.quantile_loss(r = true_functional, x = RV_draw, alpha = functional_level)/random_sample_size                                                                                             
                else:
                    truth_loss_1_homogenous = loss_funcs.joint_VaR_CTE_loss(r1 = true_VaR, r2 = true_functional, x = RV_draw, alpha = functional_level)
                    truth_loss_0_homogenous = loss_funcs.joint_VaR_CTE_loss_0_homogenous(r1 = true_VaR, r2 = true_functional, x = RV_draw, alpha = functional_level)
                    truth_loss_standard_quantile = None
                RMSE_results_dict[functional_level]['Truth'] = functional_estimate - true_functional
                quantile_loss_results_dict[functional_level]['Truth'] = truth_loss_standard_quantile
                loss_1_homogenous_results_dict[functional_level]['Truth'] = truth_loss_1_homogenous
                loss_0_homogenous_results_dict[functional_level]['Truth'] = truth_loss_0_homogenous
                
                #Compute scores for Murphy Diagrams
#                if functional == 'VaR':
#                    murphy_diagram_score = Murphy_Diagram_Scores(VaR_murphy_ranges, functional, true_functional, functional_estimate, functional_level, simulation_distribution, shape_param, scale_param)
#                Murphy_diagram_scores_dict[functional_level][model] = murphy_diagram_score                        
    return RMSE_results_dict, loss_1_homogenous_results_dict, loss_0_homogenous_results_dict, quantile_loss_results_dict, parameter_estimation_dict
                    
def parameter_estimation_simulation(shape_param, scale_param, functional, simulation_distribution, processing_pool, random_sample_size = 10000):
    print('Starting {} {} {} {} n = {}'.format(simulation_distribution, functional, shape_param, scale_param, random_sample_size))
    if simulation_distribution == 'LogPH':
        alpha = np.array([0.622, 0.378])
        T = np.array([[-4, 3.564], 
                       [0.267, -1.813]])    
    if functional == 'VaR':
        functional_ranges = VaR_levels
        functional_columns = ['{} {}'.format(functional, x) for x in functional_ranges]
    else:
        functional_ranges = CTE_levels
        functional_columns = ['{} {}'.format(functional, x) for x in functional_ranges]
    
    final_mean_shape_parameter_estimates = []
    final_mean_scale_parameter_estimates = []
    RMSE_error_frame_list = []
    loss_frame_list_1_homogenous = []
    loss_frame_list_0_homogenous = []
    loss_frame_list_standard_quantile = []

    for initial_threshold_choice in initial_threshold_choices:
        RMSE_error_frame = pd.DataFrame(columns = ['u', 'Model'] + functional_columns)
        RMSE_error_frame['Model'] = models + ['Truth']
        loss_error_frame_1_homogenous = pd.DataFrame(columns = ['u', 'Model'] + functional_columns)
        loss_error_frame_1_homogenous['Model'] = models + ['Truth']
        loss_error_frame_0_homogenous = pd.DataFrame(columns = ['u', 'Model'] + functional_columns)
        loss_error_frame_0_homogenous['Model'] = models + ['Truth']
        loss_error_frame_standard_quantile = pd.DataFrame(columns = ['u', 'Model'] + functional_columns)
        loss_error_frame_standard_quantile['Model'] = models + ['Truth']

        #Error frames for the RMSE replication
        RMSE_frame_dict = {model:pd.DataFrame(columns = functional_columns, index = range(n_replications)) for model in models + ['Truth']}
        #Prediction frames for the consistent scoring functions
        loss_frame_1_homogenous = {model:pd.DataFrame(columns = functional_columns, index = range(n_replications)) for model in models + ['Truth']}
        loss_frame_0_homogenous = {model:pd.DataFrame(columns = functional_columns, index = range(n_replications)) for model in models + ['Truth']}
        loss_frame_standard_quantile = {model:pd.DataFrame(columns = functional_columns, index = range(n_replications)) for model in models + ['Truth']}
        
#        Murphy_diagram_scores_frame_dict = {model:{functional_level:np.zeros((501, n_replications)) for functional_level in functional_ranges} for model in models}

        mapping_list = []
        #Simulate sample
        for sample in tqdm(range(n_replications)):
            if simulation_distribution == 'GPD':
                random_sample_draw = stats.genpareto.rvs(loc = 0, c = shape_param, scale = scale_param, size = random_sample_size)   
            elif simulation_distribution == 'Pareto':
                random_sample_draw = stats.pareto.rvs(loc = 0, b = shape_param, scale = scale_param, size = random_sample_size)   
            elif simulation_distribution == 'Students-t':
                random_sample_draw = stats.t.rvs(loc = 0, df = scale_param, size = random_sample_size)   
            elif simulation_distribution == 'Weibull':
                uniform_RVs = stats.uniform.rvs(size = random_sample_size)
                random_sample_draw = scale_param*(-np.log(1-uniform_RVs))**(1/shape_param)
            elif simulation_distribution == 'Log-Normal':
                random_sample_draw = stats.lognorm.rvs(size = random_sample_size, s = scale_param) 
            elif simulation_distribution == 'Log-Logistic':
                uniform_RVs = stats.uniform.rvs(size = random_sample_size)
                random_sample_draw = uniform_RVs*(uniform_RVs/(1- uniform_RVs))**(1/shape_param)
            elif simulation_distribution == 'LogPH':
                random_sample_draw = np.array(log_PH_sample_generator(n = random_sample_size, alpha = alpha, T = T, processing_pool = processing_pool))
            mapping_list.append([random_sample_draw, random_sample_size, initial_threshold_choice, functional, functional_ranges, simulation_distribution, shape_param, scale_param])
        simulation_results = processing_pool.starmap(parameter_estimation_simulation_compute_functionals, mapping_list)
#        simulation_results = []
#        for i in tqdm(range(len(mapping_list))):
#            simulation_results.append(parameter_estimation_simulation_compute_functionals(random_sample_draw = np.array(mapping_list[i][0]), random_sample_size = mapping_list[i][1], initial_threshold_choice = mapping_list[i][2], functional = mapping_list[i][3], functional_ranges = mapping_list[i][4], simulation_distribution = mapping_list[i][5], shape_param = mapping_list[i][6], scale_param = mapping_list[i][7]))
        
        mean_shape_parameter_estimates = pd.DataFrame(columns = models, index = range(len(mapping_list)))
        mean_scale_parameter_estimates = pd.DataFrame(columns = models, index = range(len(mapping_list)))
        for i, result in tqdm(enumerate(simulation_results)):
            for functional_level in functional_ranges:
                for model in models + ['Truth']:                  
                    RMSE_frame_dict[model].at[i, '{} {}'.format(functional, functional_level)] = result[0][functional_level][model]
                    loss_frame_1_homogenous[model].at[i, '{} {}'.format(functional, functional_level)] = result[1][functional_level][model]
                    loss_frame_0_homogenous[model].at[i, '{} {}'.format(functional, functional_level)] = result[2][functional_level][model]
                    if functional == 'VaR':
                        loss_frame_standard_quantile[model].at[i, '{} {}'.format(functional, functional_level)] = result[3][functional_level][model]
        for i, result in enumerate(simulation_results):
            for model in models:             
                mean_shape_parameter_estimates.at[i, model] = simulation_results[i][4][model]['shape']
                mean_scale_parameter_estimates.at[i, model] = simulation_results[i][4][model]['scale']
        mean_shape_parameter_estimates['threshold'] = initial_threshold_choice
        mean_scale_parameter_estimates['threshold'] = initial_threshold_choice
        final_mean_shape_parameter_estimates.append(np.mean(mean_shape_parameter_estimates))
        final_mean_scale_parameter_estimates.append(np.mean(mean_scale_parameter_estimates))
               
        RMSE_error_frame['u'] = initial_threshold_choice
        loss_error_frame_1_homogenous['u'] = initial_threshold_choice
        loss_error_frame_0_homogenous['u'] = initial_threshold_choice
        if functional == 'VaR':
            loss_error_frame_standard_quantile['u'] = initial_threshold_choice
        for model in models + ['Truth']:
            RMSE_error_frame.iloc[RMSE_error_frame[RMSE_error_frame['Model'] == model].index,2:] = np.sqrt(np.mean(np.square(RMSE_frame_dict[model]))).values.T        
            loss_error_frame_1_homogenous.loc[loss_error_frame_1_homogenous[loss_error_frame_1_homogenous['Model'] == model].index, 2:] = np.mean(loss_frame_1_homogenous[model]).values.T
            loss_error_frame_0_homogenous.loc[loss_error_frame_0_homogenous[loss_error_frame_0_homogenous['Model'] == model].index, 2:] = np.mean(loss_frame_0_homogenous[model]).values.T
            if functional == 'VaR':
                loss_error_frame_standard_quantile.loc[loss_error_frame_standard_quantile[loss_error_frame_standard_quantile['Model'] == model].index, 2:] = np.mean(loss_frame_standard_quantile[model]).values.T
        RMSE_error_frame_list.append(RMSE_error_frame)
        loss_frame_list_1_homogenous.append(loss_error_frame_1_homogenous)
        loss_frame_list_0_homogenous.append(loss_error_frame_0_homogenous)
        loss_frame_list_standard_quantile.append(loss_error_frame_standard_quantile)

    final_mean_shape_parameter_estimates = pd.concat(final_mean_shape_parameter_estimates, axis = 0)
    final_mean_scale_parameter_estimates = pd.concat(final_mean_scale_parameter_estimates, axis = 0)

    all_threshold_error_frame = pd.concat(RMSE_error_frame_list, axis = 0)
    all_threshold_error_frame['Simulation Parameters'] = '{}({}, {})'.format(simulation_distribution, shape_param, scale_param)
    all_threshold_error_frame = all_threshold_error_frame[['Simulation Parameters'] + all_threshold_error_frame.columns.tolist()[:-1]]
    
    all_threshold_loss_frame_1_homogenous = pd.concat(loss_frame_list_1_homogenous, axis = 0)
    all_threshold_loss_frame_1_homogenous['Simulation Parameters'] = '{}({}, {})'.format(simulation_distribution, shape_param, scale_param)
    all_threshold_loss_frame_1_homogenous = all_threshold_loss_frame_1_homogenous[['Simulation Parameters'] + all_threshold_loss_frame_1_homogenous.columns.tolist()[:-1]]
    
    all_threshold_loss_frame_0_homogenous = pd.concat(loss_frame_list_0_homogenous, axis = 0)
    all_threshold_loss_frame_0_homogenous['Simulation Parameters'] = '{}({}, {})'.format(simulation_distribution, shape_param, scale_param)
    all_threshold_loss_frame_0_homogenous = all_threshold_loss_frame_0_homogenous[['Simulation Parameters'] + all_threshold_loss_frame_0_homogenous.columns.tolist()[:-1]]
        
    if functional == 'VaR':
        all_threshold_loss_frame_standard_quantile = pd.concat(loss_frame_list_standard_quantile, axis = 0)
        all_threshold_loss_frame_standard_quantile['Simulation Parameters'] = '{}({}, {})'.format(simulation_distribution, shape_param, scale_param)
        all_threshold_loss_frame_standard_quantile = all_threshold_loss_frame_standard_quantile[['Simulation Parameters'] + all_threshold_loss_frame_standard_quantile.columns.tolist()[:-1]]
        print('Completed {} {} {} {} n = {}'.format(simulation_distribution, functional, shape_param, scale_param, random_sample_size))
        return all_threshold_error_frame, all_threshold_loss_frame_1_homogenous, all_threshold_loss_frame_0_homogenous, all_threshold_loss_frame_standard_quantile, final_mean_shape_parameter_estimates, final_mean_scale_parameter_estimates
    else:
        all_threshold_loss_frame_standard_quantile = None
        print('Completed {} {} {} {} n = {}'.format(simulation_distribution, functional, shape_param, scale_param, random_sample_size))
        return all_threshold_error_frame, all_threshold_loss_frame_1_homogenous, all_threshold_loss_frame_0_homogenous

def global_variable_initializer():
    global n_replications, initial_threshold_choices, VaR_levels, models, CTE_levels, empirical_application_VaR_levels, penalized_MLE_lambda, penalized_MLE_alpha, empirical_application_initial_threshold_choices, VaR_initial_threshold_choices, CTE_initial_threshold_choices, optimization_bounds, VaR_murphy_ranges

    n_replications = 10000
    VaR_levels = [0.925, 0.95, 0.975, 0.99, 0.999, 0.9999]
    CTE_levels = [0.925, 0.95, 0.975, 0.99, 0.999]
    models = ['L-Moments', 'MLE', 'LME', 'NEW', 'Revised_NEW', 'MGF_AD', 'MGF_AD2R', 'WNLS', 'MDE']
    penalized_MLE_lambda = 1
    penalized_MLE_alpha = 1
    initial_threshold_choices = [100]
    optimization_bounds = [(0.001, 1), (0.001, np.inf)]
    VaR_murphy_ranges = {0.925:(0.92, 0.93), 0.95:(0.94, 0.96), 0.975:(0.965, 0.985), 0.99:(0.985, 0.995), 0.999:(0.99, 0.995), 0.9999:(0.999, 0.99995)}               

    
def main(distribution, shape, scale, functional, n):
    global n_replications, initial_threshold_choices, VaR_levels, models, CTE_levels, empirical_application_VaR_levels, penalized_MLE_lambda, penalized_MLE_alpha, empirical_application_initial_threshold_choices, VaR_initial_threshold_choices, CTE_initial_threshold_choices, optimization_bounds, VaR_murphy_ranges
    # =============================================================================
    #Initialize global variables
    global_variable_initializer()    
    if not os.path.exists(r'{}/Results/Simulation'.format(file_path)):
        os.makedirs(r'{}/Results/Simulation'.format(file_path))
    
    #Setup process pool and initialize with global variables
    num_processes = multiprocessing.cpu_count()
    processing_pool = multiprocessing.Pool(num_processes, global_variable_initializer)
    
    # =============================================================================
    # =============================================================================
    simulation_result = parameter_estimation_simulation(shape, scale, functional, distribution, processing_pool, n)
    if functional == 'VaR':
        quantile_loss_frame = simulation_result[3]
        quantile_loss_frame = assign_model_ranks(functional = functional, results_frame = quantile_loss_frame)
        mean_shape_parameter_estimates = simulation_result[4]
        mean_scale_parameter_estimates = simulation_result[5]

    RMSE_frame = simulation_result[0]
    score_0_homogenous_loss_list = simulation_result[2]
    score_1_homogenous_loss_list = simulation_result[1]
    
    #Assign Rankings
    RMSE_frame = assign_model_ranks(functional = functional, results_frame = RMSE_frame)
    score_0_homogenous_loss_frame = assign_model_ranks(functional = functional, results_frame = score_0_homogenous_loss_list)
    score_1_homogenous_loss_frame = assign_model_ranks(functional = functional, results_frame = score_1_homogenous_loss_list)
    if distribution in ['GPD', 'Students-t', 'Log-Normal', 'Log-Logistic', 'Pareto', 'Weibull']:
        if functional == 'VaR':
            quantile_loss_frame.to_csv(r'{}/Results/Simulation/{}_{}_quantile_loss_frame_n={}_shape={}_scale={}.csv'.format(file_path, distribution, functional, n, shape, scale), index = False)
            mean_shape_parameter_estimates.to_csv(r'{}/Results/Simulation/{}_mean_shape_param_n={}_shape={}_scale={}.csv'.format(file_path, distribution, n, shape, scale), index = True)
            mean_scale_parameter_estimates.to_csv(r'{}/Results/Simulation/{}_mean_scale_param_n={}_shape={}_scale={}.csv'.format(file_path, distribution, n, shape, scale), index = True)
        RMSE_frame.to_csv(r'{}/Results/Simulation/{}_{}_RMSE_error_frame_n={}_shape={}_scale={}.csv'.format(file_path, distribution, functional, n, shape, scale), index = False)
        score_0_homogenous_loss_frame.to_csv(r'{}/Results/Simulation/{}_{}_0_homogenous_loss_frame_n={}_shape={}_scale={}.csv'.format(file_path, distribution, functional, n, shape, scale), index = False)
        score_1_homogenous_loss_frame.to_csv(r'{}/Results/Simulation/{}_{}_1_homogenous_loss_frame_n={}_shape={}_scale={}.csv'.format(file_path, distribution, functional, n, shape, scale), index = False)
    elif distribution == 'LogPH':
        if functional == 'VaR':
            quantile_loss_frame.to_csv(r'{}/Results/Simulation/{}_{}_quantile_loss_frame_n={}.csv'.format(file_path, distribution, functional, n), index = False)
            mean_shape_parameter_estimates.to_csv(r'{}/Results/Simulation/{}_mean_shape_param_n={}.csv'.format(file_path, distribution, n), index = True)
            mean_scale_parameter_estimates.to_csv(r'{}/Results/Simulation/{}_mean_scale_param_n={}.csv'.format(file_path, distribution, n), index = True)
        RMSE_frame.to_csv(r'{}/Results/Simulation/{}_{}_RMSE_error_frame_n={}.csv'.format(file_path, distribution, functional, n), index = False)
        score_0_homogenous_loss_frame.to_csv(r'{}/Results/Simulation/{}_{}_0_homogenous_loss_frame_n={}.csv'.format(file_path, distribution, functional, n), index = False)
        score_1_homogenous_loss_frame.to_csv(r'{}/Results/Simulation/{}_{}_1_homogenous_loss_frame_n={}.csv'.format(file_path, distribution, functional, n), index = False)
    processing_pool.close()
    
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
    main(distribution = distribution, shape = shape, scale = scale, functional = functional, n = n)
