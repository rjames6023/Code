from arch.__future__ import reindexing
global file_path
file_path = r'C:\Users\Robert\Dropbox (Sydney Uni)\GPD_Paramaeter_Estimation_Comparison_Project'

import numpy as np
import pandas as pd
import os
import sys
sys.path.append(r'{}/code'.format(file_path)) 

from scipy import stats
from arch import arch_model
from tqdm import tqdm
from arch import bootstrap
from arch.bootstrap import MCS

import param_est_methods
import loss_funcs

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


def empirical_excercise_insurance_claims(all_claims, seed):
    VaR_dictionary = {model:{x:{VaR_level:[] for VaR_level in empirical_application_VaR_levels}
                             for x in empirical_application_initial_threshold_choices}
                      for model in models + ['convex_comb', 'min_patton_loss']}
    CTE_dictionary = {model:{x:{VaR_level:[] for VaR_level in empirical_application_VaR_levels}
                             for x in empirical_application_initial_threshold_choices}
                      for model in models + ['convex_comb', 'min_patton_loss']}

    final_mean_shape_parameter_estimates = []
    final_mean_scale_parameter_estimates = []

    for initial_threshold in empirical_application_initial_threshold_choices:
        shape_parameter_frame = pd.DataFrame(columns = models, index = [0])
        scale_parameter_frame = pd.DataFrame(columns = models, index = [0])
        
        #Create 5 data folds
        np.random.seed(seed)
        indicies = np.array([x for x in range(len(all_claims))])
        np.random.shuffle(indicies)
        fold_indicies = np.array_split(indicies, 5)
        out_sample_data = []
        for fold in fold_indicies:
            raw_insurance_data_insample = all_claims[np.delete(indicies, fold)]
            raw_insurance_data_outsample = all_claims[fold]
            out_sample_data.append(raw_insurance_data_outsample)
            
            threshold = np.percentile(raw_insurance_data_insample, 100*initial_threshold)
            tail_sample = raw_insurance_data_insample[raw_insurance_data_insample > threshold]
            excess_loss = tail_sample - threshold
            sorted_excess_loss = np.sort(excess_loss)
            
            #Construct weights vector for the WNLS estimator
            variances, weights_vector = param_est_methods.weights(nu = len(sorted_excess_loss), n = len(raw_insurance_data_insample))
            weights_vector = weights_vector[::-1]
            
            nu = len(tail_sample)
            n = len(raw_insurance_data_insample)
                
            #Fit GPD using different methods 
            parameter_estimation_dict = {model:{} for model in models}
            parameter_estimation_dict = param_est_methods.parameter_estimation(parameter_estimation_dict, raw_insurance_data_insample, sorted_excess_loss, excess_loss, tail_sample, initial_threshold, weights_vector, models = models, optimization_bounds = optimization_bounds)                

            #Convex combination
            conmvex_comb_res = convex_combination(parameter_estimation_dict={key:parameter_estimation_dict[key] for key in ['MLE', 'LME', 'Revised_NEW', 'WNLS']},
                               random_sample_draw=raw_insurance_data_insample,
                               initial_threshold=threshold,
                               excess_loss_sample=excess_loss,
                               functional_ranges=empirical_application_VaR_levels)

            min_patton_loss_results = min_patton_loss(random_sample_draw=raw_insurance_data_insample,
                                                      initial_threshold=threshold,
                                                      excess_loss_sample=excess_loss,
                                                      functional_ranges=empirical_application_VaR_levels)


            for model in models:
                shape_parameter_frame.at[0, model] = parameter_estimation_dict[model]['shape']
                scale_parameter_frame.at[0, model] = parameter_estimation_dict[model]['scale']
            shape_parameter_frame['threshold'] = initial_threshold
            scale_parameter_frame['threshold'] = initial_threshold
            final_mean_shape_parameter_estimates.append(shape_parameter_frame)
            final_mean_scale_parameter_estimates.append(scale_parameter_frame)
    
            for VaR_level in empirical_application_VaR_levels:
                for model in models:
                    if parameter_estimation_dict[model]['shape'] > 1:
                        parameter_estimation_dict[model]['shape'] = 0.99
                    if np.isnan(parameter_estimation_dict[model]['scale']) or np.isnan(parameter_estimation_dict[model]['shape']):
                        VaR = np.nan
                        CTE = np.nan
                    else:
                        VaR = threshold + (parameter_estimation_dict[model]['scale']/parameter_estimation_dict[model]['shape'])*(((n/nu)*(1-VaR_level))**(-parameter_estimation_dict[model]['shape']) - 1)
                        CTE = (VaR + parameter_estimation_dict[model]['scale'] - (parameter_estimation_dict[model]['shape'] * threshold))/(1 - parameter_estimation_dict[model]['shape'])                                
                    VaR_dictionary[model][initial_threshold][VaR_level].append(VaR)
                    CTE_dictionary[model][initial_threshold][VaR_level].append(CTE)

                #Convex comb results
                shape_estimates = np.array(
                    [parameter_estimation_dict[key]['shape'] for key in ['MLE', 'LME', 'Revised_NEW', 'WNLS']])
                scale_estimates = np.array(
                    [parameter_estimation_dict[key]['scale'] for key in ['MLE', 'LME', 'Revised_NEW', 'WNLS']])
                convex_comb_shape = conmvex_comb_res[VaR_level][:4] @ shape_estimates
                convex_comb_scale = conmvex_comb_res[VaR_level][4:] @ scale_estimates
                convex_comb_VaR_estimate = initial_threshold + (
                        convex_comb_scale / convex_comb_shape) * (
                                                   ((len(raw_insurance_data_insample) / len(excess_loss)) * (
                                                               1 - VaR_level)) ** (
                                                       -convex_comb_shape) - 1)
                if convex_comb_shape >= 1:
                    convex_comb_shape = 0.99
                CTE = (convex_comb_VaR_estimate + convex_comb_scale - (
                        convex_comb_shape * initial_threshold)) / (
                                              1 - convex_comb_shape)
                VaR_dictionary['convex_comb'][initial_threshold][VaR_level].append(convex_comb_VaR_estimate)
                CTE_dictionary['convex_comb'][initial_threshold][VaR_level].append(CTE)

                patton_loss_scale = min_patton_loss_results[VaR_level][1]
                patton_loss_shape = min_patton_loss_results[VaR_level][0]
                min_patton_loss_VaR_estimate = initial_threshold + (
                        patton_loss_scale / patton_loss_shape) * (
                                                       ((len(raw_insurance_data_insample) / len(excess_loss)) * (
                                                               1 - VaR_level)) ** (
                                                           -patton_loss_shape) - 1)
                if patton_loss_shape >= 1:
                    patton_loss_shape = 0.99
                CTE = (min_patton_loss_VaR_estimate + patton_loss_scale - (
                        patton_loss_shape * initial_threshold)) / (
                                              1 - patton_loss_shape)
                VaR_dictionary['min_patton_loss'][initial_threshold][VaR_level].append(min_patton_loss_VaR_estimate)
                CTE_dictionary['min_patton_loss'][initial_threshold][VaR_level].append(CTE)

    final_mean_shape_parameter_estimates = pd.concat(final_mean_shape_parameter_estimates, axis = 0)
    final_mean_scale_parameter_estimates = pd.concat(final_mean_scale_parameter_estimates, axis = 0)

    quantile_results_dict = {model:{initial_threshold: {VaR_level:0 for VaR_level in empirical_application_VaR_levels} for initial_threshold in empirical_application_initial_threshold_choices} for model in models}
    results_0_homogenous = {model:{initial_threshold: {VaR_level:0 for VaR_level in empirical_application_VaR_levels} for initial_threshold in empirical_application_initial_threshold_choices} for model in models}
    CTE_0_homogenous = {model:{initial_threshold: {VaR_level:0 for VaR_level in empirical_application_VaR_levels} for initial_threshold in empirical_application_initial_threshold_choices}
                        for model in models + ['convex_comb', 'min_patton_loss']}

    for initial_threshold in empirical_application_initial_threshold_choices:
        for VaR_level in empirical_application_VaR_levels:
            for model in models:
                standard_quantile_losses = []
                losses_0_homogenous = []
                CTE_losses_0_homogenous = []
                for i in range(len(fold_indicies)):
                    standard_quantile_losses.append(loss_funcs.quantile_loss(r = np.array(VaR_dictionary[model][initial_threshold][VaR_level][i]), x = out_sample_data[i], alpha = VaR_level))
                    losses_0_homogenous.append(loss_funcs.tick_loss_0_homogenous(r = np.array(VaR_dictionary[model][initial_threshold][VaR_level][i]), x = out_sample_data[i], alpha = VaR_level))
                    #CTE Loss
                    CTE_losses_0_homogenous.append(loss_funcs.joint_VaR_CTE_patton_0_homogenous(r1 = np.array(VaR_dictionary[model][initial_threshold][VaR_level][i]), r2 = np.array(CTE_dictionary[model][initial_threshold][VaR_level][i]), x = out_sample_data[i], alpha = VaR_level))

                quantile_results_dict[model][initial_threshold][VaR_level] = np.mean(standard_quantile_losses)
                results_0_homogenous[model][initial_threshold][VaR_level] = np.mean(losses_0_homogenous)
                                
                CTE_0_homogenous[model][initial_threshold][VaR_level] = np.mean(CTE_losses_0_homogenous)

            convex_comb_CTE_losses_0_homogenous = []
            min_patton_loss_CTE_losses_0_homogenous = []
            for i in range(len(fold_indicies)):
                convex_comb_CTE_losses_0_homogenous.append(loss_funcs.joint_VaR_CTE_patton_0_homogenous(
                    r1=np.array(VaR_dictionary['convex_comb'][initial_threshold][VaR_level][i]),
                    r2=np.array(CTE_dictionary['convex_comb'][initial_threshold][VaR_level][i]), x=out_sample_data[i],
                    alpha=VaR_level))
                min_patton_loss_CTE_losses_0_homogenous.append(loss_funcs.joint_VaR_CTE_patton_0_homogenous(
                    r1=np.array(VaR_dictionary['min_patton_loss'][initial_threshold][VaR_level][i]),
                    r2=np.array(CTE_dictionary['min_patton_loss'][initial_threshold][VaR_level][i]), x=out_sample_data[i],
                    alpha=VaR_level))
            CTE_0_homogenous['convex_comb'][initial_threshold][VaR_level] = np.mean(convex_comb_CTE_losses_0_homogenous)
            CTE_0_homogenous['min_patton_loss'][initial_threshold][VaR_level] = np.mean(min_patton_loss_CTE_losses_0_homogenous)

    return quantile_results_dict, results_0_homogenous, CTE_0_homogenous, final_mean_shape_parameter_estimates, final_mean_scale_parameter_estimates
                
def assign_model_ranks_empirical_analysis(empirical_application_results, models):
    empirical_application_results_list = []
    for threshold in empirical_application_initial_threshold_choices:
        results_list = []
        i = 0
        for VaR_level in empirical_application_VaR_levels:
            results_data = empirical_application_results[(empirical_application_results['Threshold'] == threshold) & (np.round(empirical_application_results['VaR Level'].astype(float),3) == VaR_level)][models]
            results_df = pd.DataFrame(data = {'Model':results_data.T.index.tolist(), 'Threshold': [threshold for x in range(len(results_data.T))], VaR_level: results_data.T.values.ravel().tolist()})
            results_df['{} Rank'.format(VaR_level)] = results_df[VaR_level].rank()
            if i == 0:
                results_list.append(results_df)
            else:
                results_list.append(results_df[[VaR_level, '{} Rank'.format(VaR_level)]])
            i +=1
        threshold_results = pd.concat(results_list, axis = 1)
        empirical_application_results_list.append(threshold_results)
    final_empirical_application_results = pd.concat(empirical_application_results_list, axis = 0)
    return final_empirical_application_results

def financial_risk_forecasting_application(index_data_dict, data_end_date, training_data_length, initial_threshold):
    if data_end_date is not None:
        index_data_dict = {stock:index_data_dict[stock][index_data_dict[stock]['date'] <= data_end_date] for stock in list(index_data_dict.keys())}
    
    quantile_results_dict = {model:{initial_threshold: {VaR_level:[] for VaR_level in empirical_application_VaR_levels} for initial_threshold in empirical_application_initial_threshold_choices} for model in models}
    quantile_all_model_MCS_rejections_count = {initial_threshold: pd.DataFrame(0, columns=empirical_application_VaR_levels, index=models, dtype=int) for initial_threshold in empirical_application_initial_threshold_choices}
    results_0_homogenous = {model:{initial_threshold: {VaR_level:[] for VaR_level in empirical_application_VaR_levels} for initial_threshold in empirical_application_initial_threshold_choices} for model in models}
    quantile_0_homogenous_MCS_rejections_count = {initial_threshold: pd.DataFrame(0, columns=empirical_application_VaR_levels, index=models, dtype=int) for initial_threshold in empirical_application_initial_threshold_choices}
    CTE_0_homogenous = {model:{initial_threshold: {VaR_level:[] for VaR_level in empirical_application_VaR_levels} for initial_threshold in empirical_application_initial_threshold_choices}
                        for model in models + ['convex_comb', 'min_patton_loss']}
    CTE_0_homogenous_MCS_rejections_count = {initial_threshold: pd.DataFrame(0, columns=empirical_application_VaR_levels, index=models, dtype=int) for initial_threshold in empirical_application_initial_threshold_choices}
    failed_optimization_proportion = {initial_threshold:{model:0 for model in models} for initial_threshold in empirical_application_initial_threshold_choices}

    mean_shape_parameter_estimates = {}
    mean_scale_parameter_estimates = {}
    for stock in tqdm(index_data_dict.keys()):
        quantile_all_model_loss_df_dict = {initial_threshold: {VaR_level: pd.DataFrame(columns=models) for VaR_level in
                                                               empirical_application_VaR_levels} for initial_threshold
                                           in empirical_application_initial_threshold_choices}
        quantile_0_homogenous_all_model_loss_df_dict = {
            initial_threshold: {VaR_level: pd.DataFrame(columns=models) for VaR_level in
                                empirical_application_VaR_levels} for initial_threshold in
            empirical_application_initial_threshold_choices}
        CTE_0_homogenous_all_model_loss_df_dict = {
            initial_threshold: {VaR_level: pd.DataFrame(columns=models) for VaR_level in
                                empirical_application_VaR_levels} for initial_threshold in
            empirical_application_initial_threshold_choices}

        VaR_dictionary = {model:{x:{VaR_level:[] for VaR_level in empirical_application_VaR_levels} for x in empirical_application_initial_threshold_choices} for model in models + ['convex_comb', 'min_patton_loss']}
        CTE_dictionary = {model:{x:{VaR_level:[] for VaR_level in empirical_application_VaR_levels} for x in empirical_application_initial_threshold_choices} for model in models + ['convex_comb', 'min_patton_loss']}
        shape_dict = {model:[] for model in models}
        scale_dict = {model:[]  for model in models}
        verifying_obs = []
        stock_data = index_data_dict[stock]
        for i, prediction_index in enumerate(range(training_data_length, len(stock_data)-1)):
            training_data = stock_data.iloc[i:i+training_data_length, -1]*100
            #Select the 1-day ahead testing data
            testing_data = stock_data.iloc[prediction_index, -1]*-100 #negative 1-day ahead return
            verifying_obs.append(testing_data)
            #Fit AR(1)-GJR-GARCH(1,1) model
            GARCH_model = arch_model(training_data.values, mean = 'ARX', lags = 1, vol = 'Garch', p = 1, o = 1, q = 1, dist = 'studentst')
            GARCH_model_fit = GARCH_model.fit(disp = 'off')
            GARCH_resids = ((GARCH_model_fit.resid[1:])/(GARCH_model_fit.conditional_volatility[1:]))*-1 #negated standardized residuals
            forecasts = GARCH_model_fit.forecast(horizon = 1)
            one_step_conditional_mean = forecasts.mean.loc[training_data_length-1, 'h.1']
            one_step_ahead_variance = np.sqrt(forecasts.variance.loc[training_data_length-1, 'h.1'])
    
            #GPD fit
            threshold = np.percentile(GARCH_resids, 100*initial_threshold)
            tail_sample = GARCH_resids[GARCH_resids > threshold]
            excess_loss = tail_sample - threshold
            sorted_excess_loss = np.sort(excess_loss)
            
            #Construct weights vector for the WNLS estimator
            variances, weights_vector = param_est_methods.weights(nu = len(sorted_excess_loss), n = len(GARCH_resids))
            weights_vector = weights_vector[::-1]
            
            nu = len(tail_sample)
            n = len(GARCH_resids)
                
            #Fit GPD using different methods 
            parameter_estimation_dict = {model:{} for model in models}
            parameter_estimation_dict = param_est_methods.parameter_estimation(parameter_estimation_dict = parameter_estimation_dict, full_sample = GARCH_resids, sorted_excess_loss = sorted_excess_loss, excess_loss = excess_loss, tail_sample = tail_sample, initial_threshold = initial_threshold, weights_vector = weights_vector, models = models, optimization_bounds = optimization_bounds)                

            #Convex combination
            conmvex_comb_res = convex_combination(parameter_estimation_dict={key:parameter_estimation_dict[key] for key in ['MLE', 'LME', 'Revised_NEW', 'WNLS']},
                               random_sample_draw=GARCH_resids,
                               initial_threshold=threshold,
                               excess_loss_sample=excess_loss,
                               functional_ranges=empirical_application_VaR_levels)

            min_patton_loss_results = min_patton_loss(random_sample_draw=GARCH_resids,
                                                      initial_threshold=threshold,
                                                      excess_loss_sample=excess_loss,
                                                      functional_ranges=empirical_application_VaR_levels)

            for model in models:
                for initial_threshold in empirical_application_initial_threshold_choices:
                    shape_dict[model].append(parameter_estimation_dict[model]['shape'])
                    scale_dict[model].append(parameter_estimation_dict[model]['scale'])
                    
            for VaR_level in empirical_application_VaR_levels:
                for model in models:
                    if parameter_estimation_dict[model]['shape'] > 1:
                        parameter_estimation_dict[model]['shape'] = 0.99
                    if np.isnan(parameter_estimation_dict[model]['scale']) or np.isnan(parameter_estimation_dict[model]['shape']):
                        VaR = np.nan
                        CTE = np.nan
                    else:
                        VaR_ = threshold + (parameter_estimation_dict[model]['scale']/parameter_estimation_dict[model]['shape'])*(((n/nu)*(1-VaR_level))**(-parameter_estimation_dict[model]['shape']) - 1)
                        VaR = one_step_conditional_mean + one_step_ahead_variance*VaR_
                        
                        CTE_ = (1/(1 - parameter_estimation_dict[model]['shape'])) + ((parameter_estimation_dict[model]['scale'] - parameter_estimation_dict[model]['shape']*threshold)/((1 - parameter_estimation_dict[model]['shape'])*VaR_))
                        CTE = one_step_conditional_mean + one_step_ahead_variance*VaR_*CTE_
                
                    VaR_dictionary[model][initial_threshold][VaR_level].append(VaR)
                    CTE_dictionary[model][initial_threshold][VaR_level].append(CTE)

                # Convex comb results
                shape_estimates = np.array(
                    [parameter_estimation_dict[key]['shape'] for key in ['MLE', 'LME', 'Revised_NEW', 'WNLS']])
                scale_estimates = np.array(
                    [parameter_estimation_dict[key]['scale'] for key in ['MLE', 'LME', 'Revised_NEW', 'WNLS']])
                convex_comb_shape = conmvex_comb_res[VaR_level][:4] @ shape_estimates
                convex_comb_scale = conmvex_comb_res[VaR_level][4:] @ scale_estimates
                convex_comb_VaR_estimate = threshold + (convex_comb_scale/convex_comb_shape)*(((n/nu)*(1-VaR_level))**(-convex_comb_shape) - 1)
                VaR = one_step_conditional_mean + one_step_ahead_variance * convex_comb_VaR_estimate

                CTE_ = (1 / (1 - convex_comb_shape)) + ((convex_comb_scale -convex_comb_shape * threshold) / ((1 -convex_comb_shape) * VaR_))
                CTE = one_step_conditional_mean + one_step_ahead_variance * convex_comb_VaR_estimate * CTE_

                VaR_dictionary['convex_comb'][initial_threshold][VaR_level].append(VaR)
                CTE_dictionary['convex_comb'][initial_threshold][VaR_level].append(CTE)

                patton_loss_scale = min_patton_loss_results[VaR_level][1]
                patton_loss_shape = min_patton_loss_results[VaR_level][0]
                patton_loss_VaR_estimate = threshold + (patton_loss_scale/patton_loss_shape)*(((n/nu)*(1-VaR_level))**(-patton_loss_shape) - 1)
                VaR = one_step_conditional_mean + one_step_ahead_variance * patton_loss_VaR_estimate

                CTE_ = (1 / (1 - patton_loss_shape)) + ((patton_loss_scale -patton_loss_shape * threshold) / ((1 -patton_loss_shape) * VaR_))
                CTE = one_step_conditional_mean + one_step_ahead_variance * patton_loss_VaR_estimate * CTE_
                VaR_dictionary['min_patton_loss'][initial_threshold][VaR_level].append(VaR)
                CTE_dictionary['min_patton_loss'][initial_threshold][VaR_level].append(CTE)

        for initial_threshold in empirical_application_initial_threshold_choices:
            for VaR_level in empirical_application_VaR_levels:
                for model in models:
                    standard_quantile_loss = loss_funcs.quantile_loss(r = np.array(VaR_dictionary[model][initial_threshold][VaR_level]), x = np.array(verifying_obs), alpha = VaR_level)
                    loss_0_homogenous = loss_funcs.tick_loss_0_homogenous(r = np.array(VaR_dictionary[model][initial_threshold][VaR_level]), x = np.array(verifying_obs), alpha = VaR_level)
    
                    quantile_results_dict[model][initial_threshold][VaR_level].append(standard_quantile_loss)
                    quantile_all_model_loss_df_dict[initial_threshold][VaR_level][model] = loss_funcs.point_quantile_loss(r = np.array(VaR_dictionary[model][initial_threshold][VaR_level]),
                                                                                                                          x = np.array(verifying_obs),
                                                                                                                          alpha = VaR_level)
                    results_0_homogenous[model][initial_threshold][VaR_level].append(loss_0_homogenous)
                    quantile_0_homogenous_all_model_loss_df_dict[initial_threshold][VaR_level][model] = loss_funcs.point_tick_loss_0_homogenous(r = np.array(VaR_dictionary[model][initial_threshold][VaR_level]),
                                                                                                                          x = np.array(verifying_obs),
                                                                                                                          alpha = VaR_level)

                    #CTE Loss
                    CTE_loss_0_homogenous = loss_funcs.joint_VaR_CTE_patton_0_homogenous(r1 = np.array(VaR_dictionary[model][initial_threshold][VaR_level]),
                                                                                         r2 = np.array(CTE_dictionary[model][initial_threshold][VaR_level]),
                                                                                         x = np.array(verifying_obs), alpha = VaR_level)
                    CTE_0_homogenous_all_model_loss_df_dict[initial_threshold][VaR_level][model] = loss_funcs.point_joint_VaR_CTE_patton_0_homogenous(r1 = np.array(VaR_dictionary[model][initial_threshold][VaR_level]),
                                                                                         r2 = np.array(CTE_dictionary[model][initial_threshold][VaR_level]),
                                                                                         x = np.array(verifying_obs), alpha = VaR_level)
                    CTE_0_homogenous[model][initial_threshold][VaR_level].append(CTE_loss_0_homogenous)

                convex_comb_CTE_loss_0_homogenous = loss_funcs.joint_VaR_CTE_patton_0_homogenous(
                    r1=np.array(VaR_dictionary['convex_comb'][initial_threshold][VaR_level]),
                    r2=np.array(CTE_dictionary['convex_comb'][initial_threshold][VaR_level]),
                    x=np.array(verifying_obs), alpha=VaR_level)
                CTE_0_homogenous['convex_comb'][initial_threshold][VaR_level].append(convex_comb_CTE_loss_0_homogenous)

                min_patton_loss_CTE_loss_0_homogenous = loss_funcs.joint_VaR_CTE_patton_0_homogenous(
                    r1=np.array(VaR_dictionary['min_patton_loss'][initial_threshold][VaR_level]),
                    r2=np.array(CTE_dictionary['min_patton_loss'][initial_threshold][VaR_level]),
                    x=np.array(verifying_obs), alpha=VaR_level)
                CTE_0_homogenous['min_patton_loss'][initial_threshold][VaR_level].append(min_patton_loss_CTE_loss_0_homogenous)

        for model in models:
            mean_shape_parameter_estimates[model] = np.mean(shape_dict[model])
            mean_scale_parameter_estimates[model] = np.mean(scale_dict[model])

        # MCS
        # optimal_block = bootstrap.optimal_block_length(stock_data['open_to_close'].values)
        # for initial_threshold in empirical_application_initial_threshold_choices:
        #     for VaR_level in empirical_application_VaR_levels:
        #         quantile_all_model_loss_df_dict[initial_threshold][VaR_level] = quantile_all_model_loss_df_dict[initial_threshold][VaR_level].fillna(method='ffill')
        #         quantile_0_homogenous_all_model_loss_df_dict[initial_threshold][VaR_level] = quantile_0_homogenous_all_model_loss_df_dict[initial_threshold][VaR_level].fillna(method='ffill')
        #         CTE_0_homogenous_all_model_loss_df_dict[initial_threshold][VaR_level] = CTE_0_homogenous_all_model_loss_df_dict[initial_threshold][VaR_level].fillna(method='ffill')
        #
        #         MCS_instance = MCS(losses=quantile_all_model_loss_df_dict[initial_threshold][VaR_level].fillna(method='ffill'),
        #                            size=0.9,
        #                            block_size=optimal_block.at[0, 'stationary'])
        #         MCS_instance.compute()
        #         for model in models:
        #             quantile_all_model_MCS_rejections_count[initial_threshold].loc[model, VaR_level] += (MCS_instance.pvalues < 0.05).astype(int).loc[model, 'Pvalue']
        #
        #         MCS_instance = MCS(losses=quantile_0_homogenous_all_model_loss_df_dict[initial_threshold][VaR_level],
        #                            size=0.9,
        #                            block_size=optimal_block.at[0, 'stationary'])
        #         MCS_instance.compute()
        #         for model in models:
        #             quantile_0_homogenous_MCS_rejections_count[initial_threshold].loc[model, VaR_level] += (MCS_instance.pvalues < 0.05).astype(int).loc[model, 'Pvalue']
        #
        #         MCS_instance = MCS(losses=CTE_0_homogenous_all_model_loss_df_dict[initial_threshold][VaR_level],
        #                            size=0.9,
        #                            block_size=optimal_block.at[0, 'stationary'])
        #         MCS_instance.compute()
        #         for model in models:
        #             CTE_0_homogenous_MCS_rejections_count[initial_threshold].loc[model, VaR_level] += (MCS_instance.pvalues < 0.05).astype(int).loc[model, 'Pvalue']

    return quantile_results_dict, results_0_homogenous, CTE_0_homogenous, failed_optimization_proportion, mean_shape_parameter_estimates, mean_scale_parameter_estimates, quantile_all_model_MCS_rejections_count, quantile_0_homogenous_MCS_rejections_count, CTE_0_homogenous_MCS_rejections_count

def global_variable_initializer():
    global initial_threshold_choices, VaR_levels, models, CTE_levels, empirical_application_VaR_levels, simulation_optimization_bounds, optimization_bounds, penalized_MLE_lambda, penalized_MLE_alpha, empirical_application_initial_threshold_choices, VaR_initial_threshold_choices, CTE_initial_threshold_choices, stock_indicies

    empirical_application_initial_threshold_choices = [0.90, 0.95]
    VaR_levels = [0.925, 0.95, 0.975, 0.99, 0.999, 0.9999]
    CTE_levels = [0.925, 0.95, 0.975, 0.99, 0.999, 0.9999]
    empirical_application_VaR_levels = [0.975, 0.99, 0.995, 0.999]
    models = ['L-Moments', 'MLE', 'LME', 'NEW', 'Revised_NEW', 'MGF_AD', 'MGF_AD2R', 'WNLS', 'MDE']
    optimization_bounds = [(0.001, 1), (0.001, np.inf)]
    penalized_MLE_lambda = 1
    penalized_MLE_alpha = 1
    VaR_initial_threshold_choices = [0.90, 0.95]
    CTE_initial_threshold_choices = [0.90, 0.95]
    stock_indicies = ['.AEX', '.AORD', '.DJI', '.FCHI', '.FTSE', '.HSI', '.IXIC', '.N225', '.SPX']
    
def main():
    # =============================================================================
    #Initialize global variables
    global_variable_initializer()
        
    if not os.path.exists(r'{}/Results/Application'.format(file_path)):
        os.makedirs(r'{}/Results/Application'.format(file_path))
    summary_statistics = pd.DataFrame(index = ['N', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Skewness', 'Kurtosis', 'Tail Index'], columns = ['Building and Contents Insurance', 'SOA Insurance'])
    
    # =============================================================================
    # Empicial Excercise 1: VaR and ES estimation for auto Insurance claims data
    # =============================================================================
    empirical_application_initial_threshold_choices = [0.9]
    claims_choice = 'claimbc' #Total collision claims from old vehicles in year.
    claims_insample = pd.read_csv(r'{}/Data/claims_data_insample.csv'.format(file_path))
    claims_insample.columns = map(str.lower, claims_insample.columns)
    claims_insample = claims_insample[claims_choice]/1e4
    claims_insample = claims_insample[claims_insample > 0]
    
    claims_outsample = pd.read_csv(r'{}/Data/claims_data_outsample.csv'.format(file_path))
    claims_outsample.columns = map(str.lower, claims_outsample.columns)
    claims_outsample = claims_outsample[claims_choice]/1e4
    claims_outsample = claims_outsample[claims_outsample > 0]

    #Summary statistics
    all_claims = pd.concat([claims_insample, claims_outsample], axis = 0) 
    all_claims = all_claims[all_claims > 0]
    all_claims.to_csv(r'{}/Data/Building and Contents_insurance_claims.csv'.format(file_path), index = False, header = True)
    summary_statistics.loc['N', 'Building and Contents Insurance'] = len(all_claims)
    summary_statistics.loc['Mean', 'Building and Contents Insurance'] = np.mean(all_claims)
    summary_statistics.loc['Median', 'Building and Contents Insurance'] = np.median(all_claims)
    summary_statistics.loc['Std', 'Building and Contents Insurance'] = np.std(all_claims)
    summary_statistics.loc['Min', 'Building and Contents Insurance'] = np.min(all_claims)
    summary_statistics.loc['Max', 'Building and Contents Insurance'] = np.max(all_claims)
    summary_statistics.loc['Skewness', 'Building and Contents Insurance'] = stats.skew(all_claims)
    summary_statistics.loc['Kurtosis', 'Building and Contents Insurance'] = stats.kurtosis(all_claims)
    upper_order_statistics = np.sort(all_claims[all_claims > np.percentile(all_claims, 90)].values)[::-1]
    hill = (np.mean(np.log(upper_order_statistics) - np.log(upper_order_statistics[-1])))**-1
    summary_statistics.loc['Tail Index', 'Building and Contents Insurance'] = hill
    
    insurance_quantile_results_dict, insurance_results_0_homogenous, insurance_CTE_0_homogenous, final_mean_shape_parameter_estimates, final_mean_scale_parameter_estimates = empirical_excercise_insurance_claims(all_claims = all_claims.values, seed = 555)

    VaR_empirical_application_results_insurance_test = pd.DataFrame(columns = ['Threshold', 'VaR Level'] + models)
    VaR_empirical_application_results_0_homogenous_insurance_test = pd.DataFrame(columns = ['Threshold', 'VaR Level'] + models)
    CTE_empirical_application_results_0_homogenous_insurance_test = pd.DataFrame(columns = ['Threshold', 'VaR Level'] + models + ['convex_comb', 'min_patton_loss'])
    index_counter = 0
    for initial_threshold in empirical_application_initial_threshold_choices: 
        for VaR_level in empirical_application_VaR_levels:
            VaR_empirical_application_results_insurance_test.loc[index_counter, 'Threshold'] = initial_threshold
            VaR_empirical_application_results_insurance_test.loc[index_counter, 'VaR Level'] = VaR_level
            VaR_empirical_application_results_0_homogenous_insurance_test.loc[index_counter, 'Threshold'] = initial_threshold
            VaR_empirical_application_results_0_homogenous_insurance_test.loc[index_counter, 'VaR Level'] = VaR_level
            CTE_empirical_application_results_0_homogenous_insurance_test.loc[index_counter, 'Threshold'] = initial_threshold
            CTE_empirical_application_results_0_homogenous_insurance_test.loc[index_counter, 'VaR Level'] = VaR_level
            for model in models:
                VaR_empirical_application_results_insurance_test.loc[index_counter, model] = np.mean(insurance_quantile_results_dict[model][initial_threshold][VaR_level]) #/1e6
                VaR_empirical_application_results_0_homogenous_insurance_test.loc[index_counter, model] = np.mean(insurance_results_0_homogenous[model][initial_threshold][VaR_level])
            for model in models + ['convex_comb', 'min_patton_loss']:
                CTE_empirical_application_results_0_homogenous_insurance_test.loc[index_counter, model] = np.mean(insurance_CTE_0_homogenous[model][initial_threshold][VaR_level])
            index_counter += 1
            
    #Assign Ranks
    final_VaR_empirical_application_results_insurance_test = assign_model_ranks_empirical_analysis(VaR_empirical_application_results_insurance_test, models=models)
    final_VaR_empirical_application_results_0_homogenous_insurance_test = assign_model_ranks_empirical_analysis(VaR_empirical_application_results_0_homogenous_insurance_test, models=models)
    final_CTE_empirical_application_results_0_homogenous_insurance_test = assign_model_ranks_empirical_analysis(CTE_empirical_application_results_0_homogenous_insurance_test, models=models + ['convex_comb', 'min_patton_loss'])
    
    final_mean_shape_parameter_estimates.to_csv(r'{}/Results/Application/shape_parameter_insurance_test.csv'.format(file_path), index = False)
    final_mean_scale_parameter_estimates.to_csv(r'{}/Results/Application/scale_parameter_insurance_test.csv'.format(file_path), index = False)
    final_VaR_empirical_application_results_insurance_test.to_csv(r'{}/Results/Application/VaR_empirical_application_results_insurance_test.csv'.format(file_path), index = False)
    final_VaR_empirical_application_results_0_homogenous_insurance_test.to_csv(r'{}/Results/Application/VaR_empirical_application_results_0_homogenous_Building and Contents_insurance_test.csv'.format(file_path), index = False)
    final_CTE_empirical_application_results_0_homogenous_insurance_test.to_csv(r'{}/Results/Application/CTE_empirical_application_results_0_homogenous_Building and Contents_insurance_test.csv'.format(file_path), index = False)

    empirical_application_initial_threshold_choices = [0.95]
    FIRE_claims = (pd.read_csv(r'{}/Data/norwegianfire.csv'.format(file_path), usecols =['claim_size']).values/1e3).ravel()    
    #Summary statistics
    summary_statistics.loc['N', 'FIRE Insurance'] = len(FIRE_claims)
    summary_statistics.loc['Mean', 'FIRE Insurance'] = np.mean(FIRE_claims)
    summary_statistics.loc['Median', 'FIRE Insurance'] = np.median(FIRE_claims)
    summary_statistics.loc['Std', 'FIRE Insurance'] = np.std(FIRE_claims)
    summary_statistics.loc['Min', 'FIRE Insurance'] = np.min(FIRE_claims)
    summary_statistics.loc['Max', 'FIRE Insurance'] = np.max(FIRE_claims)
    summary_statistics.loc['Skewness', 'FIRE Insurance'] = stats.skew(FIRE_claims)
    summary_statistics.loc['Kurtosis', 'FIRE Insurance'] = stats.kurtosis(FIRE_claims)
    upper_order_statistics = np.sort(FIRE_claims[FIRE_claims > np.percentile(FIRE_claims, 95)])[::-1]
    hill = np.mean(np.log(upper_order_statistics) - np.log(upper_order_statistics[-1]))**-1
    summary_statistics.loc['Tail Index', 'FIRE Insurance'] = hill

    FIRE_claims_quantile_results_dict, FIRE_claims_results_0_homogenous, FIRE_claims_CTE_0_homogenous, FIRE_final_mean_shape_parameter_estimates, FIRE_final_mean_scale_parameter_estimates = empirical_excercise_insurance_claims(FIRE_claims, seed = 10)
    
    VaR_empirical_application_results_FIRE_claims_test = pd.DataFrame(columns = ['Threshold', 'VaR Level'] + models)
    VaR_empirical_application_results_0_homogenous_FIRE_claims_test = pd.DataFrame(columns = ['Threshold', 'VaR Level'] + models)
    CTE_empirical_application_results_0_homogenous_FIRE_claims_test = pd.DataFrame(columns = ['Threshold', 'VaR Level'] + models + ['convex_comb', 'min_patton_loss'])
    index_counter = 0
    for initial_threshold in empirical_application_initial_threshold_choices: 
        for VaR_level in empirical_application_VaR_levels:
            VaR_empirical_application_results_FIRE_claims_test.loc[index_counter, 'Threshold'] = initial_threshold
            VaR_empirical_application_results_FIRE_claims_test.loc[index_counter, 'VaR Level'] = VaR_level
            VaR_empirical_application_results_0_homogenous_FIRE_claims_test.loc[index_counter, 'Threshold'] = initial_threshold
            VaR_empirical_application_results_0_homogenous_FIRE_claims_test.loc[index_counter, 'VaR Level'] = VaR_level
            CTE_empirical_application_results_0_homogenous_FIRE_claims_test.loc[index_counter, 'Threshold'] = initial_threshold
            CTE_empirical_application_results_0_homogenous_FIRE_claims_test.loc[index_counter, 'VaR Level'] = VaR_level
            for model in models:
                VaR_empirical_application_results_FIRE_claims_test.loc[index_counter, model] = np.mean(FIRE_claims_quantile_results_dict[model][initial_threshold][VaR_level]) #/1e6
                VaR_empirical_application_results_0_homogenous_FIRE_claims_test.loc[index_counter, model] = np.mean(FIRE_claims_results_0_homogenous[model][initial_threshold][VaR_level])
            for model in models + ['convex_comb', 'min_patton_loss']:
                CTE_empirical_application_results_0_homogenous_FIRE_claims_test.loc[index_counter, model] = np.mean(FIRE_claims_CTE_0_homogenous[model][initial_threshold][VaR_level])
            index_counter += 1
            
    #Assign Ranks
    final_VaR_empirical_application_results_FIRE_claims_test = assign_model_ranks_empirical_analysis(VaR_empirical_application_results_FIRE_claims_test, models=models)
    final_VaR_empirical_application_results_0_homogenous_FIRE_claims_test = assign_model_ranks_empirical_analysis(VaR_empirical_application_results_0_homogenous_FIRE_claims_test, models=models)
    final_CTE_empirical_application_results_0_homogenous_FIRE_claims_test = assign_model_ranks_empirical_analysis(CTE_empirical_application_results_0_homogenous_FIRE_claims_test, models=models + ['convex_comb', 'min_patton_loss'])
    
    FIRE_final_mean_shape_parameter_estimates.to_csv(r'{}/Results/Application/FIRE_shape_parameter_insurance_test.csv'.format(file_path), index = False)
    FIRE_final_mean_scale_parameter_estimates.to_csv(r'{}/Results/Application/FIRE_scale_parameter_insurance_test.csv'.format(file_path), index = False)
    final_VaR_empirical_application_results_FIRE_claims_test.to_csv(r'{}/Results/Application/VaR_empirical_application_results_FIRE_claims_test.csv'.format(file_path), index = False)
    final_VaR_empirical_application_results_0_homogenous_FIRE_claims_test.to_csv(r'{}/Results/Application/VaR_empirical_application_results_0_homogenous_FIRE_claims_test.csv'.format(file_path), index = False)
    final_CTE_empirical_application_results_0_homogenous_FIRE_claims_test.to_csv(r'{}/Results/Application/CTE_empirical_application_results_0_homogenous_FIRE_claims_test.csv'.format(file_path), index = False)

    #Export summary statistics
    summary_statistics.to_csv(r'{}/Results/Application/Insurance_Summary_Statistics.csv'.format(file_path), index = True)
    
    # =============================================================================
    # Stock index risk forecasting     
    # =============================================================================
    #Parameters
    global optimization_bounds 
    optimization_bounds = [(-0.1, 1), (0.001, np.inf)]
    empirical_application_initial_threshold_choices = [0.9]
    failed_optimization_proportion = {initial_threshold:{model:0 for model in models} for initial_threshold in empirical_application_initial_threshold_choices}

    index_data_raw = pd.read_csv(r'{}/Data/oxfordmanrealizedvolatilityindices.csv'.format(file_path), parse_dates = ['date'])
    index_data_raw.columns = map(str.lower, index_data_raw.columns)
    index_data_raw = index_data_raw[index_data_raw['symbol'].isin(stock_indicies)]
    index_data_raw = index_data_raw[['date', 'symbol', 'open_to_close']]
    index_data_raw.sort_values(by = ['symbol', 'date'], inplace = True)
    index_data_dict = {stock:index_data_raw[index_data_raw['symbol'] == stock] for stock in stock_indicies}

    (quantile_results_dict,
    results_0_homogenous,
    CTE_0_homogenous,
    failed_optimization_proportion,
    mean_shape_parameter_estimates,
    mean_scale_parameter_estimates,
     quantile_all_model_MCS_rejections_count,
     quantile_0_homogenous_MCS_rejections_count,
     CTE_0_homogenous_MCS_rejections_count) = financial_risk_forecasting_application(index_data_dict = index_data_dict,
                                                                             training_data_length = 1000,
                                                                             data_end_date = pd.to_datetime('2020-12-31',
                                                                                                            utc = True),
                                                                             initial_threshold = 0.9)

    final_quantile_results_dict = {model:{initial_threshold: {VaR_level:0 for VaR_level in empirical_application_VaR_levels} for initial_threshold in empirical_application_initial_threshold_choices} for model in models}
    final_results_0_homogenous = {model:{initial_threshold: {VaR_level:0 for VaR_level in empirical_application_VaR_levels} for initial_threshold in empirical_application_initial_threshold_choices} for model in models}
    final_CTE_0_homogenous = {model:{initial_threshold: {VaR_level:0 for VaR_level in empirical_application_VaR_levels} for initial_threshold in empirical_application_initial_threshold_choices} for model in models + ['convex_comb', 'min_patton_loss']}

    for initial_threshold in empirical_application_initial_threshold_choices: 
        for VaR_level in empirical_application_VaR_levels:
            for model in models:
                final_quantile_results_dict[model][initial_threshold][VaR_level] = np.mean(quantile_results_dict[model][initial_threshold][VaR_level])
                final_results_0_homogenous[model][initial_threshold][VaR_level] = np.mean(results_0_homogenous[model][initial_threshold][VaR_level])
            for model in models + ['convex_comb', 'min_patton_loss']:
                #CTE Loss                
                final_CTE_0_homogenous[model][initial_threshold][VaR_level] = np.mean(CTE_0_homogenous[model][initial_threshold][VaR_level])

    VaR_empirical_application_results_financial_risk_test = pd.DataFrame(columns = ['Threshold', 'VaR Level'] + models)
    VaR_empirical_application_results_0_homogenous_financial_risk_test = pd.DataFrame(columns = ['Threshold', 'VaR Level'] + models)
    CTE_empirical_application_results_0_homogenous_financial_risk_test = pd.DataFrame(columns = ['Threshold', 'VaR Level'] + models + ['convex_comb', 'min_patton_loss'])
    index_counter = 0
    for initial_threshold in empirical_application_initial_threshold_choices: 
        for VaR_level in empirical_application_VaR_levels:
            VaR_empirical_application_results_financial_risk_test.loc[index_counter, 'Threshold'] = initial_threshold
            VaR_empirical_application_results_financial_risk_test.loc[index_counter, 'VaR Level'] = VaR_level
            VaR_empirical_application_results_0_homogenous_financial_risk_test.loc[index_counter, 'Threshold'] = initial_threshold
            VaR_empirical_application_results_0_homogenous_financial_risk_test.loc[index_counter, 'VaR Level'] = VaR_level
            CTE_empirical_application_results_0_homogenous_financial_risk_test.loc[index_counter, 'Threshold'] = initial_threshold
            CTE_empirical_application_results_0_homogenous_financial_risk_test.loc[index_counter, 'VaR Level'] = VaR_level
            for model in models:
                VaR_empirical_application_results_financial_risk_test.loc[index_counter, model] = np.mean(final_quantile_results_dict[model][initial_threshold][VaR_level]) #/1e6
                VaR_empirical_application_results_0_homogenous_financial_risk_test.loc[index_counter, model] = np.mean(final_results_0_homogenous[model][initial_threshold][VaR_level])
            for model in models + ['convex_comb', 'min_patton_loss']:
                CTE_empirical_application_results_0_homogenous_financial_risk_test.loc[index_counter, model] = np.mean(final_CTE_0_homogenous[model][initial_threshold][VaR_level])
            index_counter += 1

    financial_risk_mean_shape_parameter_estimates = pd.DataFrame.from_dict(mean_shape_parameter_estimates, orient = 'index', columns = ['shape'])
    financial_risk_mean_scale_parameter_estimates = pd.DataFrame.from_dict(mean_scale_parameter_estimates, orient = 'index', columns = ['scale'])

    financial_risk_mean_shape_parameter_estimates.to_csv(r'{}/Results/Application/financial_risk_shape_parameter_.csv'.format(file_path), index = False)
    financial_risk_mean_scale_parameter_estimates.to_csv(r'{}/Results/Application/financial_risk_scale_parameter.csv'.format(file_path), index = False)
    VaR_empirical_application_results_financial_risk_test.to_csv(r'{}/Results/Application/VaR_empirical_application_results_financial_risk_test.csv'.format(file_path), index = False)
    VaR_empirical_application_results_0_homogenous_financial_risk_test.to_csv(r'{}/Results/Application/VaR_empirical_application_results_0_homogenous_financial_risk_test.csv'.format(file_path), index = False)
    CTE_empirical_application_results_0_homogenous_financial_risk_test.to_csv(r'{}/Results/Application/CTE_empirical_application_results_0_homogenous_financial_risk_test.csv'.format(file_path), index = False)
