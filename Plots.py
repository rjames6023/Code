global file_path
file_path = r'C:\Users\Robert\Dropbox (Sydney Uni)\GPD_Paramaeter_Estimation_Comparison_Project'

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import re


def findMiddle(input_list):
    middle = float(len(input_list)) / 2
    if middle % 2 != 0:
        return input_list[int(middle - .5)]
    else:
        return (input_list[int(middle)], input_list[int(middle - 1)])


def loss_func_comparison_plots(n, distribution, y_parameter_label, x_parameter_label, functional, loss_type,
                               parameter_range_slice, reverse_x_axis, initial_threshold, scale_loss, folder):
    if functional == 'VaR':
        risk_levels = VaR_levels
    else:
        risk_levels = CTE_levels
    if distribution in ['GPD', 'Log-Gamma', 'Pareto', 'Weibull', 'Log-Logistic']:
        parameter_type = 'shape'
    else:
        parameter_type = 'scale'
    parameter_order = []
    results_dictionary = {model: {risk_level: [] for risk_level in risk_levels} for model in models}
    for file in [x for x in file_set if
                 distribution in x and functional in x and loss_type in x and 'n={}'.format(n) in x]:
        parameter_order.append(float(re.search('{}=(\d+.\d+)'.format(parameter_type), file).group(1)))
        import_file = pd.read_csv(r'{}/Results/Simulation/{}'.format(file_path, file))
        import_file.rename(columns={'Unnamed: 0': 'Model'}, inplace=True)
        for model in models:
            for risk_level in risk_levels:
                results_dictionary[model][risk_level].append(
                    import_file[(import_file['Model'] == model) & (import_file['u'] == initial_threshold)][
                        '{} {}'.format(functional, risk_level)].values[0])
    results_frames = {}
    for alpha in risk_levels:
        results1_df = pd.concat([pd.DataFrame(data={'parameter': parameter_order}),
                                 pd.DataFrame({model: results_dictionary[model][alpha] for model in models})], axis=1)
        results1_df.sort_values(by=['parameter'], inplace=True)
        results_frames[alpha] = results1_df

    # if n == 2000 and functional == 'CTE':
    #     subplots_ncols = 2
    #     risk_levels = [0.975, 0.99]
    #     legend_horizontal_adjustment = -0.22
    # else:
    #     subplots_ncols = 3
    #     legend_horizontal_adjustment = -0.95
    subplots_ncols = 3
    legend_horizontal_adjustment = -0.95

    figure, axs = plt.subplots(1, subplots_ncols, figsize=(14, 6))
    x = results1_df['parameter'].values[parameter_range_slice]
    for model in [x for x in models if x != 'Truth']:
        if model == 'min_patton_loss':
            linestyle = 'dashdot'
        else:
            linestyle = 'solid'
        if model == 'min_patton_loss':
            label = 'Min Score'
        elif model == 'convex_combination':
            label = 'Convex Combination'
        else:
            label = model

        for i, alpha in enumerate(risk_levels):
            if scale_loss == True:
                axs[i].plot(x, 1 / (1 - alpha) * (results_frames[alpha][model].values[parameter_range_slice] -
                                                  results_frames[alpha]['Truth'].values[parameter_range_slice]),
                            c=plotting_color_dict[model], label=label, ls=linestyle)
            elif scale_loss == False:
                axs[i].plot(x, results_frames[alpha][model].values[parameter_range_slice] -
                            results_frames[alpha]['Truth'].values[parameter_range_slice], c=plotting_color_dict[model],
                            label=label, ls=linestyle)
            axs[i].plot(x, [0 for x in range(len(x))], c='k', lw=1)
            axs[i].set_ylabel(y_parameter_label, fontsize=18)
            axs[i].set_xlabel(x_parameter_label, fontsize=18)
            axs[i].set_title(r'$\alpha$ = {}'.format(alpha), fontsize=18)

    middle = findMiddle(x)
    if type(middle) == tuple:
        middle = round((middle[0] + middle[1]) / 2, 2)
    for i, alpha in enumerate(risk_levels):
        if reverse_x_axis == True:
            axs[i].invert_xaxis()
            axs[i].set_xlim(x[-1], x[0])
        else:
            axs[i].set_xlim(x[0], x[-1])

        axs[i].set_ylim(0)
        axs[i].tick_params(labelsize=16)
        axs[i].set_xticks([x[0]] + [middle] + [x[-1]])
        axs[i].set_xticklabels(np.round([x[0]] + [middle] + [x[-1]], 2))

    plt.legend(loc='upper center', bbox_to_anchor=(legend_horizontal_adjustment, -0.16), ncol=4, prop={'size': 16})
    plt.subplots_adjust(wspace=0.45)
    plt.savefig(r'{}/Figures/n={}/{}/{} {} Results.png'.format(file_path, n, folder, distribution, functional), dpi=100,
                bbox_inches='tight')


global plotting_color_dict, models
plotting_color_dict = {'MLE': 'b', 'LME': 'lime', 'Revised_NEW': 'orange', 'WNLS': 'red', 'MGF_AD2R': 'cyan',
                       'MGF_AD': 'magenta', 'MDE': 'dimgrey', 'convex_combination':'k', 'min_patton_loss':'k'}
models = ['Truth', 'MLE', 'LME', 'Revised_NEW', 'MGF_AD', 'MGF_AD2R', 'WNLS', 'MDE', 'convex_combination', 'min_patton_loss']
VaR_levels = [0.975, 0.99, 0.999]
CTE_levels = [0.95, 0.975, 0.99]
parameter_config_file = pd.read_excel(r'{}/Code/experiment_config.xlsx'.format(file_path))
file_set = os.listdir(r'{}/Results/Simulation'.format(file_path))

if not os.path.exists(r'{}/Figures'.format(file_path)):
    os.makedirs(r'{}/Figures'.format(file_path))

for n in [1000, 2000]:
    if not os.path.exists(r'{}/Figures/n={}'.format(file_path, n)):
        os.makedirs(r'{}/Figures/n={}'.format(file_path, n))

    folder = 'quantile_tick_loss'
    if not os.path.exists(r'{}/Figures/n={}/{}'.format(file_path, n, folder)):
        os.makedirs(r'{}/Figures/n={}/{}'.format(file_path, n, folder))
    # Students-t results VaR
    loss_func_comparison_plots(n=n,
                               distribution='Students-t',
                               y_parameter_label=r'$S_{1}(x, y)$',
                               x_parameter_label=r'$\nu$',
                               functional='VaR',
                               loss_type='quantile_loss',
                               parameter_range_slice=range(1, 8),
                               reverse_x_axis=True,
                               initial_threshold=100,
                               scale_loss=True,
                               folder=folder)
    # GPD VaR
    loss_func_comparison_plots(n=n, distribution='GPD', y_parameter_label=r'$S_{1}(x, y)$', x_parameter_label=r'$\xi$',
                               functional='VaR', loss_type='quantile_loss', parameter_range_slice=range(3, (18 - 1)),
                               reverse_x_axis=False, initial_threshold=100, scale_loss=True, folder=folder)
    # Pareto VaR
    loss_func_comparison_plots(n=n, distribution='Pareto', y_parameter_label=r'$S_{1}(x, y)$',
                               x_parameter_label=r'$\sigma$', functional='VaR', loss_type='quantile_loss',
                               parameter_range_slice=range(2, 14), reverse_x_axis=True, initial_threshold=100,
                               scale_loss=True, folder=folder)
    # Weibull VaR
    loss_func_comparison_plots(n=n, distribution='Weibull', y_parameter_label=r'$S_{1}(x, y)$',
                               x_parameter_label=r'$k$', functional='VaR', loss_type='quantile_loss',
                               parameter_range_slice=range(0, 25), reverse_x_axis=True, initial_threshold=100,
                               scale_loss=False, folder=folder)
    # Log-Normal VaR
    loss_func_comparison_plots(n=n, distribution='Log-Normal', y_parameter_label=r'$S_{1}(x, y)$',
                               x_parameter_label=r'$s$', functional='VaR', loss_type='quantile_loss',
                               parameter_range_slice=range(12, 25), reverse_x_axis=False, initial_threshold=100,
                               scale_loss=True, folder=folder)
    # Log-Logistic VaR
    loss_func_comparison_plots(n=n, distribution='Log-Logistic', y_parameter_label=r'$S_{1}(x, y)$',
                               x_parameter_label=r'$b$', functional='VaR', loss_type='quantile_loss',
                               parameter_range_slice=range(7, 30), reverse_x_axis=True, initial_threshold=100,
                               scale_loss=True, folder=folder)


    folder = 'CTE_0_homogenous_loss'
    if not os.path.exists(r'{}/Figures/n={}/{}'.format(file_path, n, folder)):
        os.makedirs(r'{}/Figures/n={}/{}'.format(file_path, n, folder))
    # Students-t results CTE
    loss_func_comparison_plots(n=n, distribution='Students-t', y_parameter_label=r'$S_{4}(x_{1}, x_{2}, y)$',
                               x_parameter_label=r'$\nu$', functional='CTE', loss_type='0_homogenous_loss',
                               parameter_range_slice=range(1, 8), reverse_x_axis=True, initial_threshold=100,
                               scale_loss=True, folder=folder)
    # GPD CTE
    loss_func_comparison_plots(n=n, distribution='GPD', y_parameter_label=r'$S_{4}(x_{1}, x_{2}, y)$',
                               x_parameter_label=r'$\xi$', functional='CTE', loss_type='0_homogenous_loss',
                               parameter_range_slice=range(3, (18 - 2)), reverse_x_axis=False, initial_threshold=100,
                               scale_loss=False, folder=folder)
    # Pareto CTE
    loss_func_comparison_plots(n=n, distribution='Pareto', y_parameter_label=r'$S_{4}(x_{1}, x_{2}, y)$',
                               x_parameter_label=r'$\sigma$', functional='CTE', loss_type='0_homogenous_loss',
                               parameter_range_slice=range(2, 14), reverse_x_axis=True, initial_threshold=100,
                               scale_loss=True, folder=folder)
    # Weibull CTE
    loss_func_comparison_plots(n=n, distribution='Weibull', y_parameter_label=r'$S_{4}(x_{1}, x_{2}, y)$',
                               x_parameter_label=r'$k$', functional='CTE', loss_type='0_homogenous_loss',
                               parameter_range_slice=range(0, 25), reverse_x_axis=True, initial_threshold=100,
                               scale_loss=False, folder=folder)
    # Log-Normal CTE
    loss_func_comparison_plots(n=n, distribution='Log-Normal', y_parameter_label=r'$S_{4}(x_{1}, x_{2}, y)$',
                               x_parameter_label=r'$s$', functional='CTE', loss_type='0_homogenous_loss',
                               parameter_range_slice=range(12, 25), reverse_x_axis=False, initial_threshold=100,
                               scale_loss=False, folder=folder)
    # Log-Logistic CTE
    loss_func_comparison_plots(n=n, distribution='Log-Logistic', y_parameter_label=r'$S_{4}(x_{1}, x_{2}, y)$',
                               x_parameter_label=r'$b$', functional='CTE', loss_type='0_homogenous_loss',
                               parameter_range_slice=range(7, 30), reverse_x_axis=True, initial_threshold=100,
                               scale_loss=False, folder=folder)
