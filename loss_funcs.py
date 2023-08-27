# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 07:56:24 2021

@author: Robert James
"""
import numpy as np

# =============================================================================
    #Standard Quantile Loss, Making and Evaluating Point Forecasts Gneiting 2011
def quantile_loss(r, x, alpha):
    return (1/(1-alpha))*np.nanmean(((x <= r) - alpha)*(r-x))   
def point_quantile_loss(r, x, alpha):
    return ((x <= r) - alpha)*(r-x)

# =============================================================================
    #Loss functions of ELICITABILITY AND BACKTESTING: PERSPECTIVES FOR BANKING REGULATION1 by Nolde and Zeiglel 2017
def tick_loss_1_homogenous(r, x, alpha):
    return (1/(1-alpha))*(np.nanmean((1 - alpha - (x > r))*r + (x > r)*x))

def tick_loss_0_homogenous(r, x, alpha):
    to_log = np.where(x <=0, 1e-6, x)
    return (1/(1-alpha))*(np.nanmean((1 - alpha - (x > r))*np.log(r) + (x > r)*np.log(to_log)))

def point_tick_loss_0_homogenous(r, x, alpha):
    to_log = np.where(x <=0, 1e-6, x)
    return (1 - alpha - (x > r))*np.log(r) + (x > r)*np.log(to_log)

def joint_VaR_CTE_loss(r1,r2,x,alpha):
    return (1/(1-alpha))*(np.nanmean((x > r1)*((x-r1)/(2*np.sqrt(r2))) + (1 - alpha)*((r1 + r2)/(2*np.sqrt(r2)))))

def joint_VaR_CTE_loss_0_homogenous(r1,r2,x,alpha):
    return (1/(1-alpha))*(np.nanmean((x > r1)*((x-r1)/(r2)) + (1 - alpha)*((r1/r2) - 1 + np.log(r2))))

def joint_VaR_CTE_patton_0_homogenous(r1,r2,x,alpha):

    # Invert all components of the loss function
    r1 = r1*-1
    r2 = r2*-1
    x = x*-1
    alpha = 1-alpha

    return (1/(1-alpha))*np.nanmean(-(1/(alpha*r2))*(x <= r1)*(r1-x) + (r1/r2) + np.log(-r2) - 1)

def point_joint_VaR_CTE_patton_0_homogenous(r1,r2,x,alpha):

    # Invert all components of the loss function
    r1 = r1*-1
    r2 = r2*-1
    x = x*-1
    alpha = 1-alpha

    return -(1/(alpha*r2))*(x <= r1)*(r1-x) + (r1/r2) + np.log(-r2) - 1