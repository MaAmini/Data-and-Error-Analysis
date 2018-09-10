# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:50:06 2018

@author: Marzieh
"""
# ipython magic
#%matplotlib notebook
#%load_ext autoreload
#%autoreload 2
#
#%matplotlib inline

# plot configuration
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("ggplot")
# import seaborn as sns # sets another style
matplotlib.rcParams['lines.linewidth'] = 3
fig_width, fig_height = (7.0,5.0)

matplotlib.rcParams['figure.figsize'] = (fig_width, fig_height)

# font = {'family' : 'sans-serif',
#         'weight' : 'normal',
#         'size'   : 18.0}
# matplotlib.rc('font', **font)  # pass in the font dict as kwar

import chaospy as cp
import numpy as np
import openturns as ot
#from openturns.viewer import View
#from monte_carlo import generate_sample_matrices_mc
#from monte_carlo import calculate_sensitivity_indices_mc
unit_mmhg_pa = 133.3
unit_pa_mmhg = 1./unit_mmhg_pa
unit_cm2_m2 = 1. / 100. / 100.
unit_m2_cm2 = 1. / unit_cm2_m2

# begin quadratic area model
def quadratic_area_model(pressure_range, samples):
    """
    calculate the arterial vessel wall for a set pressure range
    from 75 to 140 mmHg for a given set of reference wave speeds
    area and pressure.

    :param
        pressure_range: np.array
            pressure range for which to calculate the arterial area

        samples: np.array with shape (4:n_samples)
            sample matrix where
            first indices correspond to
                a_s: reference area
                c_s: reference waves seed
                p_s: reference pressure
                rho: blood density
            and n_samples to the number of samples

    :return:
        arterial pressure : np.array
            of size n_samples
    """
    pressure_range = pressure_range.reshape(pressure_range.shape[0], 1)
    a_s, c_s, p_s, rho = samples
    beta = 2*rho*c_s**2/np.sqrt(a_s)*a_s
    #C_Laplace = (2. * ((P - Ps) * As / betaLaplace + np.sqrt(As))) * As / betaLaplace
    return ((pressure_range - p_s)*a_s/beta + np.sqrt(a_s))**2.

# begin exponential area model
def logarithmic_area_model(pressure_range, samples):
    """
    calculate the arterial vessel wall for a set pressure range
    from 75 to 140 mmHg for a given set of reference wave speeds
    area and pressure.

    :param
        pressure_range: np.array
            pressure range for which to calculate the arterial area

        samples: np.array with shape (4:n_samples)
            sample matrix where
            first indices correspond to
                a_s: reference area
                c_s: reference waves seed
                p_s: reference pressure
                rho: blood density
            and n_samples to the number of samples

    :return:
        arterial pressure : np.array
            of size n_samples
    """
    pressure_range = pressure_range.reshape(pressure_range.shape[0], 1)
    a_s, c_s, p_s, rho = samples
    beta = 2.0*rho*c_s**2./p_s
    #C_hayashi = 2.0 * As / betaHayashi * (1.0 + np.log(P / Ps) / betaHayashi) / P
    return a_s*(1.0 + np.log(pressure_range / p_s)/beta) ** 2.0

# start deterministic comparison
pressure_range = np.linspace(45, 180, 100) * unit_mmhg_pa
a_s = 5.12 * unit_cm2_m2
c_s = 6.25609258389
p_s = 100 * unit_mmhg_pa
rho = 1045.

plt.figure()
for model, name in [(quadratic_area_model, 'Quadratic model'), (logarithmic_area_model, 'Logarithmic model')]:
    y_area = model(pressure_range, (a_s, c_s, p_s, rho))
    plt.plot(pressure_range * unit_pa_mmhg, y_area * unit_m2_cm2, label=name)
    plt.xlabel('Pressure [mmHg]')
    plt.ylabel('Area [cm2]')
    plt.legend()
plt.tight_layout()

 #Create marginal and joint distributions
dev = 0.05
a_s = 5.12 * unit_cm2_m2
A_s = ot.Uniform(a_s * (1. - dev), a_s*(1. + dev))
c_s = 6.25609258389
C_s = ot.Uniform(c_s * (1. - dev), c_s*(1. + dev))

p_s = 100 * unit_mmhg_pa
P_s = ot.Uniform(p_s * (1. - dev), p_s*(1. + dev))

rho = 1045.
Rho = ot.Uniform(rho * (1. - dev), rho*(1. + dev))

#jpdf = cp.J(A_s, C_s, P_s, Rho)
marginals = [A_s, C_s, P_s, Rho]
distribution = ot.ComposedDistribution(marginals)

jpdf=distribution
#class=ComposedDistribution name=ComposedDistribution dimension=4 copula=class=IndependentCopula name=IndependentCopula dimension=4 marginal[0]=class=Uniform name=Uniform dimension=1 a=0.0004864 b=0.0005376 marginal[1]=class=Uniform name=Uniform dimension=1 a=5.94329 b=6.5689 marginal[2]=class=Uniform name=Uniform dimension=1 a=12663.5 b=13996.5 marginal[3]=class=Uniform name=Uniform dimension=1 a=992.75 b=1097.25

# scatter plots
pressure_range = np.linspace(45, 180, 100) * unit_mmhg_pa
Ns = 200
sample_scheme = 'R'
samples = jpdf.sample(Ns, sample_scheme)

for model, name, color in [(quadratic_area_model, 'Quadratic model', '#dd3c30'), (logarithmic_area_model, 'Logarithmic model', '#2775b5')]:
    # evaluate the model for all samples
    Y_area = model(pressure_range, samples)

    plt.figure()
    plt.title('{}: evaluations Y_area'.format(name))
    plt.plot(pressure_range * unit_pa_mmhg, Y_area * unit_m2_cm2)
    plt.xlabel('Pressure [mmHg]')
    plt.ylabel('Area [cm2]')
    plt.ylim([3.5,7.])

    plt.figure()
    plt.title('{}: scatter plots'.format(name))
    for k in range(len(jpdf)):
        plt.subplot(len(jpdf)/2, len(jpdf)/2, k+1)
        plt.plot(samples[k], Y_area[0]*unit_m2_cm2, '.', color=color)
        plt.ylabel('Area [cm2]')
        plt.ylim([3.45, 4.58])
        xlbl = 'Z' + str(k)
        plt.xlabel(xlbl)
    plt.tight_layout()