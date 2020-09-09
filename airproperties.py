# Copyright (C) 2018 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, airproperties.py, is a python library of the                  #
# thermophysical properties of air.  The properties that are pressure-     #
# independent, both constant and temperature-dependent, are taken from the #
# DIPPR(R) Public database.  A such, it can only be used while a student   #
# at BYU.                                                                  #
#                                                                          #
# airproperties.py is distributed in the hope that it will be useful,      #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     #
#                                                                          #
# All published work which utilizes this library, or other property data   #
# from the DIPPR(R) database should have a Public or Sponsor license for   #
# the DIPPR(R) database and include the citation below.                    #
# R. L. Rowley, W. V. Wilding, J. L. Oscarson, T. A. Knotts, N. F. Giles,  #
# DIPPR® Data Compilation of Pure Chemical Properties, Design Institute    #
# for Physical Properties, AIChE, New York, NY (2017).                     #
#                                                                          #
# All published work which utilzes the data obtained from the vdnsat       #
# function should also have a copy of and cite the following.              #
# T. L. Bergman, A. S. Lavine, F. P. Incropera, D. P. Dewitt, Fundamentals #
# of Heat and Mass Transfer 7th ed., John Wiley & Sons, Hoboken, NJ,       #
# (2011).                                                                  #
#                                                                          #
# ======================================================================== #
# airproperties.py                                                         #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - February 2018                                              #
# ======================================================================== #

# ======================================================================== #
# airproperties.py                                                         #
#                                                                          #
# This library contains functions for the properties of air.               #
# Most values for the pressure-independent properties come from the        #
# DIPPR(R) Public database, and the DIPPR(R) abbreviations are used.       #
# The density at 1 atm comes from a spline of the data found in the        #
# 7th ed. of "The Fundamentals of Heat and Mass Transfer" by Bergman et.   #
# al. Density-related properties for the vapor phase use the DIPPR(R)      #
# and this spline function of the density data from Bergman et. al.        #              #                                                                          #
# The library can be loaded into python via the following command:         #
# import airproperties as air                                              #
#                                                                          #
# When imported in this way, the properties can be accessed as:            #
# air.tc for the critical temperature and air.vtc(t) for the vapor         #
# thermal conductivity at temperature t where t is in units of K.          #
# A complete list of properties, and the associated units, are found       #
# below.                                                                   #
#                                                                          #
# Function    Return Value                             Input Value         #
# ---------   --------------------------------------   -----------------   #
# tc          critical temperature in K                none                #
# pc          critical pressure in Pa                  none                #
# vc          critical volume in m**3/mol              none                #
# zc          critical compressibility factor          none                #
# mw          molecular weight in kg/mol               none                #
# acen        acentric factor                          none                #
# icp(t)      ideal gass heat capacity in J/mol/K      temperature in K    #
# vtc(t)      vapor thermal conductivity in W/m/K      temperature in K    #
# vvs(t)      vapor viscosity in Pa*s                  temperature in K    #
# hvp(t)      heat of vaporization in J/mol            temperature in K    # 
# rho1atm(t)  density at 1 atm in kg/m**3             temperature in K    #
# nu1atm(t)   kinematic viscosity at 1 atm in m**2/s   temperature in K    #
# alpha1atm(t)thermal diffusivity at 1atm in m**2/s    temperature in K    #
# pr1atm(t)   Prandtl number at 1 atm                  temperature in K    #
# ======================================================================== #

import numpy as np
from scipy   import interpolate

# critical temperature
tc = 132.45 # units of K
  
# critical pressure
pc = 3.774E6 # units of Pa

# critical volume
vc = 0.09147 / 1000.0 # convert from m**3/kmol to m**3/mol

# critical compressibility factor
zc = 0.313 # unitless

# acentric factor
acen = 0 # unitless

# molecular weight
mw = 0.02896 # units of kg/mol
  
def icp(t): # ideal gas heat capacity
    A = 2.8958E+04
    B = 9.3900E+03
    C = 3.0120E+03
    D = 7.5800E+03
    E = 1.4840E+03
    y = A+B*((C/t)/np.sinh(C/t))**2+D*((E/t)/np.cosh(E/t))**2
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return y # units of J/mol/K

def vtc(t): # thermal conductivity
    A = 3.1417E-04
    B = 7.7860E-01
    C = -7.1160E-01
    D = 2.1217E+03
    y = A*t**B/(1+C/t+D/t**2)
    return y # units of W/m/K

def vvs(t): # viscosity
    A = 1.4250E-06
    B = 5.0390E-01
    C = 1.0830E+02
    D = 0
    y = A*t**B/(1+C/t+D/t**2)
    return y # Pa*s

def rho1atm(t): # density at 1 atm
    DT=np.array([100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,3000])
    DR=np.array([3.5562,2.3364,1.7458,1.3947,1.1614,0.9950,0.8711,0.7740,0.6964,0.6329,0.5804,0.5356,0.4975,0.4643,0.4354,0.4097,0.3868,0.3666,0.3482,0.3166,0.2902,0.2679,0.2488,0.2322,0.2177,0.2049,0.1935,0.1833,0.1741,0.1658,0.1582,0.1513,0.1488,0.1389,0.1135])
    tck = interpolate.splrep(DT,DR)
    y=interpolate.splev(t,tck)
    return y # kg/m^3

def nu1atm(t): # kinetmatic viscosity at 1 atm
    return vvs(t)/rho1atm(t) # m^2/s

def alpha1atm(t): # thermal diffusivity
    return vtc(t)/rho1atm(t)*mw/icp(t) # m^2/s

def pr1atm(t): # Prandtl Number
    return nu1atm(t)/alpha1atm(t) 
    

  

    

  
