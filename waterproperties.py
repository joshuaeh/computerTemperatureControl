# Copyright (C) 2018 Thomas Allen Knotts IV - All Rights Reserved          #
# This file, waterproperties.py, is a python library of the                #
# thermophysical properties of liquid water.  The properties, both         #
# constant and temperature-dependent, are taken from the DIPPR(R) Sample   #
# database which can be accessed at <https://dippr.byu.edu>.               #
# The vapor density at 1 atm comes from a spline of the data found in the  #
# 7th ed. of "The Fundamentals of Heat and Mass Transfer" by Bergman et.   #
# al. Density-related properties for the vapor phase use the DIPPR(R)      #
# and this spline function.                                                #
#                                                                          #
# waterproperties.py is distributed in the hope that it will be useful,    #
# but WITHOUT ANY WARRANTY; without even the implied warranty of           #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
# GNU General Public License for more details.                             #
#                                                                          #
# All published work which utilizes this library, or other property data   #
# from the DIPPR(R) database, should include the citation below.           #
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
# waterproperties.py                                                       #
#                                                                          #
# Thomas A. Knotts IV                                                      #
# Brigham Young University                                                 #
# Department of Chemical Engineering                                       #
# Provo, UT  84606                                                         #
# Email: thomas.knotts@byu.edu                                             #
# ======================================================================== #
# Version 1.0 - February 2018                                              #
# Version 1.1 - October 2019 Minor corrections to documentation of tsat.   #
# ======================================================================== #

# ======================================================================== #
# waterproperties.py                                                       #
#                                                                          #
# This library contains functions for the properties of liquid water.      #
# Most values come from the DIPPR(R) Sample database, and the DIPPR(R)     #
# abbreviations are used.                                                  #
#                                                                          #
# The library can be loaded into python via the following command:         #
# import waterproperties as water                                          #
#                                                                          #
# When imported in this way, the properties can be accessed as:            #
# water.tc for the critical temperature and water.vp(t) for the vapor      #
# pressure at temperature t where t is in units of K.                      #
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
# ldn(t)      liquid density in kg/m**3                temperature in K    #
# lcp(t)      liquid heat capacity in J/mol/K          temperature in K    #
# ltc(t)      liquid thermal conductivity in W/m/K     temperature in K    #
# vp(t)       liquid vapor pressure in Pa              temperature in K    #
# hvp(t)      heat of vaporization in J/mol            temperature in K    # 
# pr(t)       Prandtl number                           temperature in K    #
# lvs(t)      liquid viscosity in Pa*s                 temperature in K    #
# nu(t)       liquid kinematic viscosity in m**2/s     temperature in K    #
# tsat(p)     temperature at saturation in K           pressure in Pa      #
# vvs(t)      vapor (steam) viscosity in Pa*s          temperature in K    #
# vtc(t)      vapor (steam) therm. conductiv. in W/m/K temperature in K    #
# vdnsat(t)   vapor (steam) density at saturation      temperature in K    #
# ======================================================================== #

import numpy as np
from scipy.optimize import fsolve
from scipy   import interpolate

# critical temperature
tc = 647.096 # units of K

# critical pressure
pc = 2.20640E7 # units of Pa

# critical volume
vc = 0.0559472/1000.0 # unit of m**3/mol; convert from m**3/kmol to m**3/mol

# critical compressibility factor
zc = 0.229 # unitless

# acentric factor
acen = 0.344861 # unitless

# molecular weight
mw = 0.01801528 # units of kg/mol
  
def ldn(t): # liquid density
    A = 1.7874E+01
    B = 3.5618E+01
    C = 1.9655E+01
    D = -9.1306E+00
    E = -3.1367E+01
    F = -8.1356E+02
    G = -1.7421E+07
    tr = t/tc
    x = 1.0-tr
    y = A + B * x**(1.0/3.0) + C * x**(2.0/3.0) + D * x**(5.0/3.0) + E * x**(16.0/3.0) + F * x**(43.0/3.0) + G * x**(110.0/3.0)
    y = y * 1000 # convert from kmol/m^3 to mol/m^3
    y = y * mw # convert from mol/m^3 to kg/m^3
    return y # units of kg/m^3
  
def lcp(t): # liquid heat capacity
    A = 2.7637E+05
    B = -2.0901E+03
    C = 8.1250E+00   
    D = -1.4116E-02
    E = 9.3701E-06
    y = A + B * t + C * t**2 + D * t**3 + E * t**4
    y = y / 1000 # convert from J/kmol/K to J/mol/K
    return y # units of J/mol/K

def ltc(t): # liquid thermal conductivity
    A = -4.3200E-01
    B = 5.7255E-03
    C = -8.0780E-06
    D = 1.8610E-09
    E = 0
    y = A + B * t + C * t**2 + D * t**3 + E * t**4
    return y # units of W/m/K

def vp(t): # liquid vapor pressure
    A = 7.3649E+01
    B = -7.2582E+03
    C = -7.3037E+00
    D = 4.1653E-06
    E = 2.0000E+00
    y = np.exp(A + B / t + C * np.log(t) + D * t**E)
    return y # units of Pa
    
def hvp(t): # heat of vaporization
    A = 5.6600E+07
    B = 6.12041E-01
    C = -6.25697E-01
    D = 3.98804E-01
    tr = t/tc
    y = A * (1.0-tr)**(B + C * tr + D * tr**2)
    y = y / 1000 # convert from J/kmol to J/mol
    return y # J/mol
    
def lvs(t): # liquid viscosity
    A = -5.2843E+01
    B = 3.7036E+03
    C = 5.8660E+00
    D = -5.8790E-29
    E = 10.0000E+00
    y = np.exp(A + B / t + C * np.log(t) + D * t**E)
    return y # units of Pa*s

def nu(t): # kinematic liquid viscosity
    return lvs(t)/ldn(t) # m**2/s

def pr(t): # Prandtl number
    return lcp(t)*lvs(t)/ltc(t)/mw # unitless

def ftsat(t,p): # function to calculate tsat with fsolve
    return vp(t) - p

def tsat(p): # saturation temperature (K) at pressure p (Pa)
    x = 700 # guess in K
    y = fsolve(ftsat,x,p)
    return(y[0]) # K
    
def vvs(t): # vapor (steam) viscosity
    A = 1.7096E-08
    B = 1.1146
    return A*t**B # Pa*s

def vtc(t): # vapor (steam) thermal conductivity
    A = 6.2041E-06
    B = 1.1146
    return A*t**B # W/m/K   

def vdnsat(t): # saturated vapor (steam) density   
    tdata = [273.15,275,280,285,290,295,300,305,310,315,320,325,330,335,340,345,350,355,360,365,370,373.15,375,380,385,390,400,410,420,430]
    ddata = [206.3,181.7,130.4,99.4,69.7,51.94,39.13,29.74,22.93,17.82,13.98,11.06,8.82,7.09,5.74,4.683,3.846,3.18,2.645,2.212,1.861,1.679,1.574,1.337,1.142,0.98,0.731,0.553,0.425,0.331]
    tck = interpolate.splrep(tdata,ddata)
    y=interpolate.splev(t,tck)
    return 1.0/y # kg/m**3

  
