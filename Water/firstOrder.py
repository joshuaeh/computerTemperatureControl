# Imports
import numpy as np
import waterproperties as wp
import airproperties as ap
from scipy.integrate import odeint

#assuming lumped capacitance on CPU Chip

# Constants
# CPU
cpu_m = .0455  # mass of cpu (Kg)
cpu_cp = 700  # Heat Capacity CPU (J/ Kg/ K)
cpu_q_max = 105  # maximum heat output (W)
cpu_A = .039878**2  # CPU Surface Area (m^2)

# cold Plate
coldPlate_l = .003  # Cold plate thickness (m)
coldPlate_k = 401  # Cold Plate Conductivity (W/ m K)

# liquid
liquid_m = 1  # (Kg) mass/ mass flow car 
# Assume Laminar, fully developped flow:
# Nu = 3.66

# radiator
# hx_U =   # Universal Heat Transfer Constant of HX ()
hx_height = .0296  # fin height (m)
hx_length = .240  # length of HX in (m)
hx_width = .105  # width of a fin (m)
hx_num_fins = 30/.0224  # num fins per length (m)
hx_A = 2 * hx_height * hx_length * hx_width * hx_num_fins

# fans
fan_max = .01878351  # max volume flow rate of fans (m^3)
fan_ca = .120**2  # fan cross sectional area (m^2)

# Functions
def tempSim(T, t, q, vol_air, T_air):
    '''

    '''
    T_cpu, T_liquid = T

    # calculate convection constant for air and liquid
    # air
    vel_air = vol_air / (2 * fan_ca)
    Re_air = vel_air * hx_width / ap.nu1atm(T_air)
    Nu_air = .680 * Re_air**(.5) * ap.pr1atm(T_air)
    h_air_hx = Nu_air * ap.vtc(T_air) / hx_width
    # liquid
    h_liquid = 3.66 * wp.ltc(T_liquid)/ .005

    # calculate Resistances
    R_air_to_hx = 1/(h_air_hx * hx_A)
    R_hx = .0005/(coldPlate_k * hx_A)
    R_hx_to_liquid = 1/(h_liquid * hx_A)
    R_liquid_to_coldPlate = 1/(h_liquid * cpu_A*2.5)
    R_coldPlate = coldPlate_l/(coldPlate_k * cpu_A)

    # calculate Q's
    q_cpu_to_liquid = (T_liquid - T_cpu) / (R_liquid_to_coldPlate + R_coldPlate)  # problem inhigh CPU temps is in R_liquid to cold plate

    q_liquid_to_air = (T_air - T_liquid) / (R_air_to_hx + R_hx +  R_hx_to_liquid)

    # Calculate change in temperature
    dT_cpu = (q + q_cpu_to_liquid) / (cpu_m * cpu_cp)
    dT_liquid = (-q_cpu_to_liquid + q_liquid_to_air) / (liquid_m * wp.lcp(T_liquid))

    return np.array([dT_cpu, dT_liquid])

