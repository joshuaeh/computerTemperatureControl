import airproperties as ap
import numpy as np

mCPU = .0455  # kg (Mass of CPU)
Cp = 700  # J/(kg*K) (Heat capacity of silicon CPU)
P = 10.5 / 100 * 2 + 2 * .001  # m (Perimeter of fin)
k_copper = 385  # W/(m*K) (Assume constant thermal conductivity of copper fin material)
Ac = .105 * .001  # m (Cross-sectional area of fin)
L = 6 / 100  # m How far fin jets out from CPU
Tinf = 298  # K (Ambient Conditions)
QCPU_max = 105  # W
SA = .039878 ** 2  # m^2 (surface area of CPU)
CPU_A = .039878 ** 2  # CPU Surface Area (m^2)
CPU_L = .039878  # m (Length of CPU)
fan_max = .01878351  # max volume flow rate of fans (m^3)
fan_ca = .120 ** 2  # fan cross sectional area (m^2)

SP = 90 + 273.15  # K (Set Point Temperature)


def dTCPUdt(T, t, Q_CPU, vol_air, T_air):
    θb = T - T_air

    vel_air = vol_air / fan_ca
    μ = ap.vvs(Tinf)  # m^2/s (kinematic viscosity of air at ambient conditions)
    n = 50  # (number of fins)
    fin_gap_width = 10.5 / 100 / n  # m

    # Reynolds number from air properties
    Re_air = n * vel_air * fin_gap_width / ap.nu1atm(T_air)
    Nu_air = .680 * Re_air ** (.5) * ap.pr1atm(T_air)
    h_air_fin = Nu_air * ap.vtc(T_air) / fin_gap_width

    m = np.sqrt(2 * h_air_fin / (k_copper * t + .001))
    M = np.sqrt(h_air_fin * P * k_copper * Ac) * θb

    # Following equation obtained from "Fundamentals of Heat and Mass Transfer" 8E by Bergman and Lavine
    Qfin = M * ((np.sinh(m * L) + h_air_fin / m / k_copper * np.cosh(m * L)) / (
                np.cosh(m * L) + h_air_fin / m / k_copper * np.sinh(m * L)))

    dTCPUdt = Q_CPU - Qfin
    return dTCPUdt
