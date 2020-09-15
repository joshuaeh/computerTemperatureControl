import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

from Air.firstPrinciplesAir import sim_air

# initialize arrays
## Simulation arrays
n = 3600
t = np.linspace(0, n, n+1)
dt = t[1] - t[0]

sp = np.ones(n+1) * (273.15 + 90)

q_cpu = np.ones(n+1) * 75
q_cpu[600:] = 90
q_cpu[1800:] = 50
q_cpu[3000:] = 100

T_ambient = np.ones(n+1) * (273.15 + 25)
T_ambient[1000:] = (273.15 + 35)
T_ambient[2000:] = (273.15 + 20)
## Storage Arrays
P = np.zeros(n+1)
I = np.zeros(n+1)
D = np.zeros(n+1)
FF = np.zeros(n+1)  # feed forward contribution
E = np.zeros(n+1)
SSE = np.zeros(n+1)
T_cpu = np.ones(n+1) * 300
u = np.ones(n+1) * 3600

fan_max = .01878351
op0 = u[0]  # ubias to prevent kick
# define controller saturation points to prevent anti-reset windup
op_hi = 3600  # RPM
op_lo = 0     # RPM
# initialize PID Parameters
KP = -801 * 2 * fan_max / 3600  # (deg K)/(m^3/sec)  # TODO change this so that we are in RPM instead
tauP = .24  # sec
thetaP = .92  # sec

## IMC Tuning Parameters
tuning_style = 'Moderate'  # TODO Relocate User Input?
if tuning_style.lower() == 'aggressive':
    tauC = max(.1 * tauP, .8 * thetaP)
elif tuning_style.lower() == 'moderate':
    tauC = max(tauP, 8 * thetaP)
elif tuning_style.lower() == 'conservative':
    tauC = max(10 * tauP, 80 * thetaP)
else:
    print(f'Tuning Style Parameter must be either "aggressive" "moderate" or "conservative".\n'
          f'Your value is: {tuning_style}'
          f'Defaulting to Moderate Tuning')
    tauC = max(tauP, 8 * thetaP)
Kc = (tauP + .5 * thetaP) / (KP * (tauC + .5 * thetaP))
tauI = tauP + .5 * thetaP
tauD = 0  # Assume 0 unless oscillation is a problem
# tauD = (tauP * thetaP) / (2 * tauP + thetaP)

# Create Kff Relations
#   d * Kd(deg change/ unit disturbance) / kP (deg K / m^3/sec) = m^3/sec
## Heater
Kd_q = 52/105  # degrees change per Watt

#  T Ambient
def Kd_Ta(T_ambient):
    return (T_ambient - 293) / 3  # degrees change per degree change outside

# loop through PID
for i in range(1,n-1):
    E[i] = sp[i] - T_cpu[i]
    SSE[i] = SSE[i-1] * E[i]**2
    P[i] = Kc * E[i]
    I[i] = Kc / tauI * (E[i] * dt) + E[i-1]
    D[i] = -Kc * tauD * ((T_cpu[i] - T_cpu[i-1])/dt)
    FF[i] = Kd_q * KP * q_cpu[i] + (T_ambient[i] - 293)/3 * KP
    u[i] = P[i] + I[i] + D[i] + FF[i]

    # anti reset windup prevention
    u[i] = min(op_hi, u[i])
    u[i] = max(op_lo, u[i])

    # simulate
    vol_air = u[i] / 3600 * 2 * .01878351
    y = odeint(sim_air, T_cpu[i], [0, dt], args=(q_cpu[i], vol_air, T_ambient[i]))
    T_cpu[i+1] = y[-1]




# Fit PID Parameters to Simulated Data

# Show Results

