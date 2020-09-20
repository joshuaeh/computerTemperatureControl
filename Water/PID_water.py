import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint

from Air.firstPrinciplesAir import sim_air

Kp = -.2
taup = 6
Tss = 360
uss = 100


# FOPDT Model
def Air_FOPDT(T_cpu, t, u, q, Ta):
    # Disturbance effects
    dTdt_q = (q - 105) / 20
    dTdt_Ta = (Ta - 298)
    D = dTdt_q + dTdt_Ta
    # FOPDT Model
    dTdt = (-(T_cpu - Tss) + Kp * (u-uss))/taup + D
    return dTdt


# initialize arrays
## Simulation arrays
n = 3600
t = np.linspace(0, n, n+1)
dt = t[1] - t[0]

sp = np.ones(n+1) * (273.15 + 60)

# q_cpu: random each minute between 10-105 W
q_cpu = np.ones(n+1)
for i in range(1, n+1):
    q_cpu[i] = q_cpu[i-1] + np.random.uniform(-.1, .1)
    q_cpu[i] = min(q_cpu[i], 105)
    q_cpu[i] = max(q_cpu[i], 10)
# ambient temperature bounded between 15 and 32 C
T_ambient = np.ones(n+1) * (273.15 + 25)
for i in range(1, n+1):
    T_ambient[i] = T_ambient[i-1] + np.random.uniform(-.1, .1)
    T_ambient[i] = min(T_ambient[i], 273.15 + 32)
    T_ambient[i] = max(T_ambient[i], 273.15 + 15)

## Storage Arrays
P = np.zeros(n+1)
I = np.zeros(n+1)
D = np.zeros(n+1)
FF = np.zeros(n+1)  # feed forward contribution
E = np.zeros(n+1)
SSE = np.zeros(n+1)
T_cpu = np.ones(n+1) * 300
u = np.ones(n+1) * 100

fan_max = .01878351
op0 = u[0]  # ubias to prevent kick
# define controller saturation points to prevent anti-reset windup
op_hi = 100  # % fan
op_lo = 0     # % fan

# initialize PID Parameters
KP = Kp  # (deg K)/(m^3/sec)  # TODO change this so that we are in RPM instead
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

# loop through PID
for i in range(1,n-1):
    E[i] = sp[i] - T_cpu[i]
    P[i] = Kc * E[i]
    I[i] = Kc / tauI * (E[i] * dt) + I[i-1]
    D[i] = -Kc * tauD * (T_cpu[i] - T_cpu[i-1])/dt
    u[i] = P[i] + I[i] # + D[i] + FF[i]

    # anti reset windup prevention
    if u[i] > op_hi:
        u[i] = min(op_hi, u[i])
        I[i] = I[i-1]
    elif u[i] < op_lo:
        u[i] = max(op_lo, u[i])
        I[i] = I[i - 1]
    # Prevent more that a 10% change per second
    u[i] = max(u[i], u[i - 1] * .7)
    u[i] = min(u[i], u[i - 1] * 1.3)
    # simulate
    y = odeint(Air_FOPDT, T_cpu[i], [0, dt], args=(u[i], q_cpu[i], T_ambient[i]))

    # simulate measurement noise:
    y + np.random.uniform(-2, 2)
    T_cpu[i+1] = y[-1]


# Show Results
def mse(A, B):
    return (np.square(A - B)).mean(axis=0)


j = 40  # graph lower bound (index)
k = -2  # graph upper bound

plt.figure(figsize=(12, 20))
plt.subplot(4, 1, 1)
plt.plot(t[j:k], T_cpu[j:k]-273.15, label='Computer Temperature')
plt.plot(t[j:k], sp[j:k]-273.15, label='Set Point')
plt.ylabel(r'Temperature ($^\circ$C)')
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(t[j:k], u[j:k], 'k-')
plt.ylabel('Fan')
plt.text(j, u[j:k].min(), fr'MSE: {round(mse(T_cpu[j:k], sp[j:k]), 4)} deg$^2$/sec')
plt.subplot(4, 1, 3)
plt.plot(t[j:k], q_cpu[j:k], 'r-')
plt.ylabel('CPU Heat (W)')
plt.subplot(4, 1, 4)
plt.plot(t[j:k], T_ambient[j:k]-273.15, 'r-')
plt.ylabel('Ambient Temperature')
plt.savefig('FinalPIDControl_Water_lessDynamic.png')
plt.show()