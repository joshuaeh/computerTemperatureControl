import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from Air.firstPrinciplesAir import dTCPUdt

Ta = 25 + 273.15  # ambient temperature (deg K)
fan_max = .01878351  # max volume flow rate of fans (m^3)
q_max = 105  # maximum CPU Wattage (W)

# step test 1: fan
# Initialize arrays
n = 3600  # sim 1 hour
time = np.linspace(0, n, n+1)
T_air = np.ones(n+1) * Ta
q = np.ones(n+1) * 100  # 100 W power usage
# fan steps
fan = np.ones(n+1) * .3
fan[300:] = .01
fan[600:] = .1
fan[900:] = .5
fan[1200:] = .03
fan[1500:] = .7
fan[1800:] = .15
fan[2100:] = .6
fan[2400:] = .05
fan[2700:] = .3
fan[3000:] = .2
fan = fan * fan_max * 2

initial_values = [Ta]
temps = np.ones(n+1) * Ta
# simulate
for i in range(n):
    y = odeint(dTCPUdt, initial_values, [0,1], args=(q[i], fan[i], T_air[i]))
    initial_values = y[-1]
    temps[i+1] = initial_values

# save data
Tcpu = temps
d = {'Time': time, 'Tcpu': Tcpu, 'Fan': fan, 'Q': q, 'T_air': T_air}
df = pd.DataFrame.from_dict(data=d)
df.to_csv('w_Fan_step.csv')

# plot results
plt.figure(figsize=(5, 4))
plt.subplot(2, 1, 1)
plt.plot(time/60, temps-273.15, label='CPU Temp')
plt.legend()
plt.ylabel('Temp (deg C)')
plt.subplot(2, 1, 2)
plt.plot(time/60, fan/(fan_max * 2), label='fan(%)')
plt.ylabel('fan %')
plt.xlabel('Time (min)')
plt.savefig('w_fan_step.png')

# step test 2: q
# Initialize arrays
n = 3600  # sim 1 hour
time = np.linspace(0, n, n+1)
fan = np.ones(n+1) * fan_max * .8 * 2
T_air = np.ones(n+1) * Ta

q = np.ones(n+1)
q[0:] = 0
q[1800:] = q_max

initial_values = [Ta]
temps = np.ones(n+1)*Ta
# simulate
for i in range(n):
    y = odeint(dTCPUdt, initial_values, [0,1], args=(q[i], fan[i], T_air[i]))
    initial_values = y[-1]
    temps[i+1] = initial_values

# save data
Tcpu = temps
d = {'Time': time, 'Tcpu': Tcpu, 'Fan': fan, 'Q': q, 'T_air': T_air}
df = pd.DataFrame(data=d)
df.to_csv('w_q_step.csv')

# plot results
plt.figure(figsize=(5, 4))
plt.subplot(2, 1, 1)
plt.plot(time/60, temps-273.15, label='CPU Temp')
plt.legend()
plt.ylabel('Temp (deg C)')
plt.subplot(2, 1, 2)
plt.plot(time/60, q/115, label='heater(%)')
plt.ylabel('Heater %')
plt.xlabel('Time (min)')
plt.savefig('w_q_step.png')

# step test 3: T_outside
# Initialize arrays
n = 3600  # sim 1 hour
time = np.linspace(0, n, n+1)
fan = np.ones(n+1) * fan_max * .8 * 2
q = np.ones(n+1) * .8 * q_max
T_air = np.ones(n+1) * (Ta-5)
T_air[1800:] = (Ta+10)

initial_values = [Ta]
temps = np.ones(n+1) * Ta
# simulate
for i in range(n):
    y = odeint(dTCPUdt, initial_values, [0,1], args=(q[i], fan[i], T_air[i]))
    initial_values = y[-1]
    temps[i+1] = initial_values

# save data
Tcpu = temps
d = {'Time': time, 'Tcpu': Tcpu, 'Fan': fan, 'Q': q, 'T_air': T_air}
df = pd.DataFrame(data=d)
df.to_csv('w_Ta_step.csv')

# plot results
plt.figure(figsize=(5, 4))
plt.subplot(2, 1, 1)
plt.plot(time/60, temps, label='CPU Temp')
plt.legend()
plt.ylabel('Temp (deg C)')
plt.subplot(2, 1, 2)
plt.plot(time/60, T_air-273.15)
plt.ylabel('T ambient (deg C)')
plt.xlabel('Time (min)')
plt.savefig('w_Ta_step.png')


