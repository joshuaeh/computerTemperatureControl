import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from firstOrder import tempSim

# change cwd to be hwere file is running
os.chdir(os.path.dirname(sys.argv[0]))

Ta = 25 + 273.15  # ambient temperature (deg K)
fan_max = .01878351  # max volume flow rate of fans (m^3)

# step test 1: fan
# Initialize arrays
n = 3600  # sim 1 hour
time = np.linspace(0, n, n+1)
T_air = np.ones(n+1) * Ta
q = np.ones(n+1) * 100  # 100 W power usage
# fan
fan = np.ones(n+1) * .8
fan[1000:] = .3
fan[2000:] = 1
fan[3000:] = .5
fan = fan * fan_max * 2

initial_values = [Ta, Ta]
temps = np.zeros((n+1, 2))
# simulate
for i in range(n):
    y = odeint(tempSim, initial_values, [0,1], args=(q[i], fan[i], T_air[i]))
    initial_values = y[-1]
    temps[i+1] = initial_values

# save data
d = {'Time': time, 'Temp':temps, 'Fan':fan, 'Q':q, 'T_air':T_air}
df = pd.DataFrame(data=d)
df.to_csv('Fan_step_test.csv')

# plot results
plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(time/60, temps[:,0], label='CPU Temp')
plt.plot(time/60, temps[:,1], label='Liquid Temp')
plt.legend()
plt.ylabel('deg K')
plt.subplot(2, 1, 2)
plt.plot(time/60, q/115, label='heater(%)')
plt.ylabel('Heater %')
plt.xlabel('Time (min)')
plt.show()
plt.savefig('q_step.png')


# step test 2: q
# Initialize arrays
n = 3600  # sim 1 hour
time = np.linspace(0, n, n+1)
fan = np.ones(n+1) * fan_max * .8 * 2
T_air = np.ones(n+1) * Ta

q = np.ones(n+1) * 200 # 75 W power usage
q[1000:] = 100
q[2000:] = 300
q[3000:] = 400


initial_values = [Ta, Ta]
temps = np.zeros((n+1, 2))
# simulate
for i in range(n):
    y = odeint(tempSim, initial_values, [0,1], args=(q[i], fan[i], T_air[i]))
    initial_values = y[-1]
    temps[i+1] = initial_values

# save data
d = {'Time': time, 'Temp':temps, 'Fan':fan, 'Q':q, 'T_air':T_air}
df = pd.DataFrame(data=d)
df.to_csv('q_step_test.csv')

# plot results
plt.figure(figsize=(10, 7))
plt.subplot(2, 1, 1)
plt.plot(time/60, temps[:,0], label='CPU Temp')
plt.plot(time/60, temps[:,1], label='Liquid Temp')
plt.legend()
plt.ylabel('deg K')
plt.subplot(2, 1, 2)
plt.plot(time/60, q/115, label='heater(%)')
plt.ylabel('Heater %')
plt.xlabel('Time (min)')
plt.show()
plt.savefig('q_step.png')





