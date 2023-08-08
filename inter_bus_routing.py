# Load the needed libraries
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
# Import all classes of PuLP module
from pulp import *
from IPython.display import Image, display

from utils import PassangerDemandScenarios

# The next cell contains constants taken from papers, population density, or distances between
#  cities obtained from google.com.
11033 # Trips per day
(2566, 51) # 'Taipei', 'Taichung' Trips per day.

# Taiwan inter-city bus carrier
cities = ('Taipei', 'Taichung', 'Chiayi', 'Tainan', 'Kaohsiung')
# Population:
population = {
    'Taipei': 2_700_000,
    'Taichung': 2_800_000,
    'Chiayi': 274_000,
    'Tainan': 1_900_000,
    'Kaohsiung': 2_800_000
}

# Travel time between pair of cities, in seconds.
# czyli droga jest przedstawiona w sekundach a nie np w kilometrach
travel_time = {
    ('Taipei', 'Taichung'): (60+49) * 60,
    ('Taipei', 'Tainan'): (60 + 49) * 60,
    ('Taipei', 'Chiayi'): (120 + 39) * 60,
    ('Taipei', 'Taichung'): (3*60 + 4) * 60,
    ('Taipei', 'Kaohsiung'): (3*60 + 30) * 60,

    ('Taichung', 'Tainan'): (60 + 13) * 60,
    ('Taichung', 'Chiayi'): (60 + 36) * 60,
    ('Taichung', 'Kaohsiung'): (2*60 + 6) * 60,

    ('Chiayi', 'Tainan'): (0 + 52) * 60,
    ('Chiayi', 'Kaohsiung'): (60 + 15) * 60,

    ('Tainan', 'Kaohsiung'): (0 + 39) * 60,
}

# Station bus capacity constraint
station_cap = {
    'Taipei': 27,
    'Taichung': 28,
    'Chiayi': 3,
    'Tainan': 19,
    'Kaohsiung': 28
}

M, S = 30, 30
n_iterations=3
pass_scenarios = PassangerDemandScenarios(cities, S=S, M=M, n_iterations=n_iterations)
pass_scenarios.generate()

#To reduce the complexity of the spatiotemporal network, we discretize time into 15-minute bins.
# Thus, the time/cycle of a day is represented as {0:00, 0:15, 0:30, ... 23:30, 23:45}, or using
#  indices: {0, 1, 2, ... 95}.

# Discrete Temporal Network with tao = 15 min.
tao = 15 * 60 # seconds
taos = int(24*60*60/tao) # 96, the number of tao period in the day.

# Convert the travel distance from seconds to number of taos (rounded up).
travel_time_tao = {}
def ceil(val, step):
    cocient = val // step
    mod = val % step
    if mod == 0:
        return cocient
    return cocient + 1
for i in range(len(cities)):
    for j in range(i+1, len(cities)):
        travel_time_tao[(cities[i], cities[j])] = ceil(travel_time[(cities[i], cities[j])], step=tao)
        travel_time_tao[(cities[j], cities[i])] = ceil(travel_time[(cities[i], cities[j])], step=tao)

# Model constant declaration.
m, s = 0, 0
RAR = 0.6
F = 170 # fleet size
W = 0.6 # <- to dziala
K = 35 # <- to dziala

C, E = {}, {}
for i in range(len(cities)):
    for j in range(i+1, len(cities)):
        C[(cities[i], cities[j])] = travel_time[(cities[i], cities[j])] * 10 / 3600
        C[(cities[j], cities[i])] = travel_time[(cities[i], cities[j])]* 10 / 3600
for i in range(len(cities)):
    for j in range(i+1, len(cities)):
        E[(cities[i], cities[j])] = travel_time[(cities[i], cities[j])] * 100 / 3600
        E[(cities[j], cities[i])] = travel_time[(cities[i], cities[j])] * 100 / 3600

#Create the pulp Model

# Next image is to help reference pulp model equations.
#display(Image(filename='eq_reference.jpg'))

# Create the problem variable to contain the problem data
model = LpProblem("MSFProblem", LpMinimize)

# Create X, the service, holding, and cycling arc flow variables for the fleet network.
# and the contraints relative to it

X = {}

# Create X, the service arc flow variable for the fleet network.
for city0 in cities:
    for city1 in cities:
        if city0 == city1:
            continue
        for t0 in range(taos):
            t1 = t0 + travel_time_tao[(city0, city1)]
            if t1 < taos:
                X[(city0, city1, t0)] = LpVariable(f'X_{city0}_{city1}_{t0}', lowBound=0, upBound=1, cat='Integer')

# Create X, the holding arc flow variable for the fleet network.
for city0 in cities:
    for t0 in range(taos):
        X[(city0, city0, t0)] = LpVariable(f'X_{city0}_{city0}_{t0}', lowBound=0, upBound=1, cat='Integer')

# Create X, the cycling arc flow variable for the fleet network.
for city0 in cities:
    X[(city0, city0, taos)] = LpVariable(f'X_{city0}_{city0}_{taos}', lowBound=0, upBound=1, cat='Integer')

# Equation #6
for city0 in cities:
    for t0 in range(taos):
        X[(city0, city0, t0)] = LpVariable(f'X_{city0}_{city0}_{t0}', lowBound=0, upBound=1, cat='Integer')
        model += X[(city0, city0, t0)] <= station_cap[city0], f'station_cap_{city0}_{t0}'

# Bus mass conservation @ cycle path. 
# Equation #4.
model += sum([X[(city0, city0, taos)] for city0 in cities]) <= F, f'buss_mass_conservation'

# Bus mass conservation @ statyion node.
# Equation #2
for city0 in cities:
    for t in range(taos):
        model += sum([X[(city0, city1, travel_time_tao[(city0, city1)])] for city1 in cities
                      if city1 != city0 and travel_time_tao[(city0, city1)] < taos])  \
                 + X[(city0, city0, t)] \
                 - (sum([X[(city1, city0, travel_time_tao[(city0, city1)])] for city1 in cities
                      if city1 != city0 and t - travel_time_tao[(city0, city1)] >= 0])  \
                 + X[(city0, city0, t)]) == 0, f'buss_mass_conservation_{city0}_{t}'

# Passanger network representation
Y = {}
# Create Y, the service arc flow variable for the fleet network.
for city0 in cities:
    for city1 in cities:
        if city0 == city1:
            continue
        for t0 in range(taos):
            t1 = t0 + travel_time_tao[(city0, city1)]
            if t1 < taos:
                Y[(city0, city1, t0)] = LpVariable(f'Y_{city0}_{city1}_{t0}', lowBound=0, cat='Integer')

# Passger - Bus Sync.
# Equation #5
for city0 in cities:
    for city1 in cities:
        if city0 == city1:
            continue
        for t0 in range(taos):
            t1 = t0 + travel_time_tao[(city0, city1)]
            if t1 < taos:
                model += Y[(city0, city1, t0)] <= K * W * X[(city0, city1, t0)], f'pass_mass_conservation_{city0}_{city1}_{t0}'

U = pass_scenarios.scenarios_avg 
# Equations #7
for city0 in cities:
    for city1 in cities:
        if city0 == city1:
            continue
        model += sum([Y[(city0, city1, t0)] for t0 in range(taos) if t0 + travel_time_tao[(city0, city1)] < taos]) \
                    <= U[(city0, city1)], f'daily_pass_mass_conservation_{city0}_{city1}'


# Declare objective of the model
model += sum([C[(city0, city1)]*X[(city0, city1, t0)] \
                for city0 in cities for city1 in cities if city0 != city1 for t0 in range(taos) \
                    if t0 + travel_time_tao[(city0, city1)] < taos and city0 != city1]) \
        - sum([E[(city0, city1)]*Y[(city0, city1, t0)]
               for city0 in cities for city1 in cities if city0 != city1 for t0 in range(taos) \
               if t0 + travel_time_tao[(city0, city1)] < taos]) # Equation #1

# The problem is solved using PuLP's choice of Solver
status = model.solve()

# print results
if LpStatus[results] == 'Optimal': print('The solution is optimal.')

for city0 in cities:
    for city1 in cities:
        if city0 == city1:
            continue
        for t0 in range(taos):
            t1 = t0 + travel_time_tao[(city0, city1)]
            if t1 < taos:
                val = X[(city0, city1, t0)].value()
            if val:
                print(f'{city0} -> {city1} @ {t0}: {val}')
    print('-'*(64+32))