import numpy as np
from scipy.stats import truncnorm

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

trip_metrics = {} # slownik jest inicjowany w celu przechowywania wskaznikow podrozy 
# (srednia i odchylenie standardowe) dla kazdej pary miast. Wskazniki te sa obliczane na podstawie populacji miast.
for city0 in cities:
    for city1 in cities:
        if city0 == city1: continue
        mu = population[city0] * population[city1] / (population['Taipei'] * population['Taichung']) * 2650
        ds = 51 / 2566 * mu
        trip_metrics[(city0, city1)] = (mu, ds)

# sluzy do generowania scenariuszy zapotrzebowania na pasazerow
class PassangerDemandScenarios:
    def __init__(self, cities, S, M, n_iterations):
        self.cities = cities # lista miast
        self.S = S  # liczba scenariuszy
        self.M = M  # liczba pasazerow
        self.n_iterations = n_iterations # liczba iteracji generowania scenariuszy
        self.lower_bound = 0.25  # Lower truncation limit
        self.upper_bound = 2.25  # Upper truncation limit
        self.scenarios = {}
        self.scenarios_avg = {}

# method within the PassangerDemandScenarios class is responsible for generating the demand
# scenarios for each pair of cities.
    def generate(self):
        for city0 in cities:
            for city1 in cities:
                if city0 == city1: continue

                # For each pair of cities, the mean and standard deviation of the trip metrics are
                #  retrieved from the trip_metrics dictionary
                mean, std = trip_metrics[(city0, city1)]

                # Lower and upper bounds are calculated based on the mean values, and a truncated
                # normal distribution is created using the truncnorm function from scipy.stats.
                lower_bound = mean * self.lower_bound
                upper_bound = mean * self.upper_bound
                
                # Generate a sample from the truncated normal distribution
                try:
                    dist = truncnorm((lower_bound - mean) / std, (upper_bound - mean) / std, loc=mean, scale=std)
                    # Generate a sample from the truncated normal distribution
                    samples = dist.rvs(size=self.S, random_state=2023)
                except:
                    print(mean, std, (lower_bound - mean) / std, (upper_bound - mean) / std)
                
                # The generated samples are stored in the self.scenarios dictionary, where the key is a
                #  tuple representing the pair of cities, and the value is an array of samples.
                self.scenarios[(city0, city1)] = samples
                
                # Additionally, the average demand for each pair of cities is calculated
                #  and stored in the self.scenarios_avg dictionary.
                self.scenarios_avg[(city0, city1)] = np.mean(samples)

