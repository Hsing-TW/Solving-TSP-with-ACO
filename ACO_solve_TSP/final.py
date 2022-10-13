import numpy as np
import pandas as pd
import random
import argparse
import logging
import csv


# Logging for File
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler('final.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


class City:
    def __init__(self, name, coordinate):
        self.name = name
        self.coordinate = coordinate
        logger.debug("City: CREATE - " + str(self))

    def __str__(self):
        return f"{self.name}[{self.coordinate[0]}, {self.coordinate[1]}]"

    __repr__ = __str__


class Cities:
    def __init__(self, filename):
        self.data = []
        self.load_data(filename)
        logger.debug("Cities: Data Size - " + str(len(self)))

    def load_data(self, filename):
        logger.debug("Cities: Load Data from " + filename)
        data = pd.read_csv(filename, header=None, sep=' ')
        for _, row in data.iterrows():
            self.data.append(City(row[0], np.array([row[1], row[2]])))

    @staticmethod
    def get_euclidean_distance(city_a, city_b):
        return np.sqrt(np.power(city_a.coordinate - city_b.coordinate, 2).sum())

    def compute_objective_value(self, orders):
        total_distance = 0
        for i in range(self.__len__()):
            total_distance += self.get_euclidean_distance(self.data[orders[i]], self.data[orders[i - 1]])
        return total_distance

    def to_cities_name(self, orders):
        return "[" + ", ".join(map(str, [self.data[order] for order in orders])) + "]"

    def get_cities_by_order(self, orders):
        return [self.data[order] for order in orders]

    def __str__(self):
        return "[" + ", ".join(map(str, self.data)) + "]"

    def __len__(self):
        return len(self.data)

    __repr__ = __str__


class AntSystem:
    def __init__(self, num_ants, pheromone_drop_amount, evaporate_rate,
                 pheromone_factor, visibility_factor, cities, iterations):
        self.num_ants = num_ants
        self.pheromone_drop_amount = pheromone_drop_amount
        self.evaporate_rate = evaporate_rate
        self.pheromone_factor = pheromone_factor
        self.visibility_factor = visibility_factor
        self.cities = cities
        self.iterations = iterations

        logger.debug(f"AntSystem: num_ants: {self.num_ants}, pheromone_drop_amount: {self.pheromone_drop_amount}, "
                     f"evaporate_rate: {self.evaporate_rate}, pheromone_factor: {self.pheromone_factor},"
                     f"visibility_factor: {self.visibility_factor}")

        self.one_solution = np.arange(len(self.cities), dtype=int)
        self.solutions = np.zeros((self.num_ants, len(self.cities)), dtype=int)
        for i in range(self.num_ants):
            for c in range(len(self.cities)):
                self.solutions[i][c] = c

        self.objective_value = np.zeros(self.num_ants)
        self.best_solution = np.zeros(len(self.cities), dtype=int)
        self.best_objective_value = np.finfo(np.float).max

        self.visibility = np.zeros((len(self.cities), len(self.cities)))
        self.pheromone_map = np.ones((len(self.cities), len(self.cities)))

        for from_ in range(len(self.cities)):
            for to in range(len(self.cities)):
                if from_ == to:
                    continue
                distance = self.cities.get_euclidean_distance(self.cities.data[from_], self.cities.data[to])
                self.visibility[from_][to] = 1 / distance

    def do_roulette_wheel_selection(self, fitness_list):
        sum_fitness = sum(fitness_list)
        transition_probability = [fitness / sum_fitness for fitness in fitness_list]

        rand = random.random()
        sum_prob = 0
        for i, prob in enumerate(transition_probability):
            sum_prob += prob
            if sum_prob >= rand:
                return i

    def update_pheromone(self):
        # evaporate hormones all the path
        self.pheromone_map *= (1 - self.evaporate_rate)

        # Add hormones to the path of the ants
        for solution in self.solutions:
            for j in range(len(self.cities)):
                city1 = solution[j]
                city2 = solution[j + 1] if j < len(self.cities) - 1 else solution[0]
                self.pheromone_map[city1, city2] += self.pheromone_drop_amount

    def _an_ant_construct_its_solution(self):
        candidates = [i for i in range(len(self.cities))]
        # random choose city as first city
        current_city_id = random.choice(candidates)
        self.one_solution[0] = current_city_id
        candidates.remove(current_city_id)

        # select best from candiate
        for t in range(1, len(self.cities) - 1):
            # best
            fitness_list = []
            for city_id in candidates:
                fitness = pow(self.pheromone_map[current_city_id][city_id], self.pheromone_factor) * \
                          pow(self.visibility[current_city_id][city_id], self.visibility_factor)
                fitness_list.append(fitness)

            next_city_id = candidates[self.do_roulette_wheel_selection(fitness_list)]
            candidates.remove(next_city_id)
            self.one_solution[t] = next_city_id

            current_city_id = next_city_id
        self.one_solution[-1] = candidates.pop()

    def each_ant_construct_its_solution(self):
        for i in range(self.num_ants):
            self._an_ant_construct_its_solution()
            for c in range(len(self.cities)):
                self.solutions[i][c] = self.one_solution[c]

            self.objective_value[i] = self.cities.compute_objective_value(self.solutions[i])

    def update_best_solution(self):
        for i, val in enumerate(self.objective_value):
            if val < self.best_objective_value:
                for n in range(len(self.cities)):
                    self.best_solution[n] = self.solutions[i][n]

                self.best_objective_value = val

    def train(self):
        for iteration in range(self.iterations):
            self.each_ant_construct_its_solution()
            self.update_pheromone()
            self.update_best_solution()

            # print
            logger.debug(f"========iteration {iteration + 1}========")
            logger.debug("best objective solution:")
            logger.debug(self.best_solution)
            logger.debug(self.cities.to_cities_name(self.best_solution))
            logger.debug(self.best_objective_value)

        logger.info("========final solution========")
        logger.info(self.cities.to_cities_name(self.best_solution))
        return self.cities.get_cities_by_order(self.best_solution)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Dataset location')
    parser.add_argument('-o', '--output', type=str, help='output filename', default="result.csv")
    parser.add_argument('-n', '--num_ants', help='number of ants', type=int, default=20)
    parser.add_argument('-d', '--pheromone_drop_amount', help='pheromone drop amount', type=float, default=0.001)
    parser.add_argument('-e', '--evaporate_rate', help='evaporate rate', type=float, default=0.1)
    parser.add_argument('-p', '--pheromone_factor', help='pheromone factor', type=int, default=1)
    parser.add_argument('-v', '--visibility_factor', help='visibility factor', type=int, default=3)
    parser.add_argument('-i', '--iterations', help='iterations', type=int, default=100)
    args = parser.parse_args()

    cities = Cities(args.filename)
    num_ants = args.num_ants
    pheromone_drop_amount = args.pheromone_drop_amount
    evaporate_rate = args.evaporate_rate
    pheromone_factor = args.pheromone_factor
    visibility_factor = args.visibility_factor
    iterations = args.iterations

    solver = AntSystem(num_ants, pheromone_drop_amount, evaporate_rate, pheromone_factor,
                       visibility_factor, cities, iterations)
    final_result = solver.train()

    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        for idx, result in enumerate(final_result, start=1):
            writer.writerow([idx, result.name, result.coordinate[0], result.coordinate[1]])


if __name__ == '__main__':
    main()
