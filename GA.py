import concurrent
import random
from MLP import MLP
import math
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("LunarLander-v2")


class Genetic_Algorithm():

    def __init__(self, N=100, pcross=0.6, pmut=0.1, sigma=0.1, n_tour=4, n_iter=200, num_exp=20, bounds=[-1, 1], verbose=True):
        self.MLP = MLP([8, 6, 4])
        self.chromosome_len = len(self.MLP.to_chromosome())
        self.N = N  # Tamaño de la población
        self.bounds = bounds  # Límites del espacio de búsqueda
        self.pop = self.create_pop()  # Población
        self.pcross = pcross  # Probabilidad de cruce
        self.pmut = pmut  # Probabilidad de mutación
        self.sigma = sigma  # Desviación estándar de la mutación gaussiana
        self.n_tour = n_tour  # Número de participantes en el torneo
        self.n_iter = n_iter  # Numero de iteraciones
        self.num_exp = num_exp  # Número de experimentos por individuo
        self.verbose = verbose  # Imprimir resultados

    #############################
    #######     INIT      #######
    #############################

    def create_pop(self):
        """ Crea una población de N individuos codificados en modo locus """

        pop = [[random.uniform(self.bounds[0], self.bounds[1])
                for _ in range(self.chromosome_len)] for _ in range(self.N)]
        return pop

    #############################
    ######    MUTATION     ######
    #############################

    def mutate(self, chromosome):
        """ Mutación de un cromosoma """
        for i, gen in enumerate(chromosome):
            if random.random() < self.pmut:
                mutations = [self.mutate_gaussian]

                chromosome[i] = random.choice(mutations)(gen)

        return chromosome

    def mutate_multiply(self, gen):
        """ Mutación multiplicativa """
        #! DO NOT USE
        return gen * random.uniform(0.7, 1.5)

    def mutate_gaussian(self, gen):
        """ Mutación gaussiana """
        return gen + random.gauss(0, self.sigma)

    def mutate_sign(self, gen):
        """ Mutación de signo """
        #! DO NOT USE
        return -gen

    def mutate_init(self, gen):
        """ Mutación de inicialización """
        #! DO NOT USE
        return random.uniform(self.bounds[0], self.bounds[1])

    #############################
    ######    CROSSOVER    ######
    #############################

    def crossover_one_point(self, parent1, parent2):
        """Realiza un crossover entre dos padres."""

        if random.random() < self.pcross:
            crossover_point = random.randint(1, self.chromosome_len - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            child1, child2 = parent1, parent2

        return child1, child2

    def crossover_uniform(self, parent1, parent2):
        """Realiza un crossover uniforme entre dos padres."""
        #! DO NOT USE
        for i in range(self.chromosome_len):
            beta = random.random()
            if random.random() < self.pcross:
                parent1[i], parent2[i] = beta * parent1[i] + \
                    (1 - beta) * parent2[i], (1 - beta) * \
                    parent1[i] + beta * parent2[i]

        return parent1, parent2

    def crossover(self, parent1, parent2):
        """Realiza un crossover entre dos padres."""

        return random.choice([self.crossover_one_point])(
            parent1, parent2)

    #############################
    #####   LUNAR LANDER    #####
    #############################

    def policy(self, observation):
        s = self.MLP.forward(observation)
        action = np.argmax(s)
        return action

    def run(self):
        # observation, info = env.reset(seed=42)
        observation, info = env.reset()
        ite = 0
        racum = 0
        while True:
            action = self.policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            racum += reward

            if terminated or truncated:
                r = (racum+200) / 500
                return racum

    #############################
    ######    FITNESS    ######
    #############################

    def fitness(self, chromosome):
        """ Evalua un cromosoma """
        self.MLP.from_chromosome(chromosome)

        r = 0
        for _ in range(self.num_exp):
            r += self.run()

        return r/self.num_exp

    #############################
    ######    SELECTION    ######
    #############################

    def select(self, fitness):
        """ Selección de padres por torneo """
        participants = random.sample(range(self.N), self.n_tour)
        best = None
        for participant in participants:
            if best is None:
                best = participant
            elif (fitness[participant] > fitness[best]):
                best = participant
        return self.pop[best]

    def create_children(self, fitness):
        children = []
        while len(children) < self.N:
            parent1 = self.select(fitness)
            parent2 = self.select(fitness)
            while parent1 == parent2:
                parent2 = self.select(fitness)

            child1, child2 = self.crossover(parent1, parent2)

            self.mutate(child1)
            self.mutate(child2)

            children.append(child1)
            children.append(child2)

        return children

    #############################
    #######     PLOTS     #######
    #############################

    def plot_evolution(self, fitness_history):
        plt.plot(fitness_history['best'], label='Best Fitness')
        plt.plot(fitness_history['mean'], label='Mean Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution Over Generations')
        plt.legend()
        plt.grid(True)
        plt.show()

    def log_generation_info(self, generation, fitness_list, best_fitness, best_chromosome, file_path="evolution_log.txt"):
        with open(file_path, "a") as file:
            file.write(
                f"Generación {generation}, Fitness {max(fitness_list)}\n")
            file.write(
                f"Mejor fitness: {best_fitness}, Cromosoma: {best_chromosome}\n")

    def log_params(self, file_path="evolution_log.txt"):
        with open(file_path, "a") as file:
            file.write(f"---------------------------------------------------\n")
            file.write(f"Parámetros:\n")
            file.write(f"N: {self.N}\n")
            file.write(f"pcross: {self.pcross}\n")
            file.write(f"pmut: {self.pmut}\n")
            file.write(f"sigma: {self.sigma}\n")
            file.write(f"n_tour: {self.n_tour}\n")
            file.write(f"n_iter: {self.n_iter}\n")
            file.write(f"num_exp: {self.num_exp}\n")
            file.write(f"bounds: {self.bounds}\n")
            file.write("\n")

    #############################
    ######    ALGORITHM    ######
    #############################

    def evolve(self):
        """Evoluciona la población usando algoritmo genético."""
        if self.verbose:
            self.log_params(file_path="evolution_log.txt")

        fitness_list = [self.fitness(chromosome) for chromosome in self.pop]

        fitness_history = {'best': [max(fitness_list)], 'mean': [
            np.mean(fitness_list)]}

        best_chromosome = self.pop[np.argmax(fitness_list)]
        best_fitness = max(fitness_list)

        for generation in range(self.n_iter):

            # Crear nueva generación y calcular su fitness
            children = self.create_children(fitness_list)
            # fitness_children = [self.fitness(chromosome) for chromosome in children]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                fitness_children = list(executor.map(self.fitness, children))

            # Actualizar la población
            self.pop = children
            fitness_list = fitness_children

            # Registra el mejor fitness y el fitness promedio
            fitness_history['best'].append(max(fitness_list))
            fitness_history['mean'].append(np.mean(fitness_list))

            if max(fitness_list) > best_fitness:
                best_chromosome = self.pop[np.argmax(fitness_list)]
                best_fitness = max(fitness_list)

            # Seleccionar los N mejores individuos
            if self.verbose and (generation % 10 == 0):
                print("Generación", generation, "Fitness", max(fitness_list))
                print("Mejor fitness: ", best_fitness,
                      " Cromosoma: ", best_chromosome)
                self.log_generation_info(
                    generation, fitness_list, best_fitness, best_chromosome)

        self.plot_evolution(fitness_history)

        return best_chromosome, best_fitness
