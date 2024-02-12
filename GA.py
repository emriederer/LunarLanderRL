import concurrent
import random
from MLP import MLP
import math
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("LunarLander-v2")


class Genetic_Algorithm():

    def __init__(self, N=100, crossover_method=1, pcross=1/80, pmut=0.1, sigma=0.1, selection=1, n_tour=4, n_iter=5, num_exp=1, elitism=1, bounds=[-20, 20]):
        self.MLP = MLP([8, 6, 4])
        self.N = N  # Tamaño de la población
        self.pop = self.create_pop()  # Población
        # Método de cruce [1: un punto, 2: uniforme]
        self.crossover_method = crossover_method
        self.pcross = pcross  # Probabilidad de cruce
        self.pmut = pmut  # Probabilidad de mutación
        self.sigma = sigma  # Desviación estándar de la mutación
        # Método de selección de padres [1: torneo, 2: Boltzmann]
        self.selection = selection
        self.n_tour = n_tour  # Número de participantes en el torneo
        self.n_iter = n_iter  # Numero de iteraciones
        # Número de individuos que pasan directamente a la siguiente generación
        self.elitism = elitism
        self.num_exp = num_exp  # Número de experimentos por individuo
        self.bounds = bounds

    def create_pop(self):
        """ Crea una población de N individuos codificados en modo locus """
        chromosome_len = len(self.MLP.to_chromosome())
        pop = [[random.uniform(-math.pi, math.pi)
                for _ in range(chromosome_len)] for _ in range(self.N)]
        return pop

    def mutate(self, chromosome):
        """ Mutación gausiana """
        for i in range(len(chromosome)):
            if random.random() < self.pmut:
                chromosome[i] += random.gauss(0, self.sigma)
        return chromosome

    def crossover_uniform(self, parent1, parent2):
        """Realiza un crossover uniforme entre dos padres."""
        for i in range(len(parent1)):
            beta = random.random()
            if random.random() < self.pcross:
                parent1[i], parent2[i] = beta * parent1[i] + \
                    (1 - beta) * parent2[i], (1 - beta) * \
                    parent1[i] + beta * parent2[i]

        return parent1, parent2

    def crossover_one_point(self, parent1, parent2):
        """Realiza un crossover de un punto entre dos padres."""
        if random.random() < self.pcross:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            child1, child2 = parent1, parent2

        return child1, child2

    def crossover(self, parent1, parent2):
        if self.crossover_method == 1:
            return self.crossover_one_point(parent1, parent2)
        elif self.crossover_method == 2:
            return self.crossover_uniform(parent1, parent2)
        else:
            raise ValueError("Método de cruce no válido")

    def tournament(self, fitness):
        """ Selección de padres por torneo """
        participants = random.sample(self.pop, self.n_tour)
        best = None
        for participant in participants:
            if best is None:
                best = participant
            elif (fitness[self.pop.index(participant)] < fitness[self.pop.index(best)]):
                best = participant
        return best

    def select(self, fitness):
        if self.selection == 1:
            return self.tournament(fitness)
        else:
            raise ValueError("Método de selección no válido")

    def create_children(self, fitness):
        children = []
        while len(children) < self.N:
            parent1 = self.select(fitness)
            parent2 = parent1
            while parent1 == parent2:
                parent2 = self.select(fitness)
            child1, child2 = self.crossover_uniform(parent1, parent2)

            self.mutate(child1)
            self.mutate(child2)

            children.append(child1)
            children.append(child2)

        return children

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

    def fitness(self, chromosome):
        """ Evalua un cromosoma """
        self.MLP.from_chromosome(chromosome)

        r = 0
        for _ in range(self.num_exp):
            r += self.run()

        return r/self.num_exp

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

    def evolve(self):
        """Evoluciona la población usando algoritmo genético."""

        fitness_list = [self.fitness(chromosome) for chromosome in self.pop]
        fitness_history = {'best': [max(fitness_list)], 'mean': [
            np.mean(fitness_list)]}

        best_chromosome = self.pop[np.argmax(fitness_list)]
        best_fitness = max(fitness_list)

        for generation in range(self.n_iter):

            # Ordenar los individuos de la generación anterior
            sorted_indices = np.argsort(fitness_list)[::-1]

            # Crear nueva generación y calcular su fitness
            children = self.create_children(fitness_list)
            # fitness_children = [self.fitness(chromosome) for chromosome in children]
            with concurrent.futures.ProcessPoolExecutor() as executor:
                fitness_children = list(executor.map(self.fitness, children))

            # Elitismo: seleccionar los N mejores individuos
            if self.elitism:
                new_pop = [self.pop[i] for i in sorted_indices[:self.elitism]]
                fitness_list = [fitness_list[i]
                                for i in sorted_indices[:self.elitism]]

                new_pop = new_pop + \
                    [children[i] for i in np.argsort(fitness_children)[
                        ::-1][:self.N - self.elitism]]
                fitness_list = fitness_list + \
                    [fitness_children[i] for i in np.argsort(
                        fitness_children)[::-1][:self.N - self.elitism]]

            else:
                new_pop = [children[i] for i in np.argsort(
                    fitness_children)[::-1][:self.N - self.elitism]]
                fitness_list = [fitness_children[i] for i in np.argsort(
                    fitness_children)[::-1][:self.N - self.elitism]]

            # Actualizar la población
            self.pop = new_pop

            # Registra el mejor fitness y el fitness promedio
            fitness_history['best'].append(max(fitness_list))
            fitness_history['mean'].append(np.mean(fitness_list))

            if max(fitness_list) > best_fitness:
                best_chromosome = self.pop[np.argmax(fitness_list)]
                best_fitness = max(fitness_list)

            # Seleccionar los N mejores individuos
            if generation % 10 == 0:
                print("Generación", generation, "Fitness", max(fitness_list))
                print("Mejor fitness: ", best_fitness,
                      " Cromosoma: ", best_chromosome)
                self.log_generation_info(
                    generation, fitness_list, best_fitness, best_chromosome)

        self.plot_evolution(fitness_history)

        return best_chromosome, best_fitness
