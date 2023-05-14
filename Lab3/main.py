import random
import numpy as np
from deap import base, creator, tools
import math
from deap import algorithms
import matplotlib.pyplot as plt


def y(x):
    return -5 * math.cos(10 * x) * math.sin(3 * x) / math.sqrt(x)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    x = individual[0]
    if x < 0.0001:
        x = 0.0001
    return y(x),


toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("evaluate", evaluate)


def main():
    random.seed(42)

    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.6, mutpb=0.3, ngen=200, stats=stats, halloffame=hof,
                                       verbose=True)

    return hof[0]


def plot_function():
    x_values = np.linspace(0.0001, 10, 1000)
    y_values = [-5 * np.cos(10 * x) * np.sin(3 * x) / np.sqrt(x) for x in x_values]

    plt.plot(x_values, y_values, label='Y(x) = -5 * cos(10 * x) * sin(3 * x) / sqrt(x)')
    plt.xlabel('x')
    plt.ylabel('Y(x)')
    plt.title('Графік заданої функції')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    best_individual = main()
    print("Best solution:", best_individual)
    print("Y(best solution):", y(best_individual[0]))
    plot_function()
