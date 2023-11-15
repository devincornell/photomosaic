from __future__ import annotations
import typing
import dataclasses
import random
import itertools
#import numpy as np
import math

class Genome:
    pass

@dataclasses.dataclass
class GAConfig:
    fitness: typing.Callable[[Genome], float]
    population_size: int
    parent_percentile: int # percentile of pop to use as parents

    crossover_func: typing.Callable[[Genome, Genome, float], Genome]
    crossover_rate: float
    #crossover_prob: float

    mutate_func: typing.Callable[[Genome, float], Genome]
    mutation_rate: float
    #mutation_prob: float
    
    
    
    def iterate_population(self, population: typing.List[Genome]) -> typing.List[Genome]:
        '''Perform one iteration of the population, returning a new population.'''
        parent_fitness = self.get_parents(population)
        
        
        children = self.make_children(parents)
        new_pop = parents + children
    
    #def get_parents_randomized(self, population: typing.List[Genome]) -> typing.List[Genome]:
    #    parents = self.get_parents(population)
    #    random.shuffle(parents)
    #    return parents
    
        
    def get_parents(self, population: typing.List[Genome]) -> typing.List[Genome]:
        fitnesses = ((self.fitness(g), g) for g in population)
        return list(sorted(fitnesses, key=lambda x: x[0], reversed=True))[:self.num_parents]

    @property
    def num_children(self) -> int:
        return self.population_size - self.num_parents
    
    @property
    def num_parents(self) -> int:
        return math.ceil(self.parent_percentile * self.population_size / 100)

    ################## Population Transforms ##################
    def make_children(self, population: typing.List[Genome]) -> typing.List[Genome]:
        children = list()
        for g1, g2 in self.random_pairs(population, self.num_children):
            children.append(self.crossover(g1, g2))
        return children
    
    def crossover(self, g1: typing.List[Genome], g2: typing.List[Genome]) -> typing.List[Genome]:
        new_genome = self.crossover_func(g1, g2, self.crossover_rate)
        return self.mutate_func(new_genome, self.mutation_rate)

    def random_pairs(self, population: typing.List[Genome], n: int) -> typing.List[typing.Tuple[Genome, Genome]]:
        '''Return random pairs of genomes from the population.'''
        pairs = list(itertools.combinations(population, 2))
        return random.sample(pairs, n)

    ################## Setting Parameters ##################
    def set_probabilities(self, crossover_rate: float = None, mutation_rate: float = None) -> GAConfig:
        '''Set probability values.'''
        clone_params = dict()
        if crossover_rate is not None:
            clone_params['crossover_rate'] = crossover_rate
        if mutation_rate is not None:
            clone_params['mutation_rate'] = mutation_rate
            
        return self.clone(**clone_params)
    
    def clone(self, **new_params) -> GAConfig:
        '''Set new parameter values.'''
        return self.__class__(**{**dataclasses.asdict(self), **new_params})

@dataclasses.dataclass
class IterationStats:
    old_pop_size: int
    new_pop_size: int
    num_crossover: int
    num_mutate: int
    
    crossover_prob: float
    mutation_prob: float
    crossover_rate: float
    mutation_rate: float

