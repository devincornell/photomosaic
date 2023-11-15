
import typing
import dataclasses

class Genome:
    pass

@dataclasses.dataclass
class GAConfig:
    crossover: typing.Callable[[Genome, Genome], Genome]
    mutate: typing.Callable[[Genome], Genome]
    fitness: typing.Callable[[Genome], float]
    population_size: int
    max_generations: int
    mutation_rate: float
    crossover_rate: float
    
    


