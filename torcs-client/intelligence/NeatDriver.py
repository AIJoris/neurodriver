import sys
sys.path.append('neat-python')
import neat
from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues

WARMUP_TICKS = 100
MAX_GENERATIONS = 500
class NEATDriver():
    def __init__(self):
        ## Load NEAT configuration.
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             'intelligence/config-feedforward')

        # Create the population
        self.population = neat.Population(self.config)

        # Add a stdout reporter to show progress in the terminal.
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        self.population.add_reporter(neat.Checkpointer(50))

        # Initialize values for fitness function
        self.ticks_off_track = 0
        self.ticks = 0
        self.speeds = []
        self.damage = 0
        self.dist_raced = 0
        self.generation = 0
        self.genomes = list(iteritems(self.population.population))
        self.genome_idx = 0
        self.net = neat.nn.FeedForwardNetwork.create(self.genomes[0][1], self.config)
        self.max_generations = MAX_GENERATIONS
        self.warmup_ticks = WARMUP_TICKS

    def eval_genome(self, carstate):
        try:
            avr_speed = sum(self.speeds)/float(len(self.speeds))
        except ZeroDivisionError:
            avr_speed = 0
        dist_raced = carstate.distance_raced - self.dist_raced
        avr_speed = dist_raced/float(self.ticks)
        damage = (carstate.damage - self.damage)
        fitness = 1000 - (self.ticks_off_track) + dist_raced + 1000 * avr_speed
        self.genomes[self.genome_idx][1].fitness = fitness
        print('[G{}] speed: {}, dist: {}, dmg {}, offtrack: {} ,fitness: {}'.format(\
            self.genome_idx, round(avr_speed,2), round(dist_raced,2), damage, self.ticks_off_track, round(self.genomes[self.genome_idx][1].fitness,2)))

    def next_genome(self, carstate):
        self.genome_idx += 1
        self.net = neat.nn.FeedForwardNetwork.create(self.genomes[self.genome_idx][1], self.config)
        self.speeds = []
        self.damage = 0
        self.dist_raced = carstate.distance_raced
        self.ticks = 0
        self.ticks_off_track = 0
        self.warmup_ticks = WARMUP_TICKS

    def update_population(self):
        self.population.reporters.start_generation(self.population.generation)

        # Gather and report statistics.
        best = None
        for g_id, g in self.genomes:
            if best is None or g.fitness > best.fitness:
                best = g

        self.population.reporters.post_evaluate(self.population.config, self.population.population, self.population.species, best)

        # Track the best genome ever seen.
        if self.population.best_genome is None or best.fitness > self.population.best_genome.fitness:
            self.population.best_genome = best

        # Stop after max generation is reached
        if self.population.generation > self.max_generations:
            return False

        if not self.population.config.no_fitness_termination:

          # End if the fitness threshold is reached.
          fv = self.population.fitness_criterion(g.fitness for g in itervalues(self.population.population))
          if fv >= self.population.config.fitness_threshold:
              self.population.reporters.found_solution(self.population.config, self.population.generation, best)
              return False

        # Create the next generation from the current generation.
        self.population.population = self.population.reproduction.reproduce(self.population.config, self.population.species,
                                                    self.population.config.pop_size, self.population.generation)

        # Check for complete extinction.
        if not self.population.species.species:
          self.population.reporters.complete_extinction()

          # If requested by the user, create a completely new population,
          # otherwise raise an exception.
          if self.population.config.reset_on_extinction:
              self.population.population = self.population.reproduction.create_new(self.population.config.genome_type,
                                                             self.population.config.genome_config,
                                                             self.population.config.pop_size)
          else:
              raise CompleteExtinctionException()

        # Divide the new population into species.
        self.population.species.speciate(self.population.config, self.population.population, self.population.generation)
        self.population.reporters.end_generation(self.population.config, self.population.population, self.population.species)

        self.population.generation += 1
        self.genome_idx = 0
        self.genomes = list(iteritems(self.population.population))
        return True
