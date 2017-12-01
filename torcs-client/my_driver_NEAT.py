from pytocl.driver import Driver
import matplotlib.pyplot as plt
from pytocl.car import State, Command
import subprocess
from subprocess import Popen, PIPE
import sys
sys.path.append('intelligence/neat-python')
sys.path.append('intelligence')
import neat
from neat.reporting import ReporterSet
from neat.math_util import mean
from neat.six_util import iteritems, itervalues
from NeatDriver import NEATDriver
import pickle

# imports from driver.py
import logging
import math
from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
_logger = logging.getLogger(__name__)

class MyDriver(Driver):
    def __init__(self, logdata=True):
        # Stuff from Driver class
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None

        # Stuff for NEAT
        self.total_ticks = 0
        self.warmup_calls = 0
        self.RESTART_AFTER = 19999
        self.EVAL_TIME = 2000
        self.FASTFORWARD = True
        try:
            self.neat_driver = pickle.load(open('neat_obj.p', 'rb'))
            self.net = neat.nn.FeedForwardNetwork.create(self.neat_driver.population.best_genome, self.neat_driver.config)
        except (OSError, IOError, AttributeError) as e:
            self.neat_driver = NEATDriver()


    def drive(self, carstate: State) -> Command:
        if self.total_ticks == 0 and self.FASTFORWARD:
            import keypress
        command = Command()
        self.total_ticks += 1
        if self.total_ticks > self.RESTART_AFTER or carstate.damage > 9000:
            command.meta = 1
            return command

        # Get sensor data and construct features for prediction
        speed = [carstate.speed_x*3.6]
        track_pos = [carstate.distance_from_center]
        angle = [carstate.angle]
        all_track_edges = list(carstate.distances_from_edge)
        front_edges = [all_track_edges[i] for i in [7,9,11]]
        track_edges = [all_track_edges[i] for i in [0,2,4,-1,-3,-5]]
        opponents_edges = list(carstate.opponents)
        features = speed+track_edges + [sum(front_edges)] + angle

        # Warm up phase
        if self.neat_driver.warmup_ticks > 0:
            return self.warm_up(command,carstate, front_edges)
        self.neat_driver.ticks += 1

        # Gather data for fitness function
        if sum([1 if edge < 1 else 0 for edge in track_edges]) > 0:
            self.neat_driver.ticks_off_track += 1
        self.neat_driver.speeds.append(speed[0])

        # Drive car using the network from genome i
        pred_command = self.neat_driver.net.activate(features)
        self.neat_driver.ticks += 1
        print(self.neat_driver.genome_idx, pred_command)

        # Drive car using best network
        # pred_command = self.net.activate(features)
        # print(pred_command)

        # For two output nodes
        if pred_command[0] < 0:
            command.brake = abs(pred_command[0])
            command.accelerator = 0
        else:
            command.accelerator = abs(pred_command[0])
            command.brake = 0
        command.steering = pred_command[1]

        # For three output nodes
        # command.accelerator, command.brake, command.steering = pred_command
        if front_edges[1] > 99:
            command.accelerator = 1
        self.shift(command, carstate)
        if command.gear > 3:
            command.gear = 3
        # print('[G{}]'.format(self.neat_driver.genome_idx), "acc: {}, brake: {}, steer: {}, gear: {}".format(command.accelerator,command.brake, command.steering, command.gear))

        # When evaluation time for individual is up or isn't moving enough
        dist_raced = carstate.distance_raced - self.neat_driver.dist_raced
        if self.neat_driver.ticks > self.EVAL_TIME or (self.neat_driver.ticks > 50 and sum(self.neat_driver.speeds[-10:-1]) < 2):
            # Evaluate current genome
            self.neat_driver.eval_genome(carstate)

            # Switch to next genome if there is one
            if len(self.neat_driver.genomes) > self.neat_driver.genome_idx+1:
                self.neat_driver.next_genome(carstate)

            # If not, start new generation
            else:
                self.neat_driver.update_population()
                # print('*'*10,'Updated population!','*'*10)

        # Log data
        if self.data_logger:
            self.data_logger.log(carstate, command)
        return command


    def warm_up(self, command,carstate, front_edges):
        # Warm-up in between individuals
        # if self.warmup_calls == 0:
        #     print('[Warm-up phase...]')
        self.warmup_calls += 1
        # Check whether the car is badly positioned on track
        if abs(carstate.angle) > 15 or abs(carstate.distance_from_center) > 0.5:
            # Abort mission
            if self.warmup_calls > 2000:
                command.meta = 1
                return command

            # Drive forward
            if sum(front_edges) > 8:
                return self.simple_bot(command, carstate)

            # Drive backward
            else:
                return self.reverse(command, carstate)
        # If car is in the good position, let it break
        elif carstate.speed_x*3.6 < -4:
            command.brake = 0.3
            return command

        # If the car is in a good position with a good speed
        else:
            self.neat_driver.damage = 0
            self.neat_driver.warmup_ticks = 0
            # print('[G{} starting after {} ticks...]'.format(self.neat_driver.genome_idx, self.warmup_calls))
            self.warmup_calls = 0
            command.gear = 1
        return command

    def simple_bot(self,command, carstate):
        command.gear = 1
        # command.accelerator = 0.3
        self.steer(carstate, 0.0, command)
        ACC_LATERAL_MAX = 6400 * 5
        v_x = min(10, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        v_x = 10
        self.accelerate(carstate, v_x, command)
        var = "[Bot] acc: {}, brake: {}, steer: {}".format(command.accelerator,command.brake, command.steering)
        # print(var)
        return command

    def shift(self, command, carstate):
        acceleration = command.accelerator
        # stabilize use of gas and brake:
        acceleration = math.pow(acceleration, 3)
        if acceleration > 0:
            if abs(carstate.distance_from_center) >= 1:
                # off track, reduced grip:
                acceleration = min(0.4, acceleration)
            command.accelerator = min(acceleration, 1)
            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1
        else:
            command.brake = min(-acceleration, 1)
        if carstate.rpm < 2500 and carstate.gear > 0:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1
        return command

    def reverse(self,command, carstate):
        command.gear = -1
        command.accelerator = 0.3
        command.brake = 0
        self.steer(carstate,0.0,command)
        # print('[Reversing] acc: {}, gear: {}'.format(command.accelerator, command.gear))
        return command

    def on_restart(self):
        pickle.dump(self.neat_driver, open('neat_obj.p', 'wb'))
        subprocess.call('./start.sh')
