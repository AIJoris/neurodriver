from pytocl.driver import Driver
from pytocl.car import State, Command
from intelligence.learning import train_ff_network, train_rnn, train_lstm
import torch
from torch.autograd import Variable
import numpy as np

# imports from driver.py
import logging
import math
from pytocl.analysis import DataLogWriter
from pytocl.car import State, Command, MPS_PER_KMH
from pytocl.controller import CompositeController, ProportionalController, \
    IntegrationController, DerivativeController
_logger = logging.getLogger(__name__)
import csv

USE_NET = "LSTM"
N_TIMESTEPS = 2
class MyDriver(Driver):
    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        command = Command()

        # Get sensor data and construct features for prediction
        speed = [carstate.speed_x]
        track_pos = [carstate.distance_from_center]
        angle = [carstate.angle]
        track_edges = list(carstate.distances_from_edge)

        self.steer(carstate, 0.0, command)
        ACC_LATERAL_MAX = 6400 * 5
        v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        v_x = 80
        self.accelerate(carstate, v_x, command)
        var = "acc: {}, brake: {}, steer: {}".format(command.accelerator,command.brake, command.steering)
        print('-', var)

        line = [command.accelerator, command.brake, command.steering, speed[0], track_pos[0], angle[0]]
        with open('out.csv', 'a') as out:
            writer = csv.writer(out)
            writer.writerow(line+track_edges)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command
