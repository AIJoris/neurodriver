from pytocl.driver import Driver
from pytocl.car import State, Command
from intelligence.learning import train_network
import torch
from torch.autograd import Variable

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
        self.steering_ctrl = CompositeController(
            ProportionalController(0.4),
            IntegrationController(0.2, integral_limit=1.5),
            DerivativeController(2)
        )
        self.acceleration_ctrl = CompositeController(
            ProportionalController(3.7),
        )
        self.data_logger = DataLogWriter() if logdata else None
        self.net = train_network()

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
        features = Variable(torch.FloatTensor(speed+track_pos+angle+track_edges)).float()

        # Predict
        pred = self.net(features)
        acc_pred = pred[0]
        brake_pred = pred[1]
        steer_pred = pred[2]

        # Prepare gear
        if command.accelerator > 0:
            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1
        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1

        # Prepare command
        command.accelerator = acc_pred.data[0]
        command.brake = brake_pred.data[0]/10
        command.steering = steer_pred.data[0]
        # var = "acc: {}, brake: {}, steer: {}".format(command.accelerator,command.brake, command.steering)
        # print('+', var)

        # self.steer(carstate, 0.0, command)
        # ACC_LATERAL_MAX = 6400 * 5
        # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        # v_x = 80
        # self.accelerate(carstate, v_x, command)
        # var = "acc: {}, brake: {}, steer: {}".format(command.accelerator,command.brake, command.steering)
        # print('-', var)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command
