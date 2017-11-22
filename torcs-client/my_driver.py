from pytocl.driver import Driver
import matplotlib.pyplot as plt
from pytocl.car import State, Command
from intelligence.learning import train_ff_network, train_rnn, train_lstm
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

USE_NET = "LSTM"
N_TIMESTEPS = 2
SCALE = False
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

        if USE_NET == "RNN":
            self.net = train_rnn(N_TIMESTEPS)
            var = Variable(torch.FloatTensor([0 for i in range(22)])).float()
            self.feature_timesteps = [var.view(1,var.size()[0]) for j in range(N_TIMESTEPS)]
            self.net, loss_vec = train_rnn(N_TIMESTEPS)
        elif USE_NET == "LSTM":
            self.net_steer, loss_vec_steer = train_lstm(N_TIMESTEPS, 1, use_tanh = True)
            self.net_speed, loss_vec_speed = train_lstm(N_TIMESTEPS, 2, use_tanh = False)


            # Initialize N_TIMESTEPS empty feature vectors
            self.feature_timesteps = [[0 for i in range(22)] for j in range(N_TIMESTEPS)]
            # self.feature_timesteps = [var.view(1,var.size()[0]) for j in range(N_TIMESTEPS)]
        elif USE_NET == "FF":
            # self.net, self.loss_vec = train_ff_network(scale = SCALE)
            self.net_speed, self.net_steer, self.loss_vec_speed, self.loss_vec_steer = train_ff_network(scale = SCALE)
            # plt.plot(list(range(len(self.loss_vec))),self.loss_vec)
            # print('Final loss:',self.loss_vec[-1])
            # plt.show()


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
        features = features.view(1,features.size()[0])

        if USE_NET == 'RNN':
            # For rnn keep track of previous states
            self.feature_timesteps.append(features)
            self.feature_timesteps.pop(0)
            hidden = self.net.init_hidden()
            for prev_features in self.feature_timesteps:
                pred, hidden = self.net(prev_features, hidden)
            pred = pred.data.numpy()[0]
        elif USE_NET == 'LSTM':
            self.feature_timesteps.append(speed+track_pos+angle+track_edges)
            self.feature_timesteps.pop(0)
            hidden = self.net.init_hidden()

            #pred = self.net(Variable(torch.FloatTensor(self.feature_timesteps)), N_TIMESTEPS)
            pred_steer = self.net_steer(Variable(torch.FloatTensor(self.feature_timesteps)), N_TIMESTEPS)
            pred_speed = self.net_speed(Variable(torch.FloatTensor(self.feature_timesteps)), N_TIMESTEPS)

            #print('*'*80,pred)
            #pred = pred.data[-1,:].numpy()
            # print(pred)
            #acc_pred = pred[0]
            #brake_pred = pred[1]/10
            #steer_pred = pred[2]

            acc_pred = pred_speed.data[0].numpy()
            steer_pred = pred_steer.data[0].numpy()

        elif USE_NET == 'FF':
            # pred = self.net(features).data.numpy()[0]
            acc_pred, brake_pred = self.net_speed(features).data.numpy()[0]
            steer_pred = self.net_steer(features).data.numpy()[0][0]

        print('Net: {}, acc: {}, brake: {}, steer: {}'.format(USE_NET, round(acc_pred,2), round(brake_pred,2), round(steer_pred,2)))

        # Prepare gear
        if command.accelerator > 0:
            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1
        # if carstate.rpm < 2500:
            # command.gear = carstate.gear - 1
        if not command.gear:
            command.gear = carstate.gear or 1

        # Prepare command
        command.accelerator = acc_pred
        # command.brake = brake_pred
        command.brake = 0
        command.steering = steer_pred

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
