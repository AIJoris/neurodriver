from pytocl.driver import Driver
from pytocl.car import State, Command
from intelligence.learning import train_network

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
        print(carstate.speed_x,carstate.speed_y,carstate.speed_z)
        command = Command()
        self.steer(carstate, 0.0, command)

        # ACC_LATERAL_MAX = 6400 * 5
        # v_x = min(80, math.sqrt(ACC_LATERAL_MAX / abs(command.steering)))
        v_x = 80

        self.accelerate(carstate, v_x, command)

        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command
