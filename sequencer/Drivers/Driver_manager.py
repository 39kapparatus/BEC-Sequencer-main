# This is an undeveloped class that will be used to manage the drivers for the different devices. The idea is to have a dictionary of drivers, where the key is the device name and the value is the driver object. This class will be used to segregate the events of a sequence to the corresponding driver.


from sequencer.Drivers.ADwin_Modules import ADwin_Driver
from sequencer.Drivers.FunctionGenerator_Modules import FunctionGenerator_Driver

from sequencer.Sequence.sequence import Sequence

class Driver_manager:
    def __init__(self):
        self.drivers = {}
        self.drivers['ADwin'] = ADwin_Driver()
        self.drivers['FunctionGenerator'] = FunctionGenerator_Driver()


    def segrgate(sequence):
        for step in sequence.steps:
            if step.device in self.drivers:
                self.drivers[step.device].add_step(step)
            else:
                print('Error: Device not found')
                return False
        return True


