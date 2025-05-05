from abc import ABC, abstractmethod
from enum import Enum
from sequencer.Sequence.time_frame import TimeInstance
from sequencer.Sequence.channel import Channel, Digital_Channel, Analog_Channel
import bisect
from typing import List, Optional, Union
import numpy as np

from sequencer.Sequence.utils import exp_to_func, find_min_max

class RampType(Enum):
    LINEAR = 'Linear'
    QUADRATIC = 'Quadratic'
    EXPONENTIAL = 'Exponential'
    LOGARITHMIC = 'Logarithmic'
    GENERIC = 'Generic'
    MINIMUM_JERK = 'Minimum Jerk'

    def __str__(self):
        return self.value



class EventBehavior(ABC):
    @abstractmethod
    def get_value_at_time(self, t: float) -> float:
        pass

class Jump(EventBehavior):
    def __init__(self, target_value: float):
        self.target_value = target_value

    def edit_jump(self, target_value: float):
        self.target_value = target_value


    def get_value_at_time(self, t: float) -> float:
        return self.target_value
    
    def __repr__(self) -> str:
        return f"Jump({self.target_value})"


class Ramp(EventBehavior):
    def __init__(self, start_time_instance:'TimeInstance',end_time_instance:'TimeInstance', ramp_type: RampType = RampType.LINEAR, start_value: float = 0, end_value: float = 1, func_text: str = None, resolution=0.001):
        # if start_value == end_value:
        #     raise ValueError("start_value and end_value must be different")
        
        if end_time_instance =="temp":
            pass 
        elif start_time_instance.get_absolute_time() >= end_time_instance.get_absolute_time():
            raise ValueError("start_time_instance must be less than end_time_instance")
        
        
        self.start_time_instance = start_time_instance
        self.end_time_instance = end_time_instance

        
        if ramp_type == RampType.EXPONENTIAL and (start_value == 0 or end_value == 0):
            raise ValueError("For exponential ramp, start_value and end_value must be non-zero")

        if resolution < 0.000001:
            raise ValueError("resolution must be at least 0.000001")
        
        self.ramp_type = ramp_type
        self.start_value = start_value
        self.end_value = end_value
        self.resolution = resolution
        self.func_text = func_text
        if self.ramp_type == RampType.GENERIC:
            if func_text:
                self.func = exp_to_func(func_text)
            else:
                raise ValueError("func can only be provided for generic ramp type, not for other ramp types")
        else:
            self._set_func()
    
    def get_duration(self):
        return self.end_time_instance.get_absolute_time() - self.start_time_instance.get_absolute_time()

    def _set_func(self):
        if self.end_time_instance == "temp":
            return
        
        duration = self.get_duration()
        if self.ramp_type == RampType.LINEAR:
            self.func = lambda t: self.start_value + (self.end_value - self.start_value) * (t / duration)
        elif self.ramp_type == RampType.QUADRATIC:
            self.func = lambda t: self.start_value + (self.end_value - self.start_value) * (t / duration)**2
        elif self.ramp_type == RampType.EXPONENTIAL:
            self.func = lambda t: self.start_value * (self.end_value / self.start_value) ** (t / duration)
        elif self.ramp_type == RampType.LOGARITHMIC:
            self.func = lambda t: self.start_value + (self.end_value - self.start_value) * (np.log10(t + 1) / np.log10(duration + 1))
        elif self.ramp_type == RampType.MINIMUM_JERK:
            self.func = lambda t: self.start_value + (self.end_value - self.start_value) * (10 * (t/duration)**3 - 15 * (t/duration)**4 + 6 * (t/duration)**5)
        else:
            raise ValueError("Invalid ramp type")
    def get_start_value(self):
        return self.func(0)

    def get_end_value(self):
        return self.func(self.get_duration())
    def edit_ramp(self, 
                        start_time_instance: Optional['TimeInstance'] = None,
                        end_time_instance: Optional['TimeInstance'] = None, 
                        ramp_type: Optional[RampType] = None,
                        start_value: Optional[float] = None,
                        end_value: Optional[float] = None,
                        func_text: str = None,
                        resolution: Optional[float] = None):
        
        new_start_time_instance = start_time_instance if start_time_instance is not None else self.start_time_instance
        new_end_time_instance = end_time_instance if end_time_instance is not None else self.end_time_instance

        if ramp_type is not None:
            try:
                new_ramp_type = ramp_type
            except KeyError:
                # Handle the case where the ramp_type string is not valid
                raise ValueError(f"Invalid ramp type: {ramp_type}")
        else:
            new_ramp_type = self.ramp_type

        new_start_value = start_value if start_value is not None else self.start_value
        new_end_value = end_value if end_value is not None else self.end_value
        new_resolution = resolution if resolution is not None else self.resolution
        
        # if new_start_value == new_end_value:
        #     raise ValueError("start_value and end_value must be different")
        
        if new_start_time_instance.get_absolute_time() >= new_end_time_instance.get_absolute_time():
            raise ValueError("duration must be negative")
        
        if new_ramp_type == RampType.EXPONENTIAL and (new_start_value == 0 or new_end_value == 0):
            raise ValueError("For exponential ramp, start_value and end_value must be non-zero")
        
        if new_resolution < 0.000001:
            raise ValueError("resolution must be at least 0.000001")
        
        # Apply changes only after validation
        self.start_time_instance = new_start_time_instance
        self.end_time_instance = new_end_time_instance
        self.ramp_type = new_ramp_type
        self.start_value = new_start_value
        self.end_value = new_end_value
        self.resolution = new_resolution

        if func_text:
            self.func_text = func_text
            self.func = exp_to_func(func_text)
        else:
            self._set_func()


    def get_value_at_time(self, t: float) -> float:
        if 0 <= t <= self.get_duration():
            return self.func(t)
        else:
            return self.func(self.get_duration())
    
    def __repr__(self) -> str:
        return f"Ramp({self.get_duration()}, {self.ramp_type.value}, {self.get_start_value()}, {self.get_end_value()})"
class Digital(EventBehavior):
    def __init__(self, target_value: float):
        if target_value not in [0, 1]:
            raise ValueError("target_value must be 0 or 1")
        self.target_value = target_value

    def edit_digital(self, target_value: float):
        if target_value not in [0, 1]:
            raise ValueError("target_value must be 0 or 1")
        self.target_value = target_value

    def get_value_at_time(self, t: float) -> float:
        return self.target_value
    
    def __repr__(self) -> str:
        return f"Digital({self.target_value})"



class Event:
    def __init__(self, channel: Optional[Union[Digital_Channel, Analog_Channel]] , behavior: EventBehavior,comment:str ="",start_time_instance: Optional['TimeInstance'] = None,end_time_instance: Optional['TimeInstance'] = None):


        self.channel = channel
        self.behavior = behavior
        self.comment= comment
        self.is_sweept = False
        self.reference_original_event = self
        self.associated_parameters = []
        # check if the start_time_instance and channels already have events, raise an error if there is already an event at the same time and channel
        for event in start_time_instance.events:
            if event.channel == channel:
                raise ValueError(f"Event already exists on channel {channel.name} at time instance {start_time_instance.name}")
        if end_time_instance and end_time_instance != "temp":
            for event in end_time_instance.events:
                if event.channel == channel:
                    raise ValueError(f"Event already exists on channel {channel.name} at time instance {end_time_instance.name}")
            
        if start_time_instance is not None:
            self.start_time_instance = start_time_instance
        else:
            raise ValueError("start_time_instance is required")
        
        if isinstance(behavior, Ramp):
            if end_time_instance=="temp":
                pass
            else:
                if end_time_instance is None:
                    raise ValueError("end_time_instance is required for Ramp events")
                else :
                    self.end_time_instance = end_time_instance
        else:
            self.end_time_instance = self.start_time_instance

            
        
        


        if isinstance(behavior, Ramp) and not isinstance(behavior, Jump):
            
            self.end_time_instance =  end_time_instance 
        else:
            self.end_time_instance = self.start_time_instance

        if end_time_instance != "temp":
            self.check_for_overlap(channel, behavior, self.get_start_time(), self.get_end_time())
            self.check_if_generic_ramp_is_valid()
        


        self.children: List[Event] = []
        index = bisect.bisect_left([e.get_start_time() for e in self.channel.events], self.get_start_time())
        self.channel.events.insert(index, self)
        self.start_time_instance.add_event(self)
        if self.end_time_instance != "temp":
            if isinstance(behavior, Ramp) and not isinstance(behavior, Jump) and self.end_time_instance != self.start_time_instance:
                self.end_time_instance.add_ending_ramp(self)
    
    def check_if_generic_ramp_is_valid(self):
        if isinstance(self.behavior, Ramp) and self.behavior.ramp_type == RampType.GENERIC:
            # get the min and max values of the function
            min_value, max_value, _, _ = find_min_max(self.behavior.func, 0, self.behavior.get_duration())
            if min_value < self.channel.min_voltage or max_value > self.channel.max_voltage:
                raise ValueError(f"Generic ramp values are out of range for channel {self.channel.name} with min voltage {self.channel.min_voltage} and max voltage {self.channel.max_voltage}")
        else:
            return
            
    def get_start_time(self):
        return self.start_time_instance.get_absolute_time()

    def get_end_time(self):
        return self.end_time_instance.get_absolute_time()
    
    

    def get_event_attributes(self):
        if isinstance(self.behavior, Jump):
            
                
            return {
                "type": "jump",
                "jump_target_value": self.behavior.target_value,
                "start_time": self.get_start_time(),
                "channel_name": self.channel.name,
                "comment":self.comment,
                "start_time_instance":self.start_time_instance.name,
                "is_sweept":self.is_sweept
                }

        elif isinstance(self.behavior, Ramp):
            return {
                "type": "ramp",
                "ramp_type": self.behavior.ramp_type.value,
                "start_value": self.behavior.start_value,
                "end_value": self.behavior.end_value,
                "func": self.behavior.func_text,
                "resolution": self.behavior.resolution,
                "channel_name": self.channel.name,
                "comment":self.comment,
                "start_time_instance":self.start_time_instance.name,
                "end_time_instance":self.end_time_instance.name,
                "is_sweept":self.is_sweept
                }   
        elif isinstance(self.behavior, Digital):
            return {
                "type": "digital",
                "target_value": self.behavior.target_value,
                "start_time": self.get_start_time(),
                "channel_name": self.channel.name,
                "comment":self.comment,
                "start_time_instance":self.start_time_instance.name,
                "is_sweept":self.is_sweept,
                }



    def check_for_overlap(self, channel: Channel, behavior: EventBehavior, start_time: float, end_time: float):
        for event in channel.events:
            if not (end_time < event.get_start_time() or start_time > event.get_end_time()):
                if isinstance(behavior, Jump) and isinstance(event.behavior, Jump):
                    if start_time == event.get_start_time():
                        raise ValueError(f"Cannot have more than one jump at the same time on channel {channel.name}.")
                elif isinstance(behavior, Jump) and isinstance(event.behavior, Ramp):
                    if start_time != event.get_end_time():
                        raise ValueError(f"Jump events can only be added at the end of a ramp on channel {channel.name}.")
                else:
                    raise ValueError(f"Events on channel {channel.name} overlap with existing event {event.behavior} from {event.get_start_time()} to {event.get_end_time()}.")