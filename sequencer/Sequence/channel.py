from typing import List, Callable
import bisect
import sequencer.Sequence.event
 

class Channel:
    def __init__(self, name: str, card_number: int, channel_number: int, reset: bool, reset_value: float):
        self.name = name
        self.card_number = card_number
        self.channel_number = channel_number
        self.reset = reset
        self.reset_value = reset_value
        self.events = []
    
    def add_event(self, event):
        index = bisect.bisect_left([e.get_start_time() for e in self.events], event.get_start_time())
        self.events.insert(index, event)
    def sort_events(self):
        self.events.sort(key=lambda event: event.get_start_time())

    def __repr__(self) -> str:
        return (
            f"Channel(\n"
            f"   name={self.name},\n"
            f"   card_number={self.card_number},\n"
            f"   channel_number={self.channel_number},\n"
            f"   reset={self.reset},\n"
            f"   reset_value={self.reset_value},\n"
            f"   events={self.events}\n"
            f")"
        )
    
    #vituak function to get the channel attributes
    def get_channel_attributes(self):
        return {
            "name": self.name,
            "card_number": self.card_number,
            "channel_number": self.channel_number,
            "reset": self.reset,
            "reset_value": self.reset_value
        }

    def check_for_overlapping_events(self):
        self.events.sort(key=lambda event: event.get_start_time())
        for i in range(len(self.events) - 1):
            current_event = self.events[i]
            next_event = self.events[i + 1]
            if current_event.get_end_time() > next_event.get_start_time():
                raise ValueError(f"Events {current_event} and {next_event} on channel {self.name} overlap.")

    def detect_a_ramp(self,time_instance):
        # check if the time instance between two time instances has a ramp
        # get all the ramps in the channel: 
        for event in time_instance.events:
            if event.channel == self:
                return None
        for event in time_instance.ending_ramps:
            if event.channel == self:
                return None
        ramps = [event for event in self.events if isinstance(event.behavior, sequencer.Sequence.event.Ramp)] 
        for ramp in ramps:
            if ramp.start_time_instance.get_absolute_time() < time_instance.get_absolute_time() < ramp.end_time_instance.get_absolute_time():
                value = ramp.behavior.get_value_at_time(time_instance.get_absolute_time() - ramp.start_time_instance.get_absolute_time())
                return ramp, value
        return None
    
    def get_event_by_time_instance(self,time_instance):
        for event in self.events:
            if event.start_time_instance == time_instance:
                return (event,"start")
        for event in self.events:
            if event.end_time_instance == time_instance:
                return (event,"end")
        return None
    

             
    def __eq__(self, other: object) -> bool:
        return self.name == other.name and self.card_number == other.card_number and self.channel_number == other.channel_number and self.reset == other.reset and self.reset_value == other.reset_value

class Analog_Channel(Channel):
    def __init__(self, name: str, card_number: int, channel_number: int, reset: bool = False, reset_value: float = 0, default_voltage_func: Callable[[float], float] = lambda a: a, max_voltage: float = 10, min_voltage: float = -10,LIMIT=65535, RANGE=20, OFFSET=10):
        super().__init__(name, card_number, channel_number, reset, reset_value)
        self.default_voltage_func = default_voltage_func  # Should map from whatever value to a value between -10 to 10 
        self.max_voltage = max_voltage
        self.min_voltage = min_voltage
        self.LIMIT=LIMIT
        self.RANGE=RANGE
        self.OFFSET=OFFSET

    def discretize(self,val) : # Should map from -10:10 to 0:65535
        return (val + self.OFFSET)/self.RANGE * self.LIMIT
  

    def __repr__(self) -> str:
        return (
            f"Analog_Channel(\n"
            f"   name={self.name},\n"
            f"   card_number={self.card_number},\n"
            f"   channel_number={self.channel_number},\n"
            f"   reset={self.reset},\n"
            f"   reset_value={self.reset_value},\n"
            f"   max_voltage={self.max_voltage},\n"
            f"   min_voltage={self.min_voltage},\n"
            f"   events={self.events}\n"
            f")"
        )
    def get_channel_attributes(self):
        return {
            "type": "analog",
            "name": self.name,
            "card_number": self.card_number,
            "channel_number": self.channel_number,
            "reset": self.reset,
            "reset_value": self.reset_value,
            "max_voltage": self.max_voltage,
            "min_voltage": self.min_voltage,
            "LIMIT":self.LIMIT,
            "RANGE":self.RANGE,
            "OFFSET":self.OFFSET,
            # "default_voltage_func": self.default_voltage_func

        }

    
    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) and self.default_voltage_func == other.default_voltage_func and self.max_voltage == other.max_voltage and self.min_voltage == other.min_voltage


class Digital_Channel(Channel):
    def __init__(self, name: str, card_number: int, channel_number: int,  reset: bool = False, reset_value: float = 0):
        super().__init__(name, card_number, channel_number, reset, reset_value)

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) 

    def __repr__(self) -> str:
        return (
            f"Digital_Channel(\n"
            f"   name={self.name},\n"
            f"   card_number={self.card_number},\n"
            f"   channel_number={self.channel_number},\n"
            f"   reset={self.reset},\n"
            f"   reset_value={self.reset_value},\n"
            f"   events={self.events}\n"
            f")"
        )
    def get_channel_attributes(self):
        return {
            "type": "digital",
            "name": self.name,
            "card_number": self.card_number,
            "channel_number": self.channel_number,
            "reset": self.reset,
            "reset_value": self.reset_value,
        }
