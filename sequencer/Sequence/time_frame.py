import copy
from typing import  List, Optional




class TimeInstance:
    def __init__(self, name: str, parent: Optional['TimeInstance'] = None, relative_time: int = 0):
        self.name: str = name
        self.parent: Optional['TimeInstance'] = parent
        self.relative_time: int = relative_time
        self.children: List['TimeInstance'] = []
        self.events = []
        self.ending_ramps = []
        self.is_sweept = False
        if parent:
            parent.children.append(self)
        else :
            # check if the relative time is not 0 for the root time instance
            if relative_time != 0:
                raise ValueError("Relative time must be 0 for the root time instance")
            
    def add_event(self, event) -> None:
        self.events.append(event)
    def add_ending_ramp(self, event) -> None:
        self.ending_ramps.append(event)

    def get_child_time_instance_by_name(self, name: str) -> Optional['TimeInstance']:
        for child in self.children:
            if child.name == name:
                return child
        return
    def get_descendant_by_name(self, name: str) -> Optional['TimeInstance']:
        for child in self.children:
            if child.name == name:
                return child
            descendant = child.get_descendant_by_name(name)
            if descendant:
                return descendant
        return None
    
    
    def edit_parent(self, new_parent_name: str) -> None:
        # check if the new parent is not the current time instance
        if self.name == new_parent_name:
            raise ValueError("Cannot set the parent to self")
        # check if the parent is not a descendant of the current time instance
        new_parent = self.get_time_instance_by_name(new_parent_name)
        
        if self == new_parent:
            raise ValueError("Cannot set the parent to self")
        # check if the parent is not a descendant of the current time instance
        if new_parent in self.get_all_children():
            raise ValueError("Cannot set the parent to a descendant")
        # check the parent belongs to the same sequence
        if self.get_root() != new_parent.get_root():
            raise ValueError("Cannot set the parent to a time instance from a different sequence")
        # check if editing the parent will result in negative absolute time
        if new_parent.get_absolute_time() + self.relative_time < 0: 
            raise ValueError("new relative time will result in negative absolute time")
        
        if self.parent:
            self.parent.children.remove(self)
        self.parent = new_parent
        new_parent.children.append(self)

    def edit_name(self, new_name: str) -> None:
        
        root = self.get_root()
        all_children = root.get_all_children()
        
        for child in all_children:
            if child.name == new_name:
                raise Exception("Name already exists")
        
        self.name = new_name

    def edit_relative_time(self, new_relative_time: int) -> None:
        # check if the new relative time will result in negative absolute time
        if self.parent is not None:
            if self.parent.get_absolute_time() + new_relative_time < 0:
                raise ValueError("new relative time will result in negative absolute time")
        self.relative_time = new_relative_time

    def get_absolute_time(self) -> int:
        if self.parent is None:
            return self.relative_time
        return self.relative_time + self.parent.get_absolute_time()

    def add_child_time_instance(self, name: str, relative_time: int) -> 'TimeInstance':
        #  check if the relative time will result in negative absolute time
        if self.get_absolute_time() + relative_time < 0:
            raise ValueError("new relative time will result in negative absolute time")
        return TimeInstance(name, parent=self, relative_time=relative_time)

    def delete_self(self) -> None:
        
        if self.parent is None:
            raise Exception("Cannot delete root time frame")

        # check if deleting self will result in negative absolute time for any of the children

        # get all children of the current time instance
        children = self.get_all_children()
        delta = - self.relative_time 
        for child in children:
            if child.get_absolute_time() + delta < 0:
                raise ValueError("Deleting this time instance will result in negative absolute time for one of the children")
            
        for child in self.children:
            child.parent = self.parent
            self.parent.children.append(child)
        self.parent.children.remove(self)
        self.children = []

    def get_root(self) -> 'TimeInstance':
        if self.parent is None:
            return self
        return self.parent.get_root()

    def get_all_children(self) -> List['TimeInstance']:
        children: List['TimeInstance'] = []
        for child in self.children:
            children.append(child)
            children += child.get_all_children()
        return children
    def get_all_time_instances(self) -> List['TimeInstance']:
        time_instances: List['TimeInstance'] = []
        time_instances.append(self)
        time_instances += self.get_all_children()
        return time_instances
    
    def get_all_time_instances_after_me(self) -> List['TimeInstance']:
        # get root time instance
        root = self.get_root()
        # get all time instances in the sequence
        all_time_instances = root.get_all_time_instances()
        sort_time_instances = sorted(all_time_instances, key=lambda x: x.get_absolute_time())
        index = sort_time_instances.index(self)
        return sort_time_instances[index + 1:]
    

    
    def get_time_instance_by_name(self, name: str) -> Optional['TimeInstance']:
        if self.name == name:
            return self
        # get all children of the current time instance
        children = self.get_root().get_all_children()
        for child in children:
            if child.name == name:
                return child
        return None
    

    def print_children(self, depth: int = 0) -> None:
        print("  " * depth + str(self))
        for child in self.children:
            child.print_children(depth + 1)

    def get_events_string(self, depth: int = 0) -> str:
        events_string: str = ""
        for event in self.events:
            events_string += "  " * depth + str(event) + "\n"
        for child in self.children:
            events_string += child.get_events_string(depth + 1)
        return events_string

    def create_a_deep_copy_of_all_frames(self) -> 'TimeInstance':
        root = self.get_root()
        new_root = copy.deepcopy(root)
        return new_root

    def __str__(self) -> str:
        return f"TimeInstance(name={self.name}, absolute_time={self.get_absolute_time()}, events={self.events}, relative_time={self.relative_time})"

    def __repr__(self) -> str:
        return self.__str__()
