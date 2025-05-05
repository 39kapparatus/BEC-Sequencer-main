#! /usr/bin/python3
# -*- coding: cp1252 -*-



import ADwin
import os
import sys 
from copy import deepcopy 
import numpy as np
from sequencer.Sequence.sequence import Sequence
from sequencer.Sequence.channel import Digital_Channel, Analog_Channel
from sequencer.Sequence.event import  Jump, Ramp

import time


def encode_channel(type: int, channel: int, card: int) -> int:
    """
    Encodes three integers into a single integer using bitwise operations.
    
    Args:
        type (int): The first integer (1 to 32).
        y (int): The second integer (1 to 20).
        z (int): The third integer (1 to 99).
    
    Returns:
        int: The encoded integer.
    """
    return (type << 12) | (channel << 7) | card



def calculate_time_ranges(all_events):

    time_ranges = []
    


    # Initialize variables
    current_time = 0
    active_events = []
    all_times = sorted(set([event.get_start_time() for event in all_events] + [event.get_end_time() for event in all_events]))

    # Iterate over all significant times
    for time in all_times:
        # Create a tuple for the time range and the active events before updating the active events
        if current_time != time:
            time_ranges.append(((current_time, time), active_events.copy()))
        
        # Remove events that have ended at or before the current time
        active_events = [event for event in active_events if event.get_end_time() > time]

        # Add events that are starting at the current time
        starting_events = [event for event in all_events if event.get_start_time() == time and event.get_end_time() != time]
        active_events.extend(starting_events)

        # Handle instantaneous events
        instant_events = [event for event in all_events if event.get_start_time() == event.get_end_time() == time]
        if instant_events:
            time_ranges.append(((time, time), active_events + instant_events))

        current_time = time

    return time_ranges



            
            
def calculate_sequence_data_eff(sequence: Sequence,adwin_driver) -> None:
    
    
    time_resolution = adwin_driver.proceessdelay_unit*adwin_driver.processdelay # 10**-9 * 1000 = 1 micro seconds
    #print("Time resolution: ", time_resolution)


    channel_number = np.array([])
    channel_value= np.array([])
    update_list = np.array([]) 


    processdelay_times = []    
    processdelay_value_list = []    

    all_events = deepcopy(sequence.get_all_events())
    time_ranges=calculate_time_ranges(all_events)

    start_time2 = time.time()

    for time_range in time_ranges:
        if time_range[-1]:
            if time_range[0][0]==time_range[0][1]:
                
                time_axis = np.array([time_range[0][0]])
                update_list=np.append(update_list,np.ones(int(np.rint((time_resolution+time_range[0][1]-time_range[0][0])/time_resolution)))*len(time_range[-1]))

            else :
                print("time range: ", time_range[0][0]-time_range[0][1])
                print("time resolution before: ", time_resolution)
                max_resolution = 2
                for event in time_range[-1]: 
                    if isinstance(event.behavior, Ramp) and isinstance(event.channel, Analog_Channel): 
                        if max_resolution > event.behavior.resolution:
                            max_resolution = event.behavior.resolution
                print("Max resolution: ", max_resolution)
                time_resolution = max_resolution
                print("time resolution before: ", time_resolution)

                time_axis = np.arange(time_range[0][0], time_resolution+time_range[0][1], time_resolution)
                
                len_update_list_temp =len(update_list)
                update_list=np.append(update_list,np.ones(int(np.rint((time_resolution+time_range[0][1]-time_range[0][0])/time_resolution)))*len(time_range[-1]))

                processdelay_times.append(len_update_list_temp+1)
                processdelay_times.append(len(update_list))
                processdelay_value_list.append(int(time_resolution/adwin_driver.proceessdelay_unit))

            #print("len of time axis: ", len(time_axis))

            temp_channel_number_list =[] 
            temp_channel_value_list =[]
            
            for event in time_range[-1]: 

                

                temp_channel_number_list.append(np.ones_like(time_axis)* encode_channel(type=0 if isinstance(event.channel,Analog_Channel)else 1,channel=event.channel.channel_number,card=event.channel.card_number))
                
                if isinstance(event.behavior, Jump):
                
                    if isinstance(event.channel, Digital_Channel):
                        temp_channel_value_list.append(event.behavior.target_value)
                    else :                         
                        temp_channel_value_list.append(event.channel.discretize(event.channel.default_voltage_func(event.behavior.target_value)))
                
                elif isinstance(event.behavior, Ramp) and isinstance(event.channel, Analog_Channel): 
                    temp_channel_value_list.append(event.channel.discretize(event.channel.default_voltage_func(event.behavior.func(time_axis - event.start_time))))  
                    

            
            stacked_temp_channel_number_list = np.vstack( list(temp_channel_number_list))
            stacked_temp_channel_value_list = np.vstack( list(temp_channel_value_list))

            
            stacked_temp_channel_number_list = stacked_temp_channel_number_list.T
            stacked_temp_channel_value_list = stacked_temp_channel_value_list.T
            
            
            stacked_temp_channel_number_list=stacked_temp_channel_number_list.flatten()
            stacked_temp_channel_value_list=stacked_temp_channel_value_list.flatten()
            

            channel_number=np.append(channel_number,stacked_temp_channel_number_list)
            channel_value=np.append(channel_value,stacked_temp_channel_value_list)

        else: 
            time_to_be_delayed = time_range[0][1]-time_range[0][0]
            

            #print("Time to be delayed: ", time_to_be_delayed)
            #print("Max cycles Time: ", adwin_driver.MAX_CYCLES_time)
            number_of_delays = np.floor(time_to_be_delayed / adwin_driver.MAX_CYCLES_time)
            #print("Number of delays: ", number_of_delays)
            #print(time_to_be_delayed % adwin_driver.MAX_CYCLES_time)
            reminder_of_delays =np.array(( time_to_be_delayed % adwin_driver.MAX_CYCLES_time))
            #print("Reminder of delays: ", reminder_of_delays)

            if reminder_of_delays:
                array_of_delays= np.append(-1* np.ones(int(number_of_delays))*adwin_driver.MAX_CYCLES,-1*int(reminder_of_delays/(adwin_driver.proceessdelay_unit)))
            else:
                array_of_delays=-1* np.ones(int(number_of_delays))*adwin_driver.MAX_CYCLES
            # print("array_of_delays: ", array_of_delays)
            
            update_list=np.append(update_list,array_of_delays)

    
    total_time = time.time() - start_time2 
    

    # print("sequence duration: ", sequence.sequence_dauation())
    # print(f"Total time taken: {total_time}")        
    # processdelay_times.append(0)
    # processdelay_times.append(0)
    # processdelay_value_list.append(adwin_driver.processdelay)

    return update_list, channel_number, channel_value,processdelay_times,processdelay_value_list


class ADwin_Driver:
    def __init__(self, process_file="transfer_seq_data.TC1",PROCESSORTYPE = "12", absolute_path=0,processdelay=1000):
        self.PROCESSORTYPE = PROCESSORTYPE
        if self.PROCESSORTYPE == "12":
            self.proceessdelay_unit = 10**(-9)# For T12 processor, the time unit of the Processdelay (cycle time) is 1 ns for both high priority and low priority processes
        # if the processdelay is set to 1000, the cycle time is 1 us
        elif self.PROCESSORTYPE == "12.1":
            self.proceessdelay_unit = (1.5)*10**(-9)
        elif self.PROCESSORTYPE == "11":
            self.proceessdelay_unit = (3+1/3)*10**(-9)
        elif self.PROCESSORTYPE == "10":
            self.proceessdelay_unit = 25*10**(-9)
        elif self.PROCESSORTYPE == "9":
            self.proceessdelay_unit = 25*10**(-9)

        self.DEVICENUMBER = 0x1
        self.RAISE_EXCEPTIONS = 1 

        self.queue = []
        self.current_index = 0
        
        
        # the philosophy of changing the process delay is to be able to allow the user to change the cycle time of the ADwin system if so many channels are being changed at the same time
        self.processdelay = processdelay
        
        self.MAX_CYCLES = int(2**31-1) # if we want to follow the old logic of putting everything in the units of process delay
        self.MAX_CYCLES_time = self.proceessdelay_unit*2**31 # The maximum time of a delay in seconds that can be achieved with the current process delay
        
        
        
        self.adw = ADwin.ADwin(self.DEVICENUMBER, self.RAISE_EXCEPTIONS)
        print("Booting ADwin-system... ")
        
        BTL = self.adw.ADwindir + "adwin" + self.PROCESSORTYPE + ".btl"
        self.adw.Boot(BTL)

        print("ADwin booted\n")
        print("Loading default process... ")
        if not absolute_path:
            self.adw.Load_Process(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), process_file)
            )
            print("Default process loaded\n")
        else:
            self.adw.Load_Process(process_file)
            print("Process loaded\n")

    def add_to_queue(self, sequence: Sequence):
        update_list, channel_number, channel_value,processdelay_times,processdelay_value_list = calculate_sequence_data_eff(sequence,self)
        self.queue.append({
            "update_list": update_list,

            "channel_number": channel_number,
            "channel_value": channel_value,
            "processdelay_times": processdelay_times,
            "processdelay_value_list": processdelay_value_list
        })

    def load_ADwin_Data(self, index,repeat=1):
        
        
        
        update_list= list(self.queue[index]["update_list"].astype('int') )

        channel_number= list(self.queue[index]["channel_number"].astype('int') )
        channel_value= list(self.queue[index]["channel_value"].astype('int'))

        processdelay_times= self.queue[index]["processdelay_times"]
        processdelay_value_list= self.queue[index]["processdelay_value_list"]


        start_time2 = time.time()
        self.adw.Set_Par(Index=1, Value=len(update_list))
        self.adw.Set_Par(Index=2, Value=int(self.processdelay))
        self.adw.Set_Par(Index=3, Value=int(max(update_list)))
        self.adw.SetData_Long(Data=update_list, DataNo=1, Startindex=1, Count=len(update_list))
        self.adw.SetData_Long(Data=channel_number, DataNo=2, Startindex=1, Count=len(channel_number))
        self.adw.SetData_Long(Data=channel_value, DataNo=3, Startindex=1, Count=len(channel_value))
        
        self.adw.SetData_Long(Data=processdelay_times, DataNo=9, Startindex=1, Count=len(processdelay_times))
        self.adw.SetData_Long(Data=processdelay_value_list, DataNo=10, Startindex=1, Count=len(processdelay_value_list))


        total_time = time.time() - start_time2
        print(f"Total time taken to load data: {total_time}")

    def start_process(self, process_number):
        self.adw.Start_Process(process_number)

    def initiate_experiment(self, process_number=1, index=0,repeat=1):
        self.load_ADwin_Data(index,repeat=repeat)
        self.start_process(process_number)
    
    def repeat_process(self, process_number=1, repeat=1,poll_interval=1):
        for i in range(repeat):
            self.start_process(process_number)
            self.wait_for_process_to_complete(poll_interval)
            print(f"Experiment {i + 1} Completed.")
            

    def is_process_running(self) -> bool:
        status = self.adw.Process_Status(1)
        # 1: Process is running.
        # 0: Process is not running.
        # <0: Process is being stopped.
        return status > 0
    
    def wait_for_process_to_complete(self, poll_interval=1):
        while self.is_process_running():
            print("Waiting for the current process to complete...")
            time.sleep(poll_interval)

    def initiate_all_experiments(self, process_number=1,repeat=1):
        for i in range(len(self.queue)):
            print(f"Initiating experiment {i + 1}/{len(self.queue)}")
            self.initiate_experiment(process_number, index=i,repeat=repeat)
            self.wait_for_process_to_complete()
            print(f"Experiment {i + 1} Completed.")
        #clear the queue after all the experiments have been completed
        self.queue = []
    
    
    def change_process(self, process_file):
        self.adw.Clear_Process(1)
        self.adw.Load_Process(process_file)
        print("Process loaded\n")

import matplotlib.pyplot as plt


def test_camera_trigger_seq(r=0):
    main_seq = Sequence("Camera Trigger")
    main_seq.add_analog_channel("Camera Trigger", 2, 2)
    t = 0
    main_seq.add_event("Camera Trigger", Jump(0), start_time=t)
    t = 1
    main_seq.add_event("Camera Trigger", Jump(3.3), start_time=t)
    t = 2
    main_seq.add_event("Camera Trigger", Jump(0), start_time=t)

    adwin_driver = ADwin_Driver(process_file="transfer_seq_data.TC1",processdelay=1000)
    adwin_driver.add_to_queue(main_seq)
    adwin_driver.initiate_all_experiments(process_number=1,repeat=1)
    if r :
        adwin_driver.repeat_process(process_number=1, repeat=r,poll_interval=0.1)

    

#if __name__ == "__main__":
    # test_camera_trigger_seq(r= 20)
    


if __name__ == "__main__":
    main_seq=Sequence("Testing")
    main_seq.add_analog_channel("four",4,1)
    t=main_seq.root_time_instance
    main_seq.add_event("four",Jump(3),t)

    

    # main_seq = Sequence("MOT_loading")

    # main_seq.add_analog_channel("MOT Coils", 2,1)
    # main_seq.add_analog_channel("Camera Trigger", 2, 2)
    # main_seq.add_analog_channel("Trap TTL", 2, 3)
    # main_seq.add_analog_channel("Trap FM", 2, 4)
    # main_seq.add_analog_channel("Trap AM", 2, 5)
    # main_seq.add_analog_channel("Repump TTL", 2, 6)
    # main_seq.add_analog_channel("Repump FM", 2, 7)
    # main_seq.add_analog_channel("Repump AM", 2, 8)
    # main_seq.add_analog_channel("D1 AOM FM", 3,1)
    # main_seq.add_analog_channel("D1 AOM AM", 3,2)
    # main_seq.add_analog_channel("D1 EOM FM", 3,3)
    # main_seq.add_analog_channel("D1 EOM AM", 3,4)
    # main_seq.add_analog_channel("Absorption imaging FM", 3,5)
    # main_seq.add_analog_channel("Absorption imaging TTL", 3,6)
    # main_seq.add_analog_channel("test7", 3,7)
    # main_seq.add_analog_channel("test8", 3,8)
    # t= 1

    # main_seq.add_event("MOT Coils", Jump(0), start_time=t) # 0 is the ON value for the MOT Coils and 3.3 is the OFF value
    # main_seq.add_event("Camera Trigger", Jump(0), start_time=t) # 0 is the ON value for the MOT Coils and 3.3 is the OFF value
    # main_seq.add_event("Trap TTL", Jump(3.3), start_time=t) 
    # main_seq.add_event("Trap FM", Jump(2.8), start_time=t) 
    # main_seq.add_event("Trap AM", Jump(1.5), start_time=t) 
    # main_seq.add_event("Repump TTL", Jump(3.3), start_time=t) 
    # main_seq.add_event("Repump FM", Jump(1.6), start_time=t) 
    # main_seq.add_event("Repump AM", Jump(0.5), start_time=t) 
    # main_seq.add_event("D1 AOM FM", Jump(0), start_time=t) 
    # main_seq.add_event("D1 AOM AM", Jump(10), start_time=t) 
    # main_seq.add_event("D1 EOM FM", Jump(0), start_time=t) 
    # main_seq.add_event("D1 EOM AM", Jump(0), start_time=t) 
    # main_seq.add_event("Absorption imaging FM", Jump(7.5), start_time=t) 
    # main_seq.add_event("Absorption imaging TTL", Jump(3.3), start_time=t) 
    # main_seq.add_event("test7", Jump(0), start_time=t) 
    # main_seq.add_event("test8", Jump(0), start_time=t) 


    






    # sweep_D1 = main_seq.add_event("D1 AOM FM", Jump(0), start_time=t) # 0 is the ON value for the MOT Coils and 3.3 is the OFF value
    # main_seq.add_event("D1 AOM FM", Jump(0), start_time=t) # 0 is the ON value for the MOT Coils and 3.3 is the OFF value
    # main_seq.add_event("D1 AOM AM", Jump(0), start_time=t) # 0 is the ON value for the MOT Coils and 3.3 is the OFF value
    # main_seq.add_event("test3", Jump(0), start_time=t) # 0 is the ON value for the MOT Coils and 3.3 is the OFF value
    # main_seq.add_event("test4", Jump(0), start_time=t) # 0 is the ON value for the MOT Coils and 3.3 is the OFF value
    # main_seq.add_event("test5", Jump(0), start_time=t) # 0 is the ON value for the MOT Coils and 3.3 is the OFF value
    # main_seq.add_event("test6", Jump(0), start_time=t) # 0 is the ON value for the MOT Coils and 3.3 is the OFF value
    # main_seq.add_event("test7", Jump(0), start_time=t) # 0 is the ON value for the MOT Coils and 3.3 is the OFF value
    # main_seq.add_event("test8", Jump(0), start_time=t) # 0 is the ON value for the MOT Coils and 3.3 is the OFF value



#     # Defualt values
#     t = 2
#     # main_seq.add_event("MOT Coils", Jump(0), start_time=t) # 0 is the ON value for the MOT Coils and 3.3 is the OFF value
#     #main_seq.add_event("Camera Trigger", Jump(0), start_time=t) # 
#     # main_seq.add_event("Trap TTL", Jump(3.3), start_time=t)
#     # Trap_FM_event = main_seq.add_event("Trap FM", Jump(2.5), start_time=t)
#     # main_seq.add_event("Trap AM", Jump(1.25), start_time=t)
#     # main_seq.add_event("Repump TTL", Jump(3.3), start_time=t)
#     # main_seq.add_event("Repump FM", Jump(2.5), start_time=t)
#     # main_seq.add_event("Repump AM", Jump(0.5), start_time=t)
#     Absorption_FM = main_seq.add_event("Absorption FM", Jump(8.75), start_time=t)
#     main_seq.add_event("Absorption AM", Jump(3.3), start_time=t) # 3.3 is the ON value for the AM and 0 is the OFF value

    
#     # main_seq.to_json("MOT_loading.json")
    adwin_driver = ADwin_Driver()
    adwin_driver.add_to_queue(main_seq)

    
    
    

    adwin_driver.initiate_all_experiments(process_number=1,repeat=1)
#     # adwin_driver.repeat_process(process_number=1, repeat=1000,poll_interval=0.1)


    



