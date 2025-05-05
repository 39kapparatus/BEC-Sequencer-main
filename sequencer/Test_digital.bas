'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 1
' Initial_Processdelay           = 10000
' Eventsource                    = Timer
' Control_long_Delays_for_Stop   = No
' Priority                       = High
' Version                        = 1
' ADbasic_Version                = 6.3.1
' Optimize                       = Yes
' Optimize_Level                 = 1
' Stacksize                      = 1000
' Info_Last_Save                 = DESKTOP-6MSM8KU  DESKTOP-6MSM8KU\E3
'<Header End>
'<ADbasic Header, Headerversion 001.001>
' Process_Number                 = 1
' Initial_Processdelay           = 1000
' Eventsource                    = Timer
' Control_long_Delays_for_Stop   = No
' Priority                       = High
' Version                        = 1
' ADbasic_Version                = 6.4.0
' Optimize                       = Yes
' Optimize_Level                 = 1
' Stacksize                      = 1000
' Info_Last_Save                 = DESKTOP-IAC6L9U  DESKTOP-IAC6L9U\Mahmoud Yasser
'<Header End>
'
'
' Documentation of the enteir code :


'This file takes the results of the sequences coded in Python and transfers them to the ADwin channels at the set times.
'Once compiled, the ADwin reads the binary file, which is essentially a list of channels, times and voltages/states.

#include adwinpro_all.inc

#define SIZE 50000000 'size of the data arrays



#define MAX_EVENT_COUNT Par_1 'controls when the looping stops; assigned to Par_1 in E3_load_seq.m
#define MAX_UPDATES Par_3 'max number of updates per EVENT run
#define DELAY Par_2 'the base delay

'For mapping analog voltage values to a bit pattern:
#define LIMIT 65535 'at most can have 16 bit value written to DAC


Import Math.lic 


'these arrays are predefined globals in ADbasic, but they must be declared before use
dim Data_1[SIZE] as long 'this array is used to step through Data_2 and Data_3; it indicates how many channels are to be updated per run of the EVENT section
'negative values indicate a wait/hold of the signal value (delay before changing)?
dim Data_2[SIZE] as long 'indicates the channels to be updated
dim Data_3[SIZE] as long 'indicates the values of the channels; in a 1:1 correspondence with Data_2
'Data_4 will be used for reset to zero stuff?
'reset to zero stuff?

dim Data_9[SIZE] as long
dim Data_10[SIZE] as long


'declaration of local variables
dim i as long 'loop index
dim curr_delay as long 'delay to hold a voltage/state until the appropriate time to change
dim event_count as long 'tracks number of runs of the EVENT loop
dim num_updates as long 'tracks running total of number of updates to signal values
dim ch as long 'the channel to be updated
dim state as long 'the updated value

dim process_delay_count1 as long
dim process_delay_count2 as long
dim flag_process_delay as long


dim type as long 
dim card as long 
dim channel as long
dim type_card_channel as long
dim ret_val as long 
'Note that the output voltage range of the DACs for the AOUT8/16 modules is set to ?10V bipolar and can't be
'changed (ADwin Pro II hardware manual, page 96)
'8 output channels, 16 bit resolution, < 3 us settling time

'***************************************************
'convert continuous voltage value in specified range to an n bit value, where LIMIT = 2^n - 1





INIT:
  'Processdelay is a predefined ADbasic variable that correponds to the number of cycles that the EVENT section takes
  'For the Pro II T12, Processdelay is in units of 1 ns e.g. if Processdelay is 1000 clock cycles then that corresponds to an EVENT length 1 us.
  'Note that the ProCPUT12 has a clk rate of 1000 MHz (1 GHz)



  'set the necessary digital channels to be output channels
  'DigProg sets the channels of the module specified in the first param to inputs or outputs
  'Does so in groups of 8 (each bit in second param corresponds to an octet of channels)
  'Note that on power up all channels are at first configured as inputs
  'Bit = 0 --> input
  'Bit = 1 --> output
  P2_DigProg(1,01Fh)
  
  
    
  'configure the channels of each modules for sycnhronous output: this helps to make sure that all updates that are meant to be
  'simultaneous occur at the same time in the EVENT loop, not one after the other
  'for revision E digital cards, just need to set one bit to enable all channels
  P2_Sync_Enable(1, 01b)
  
  'This is for analog channels; 0FFh corresponds to 11111111b and since our analog 
  'cards have 8 channels we sync enable all 8 of them
  P2_SYNC_ENABLE(2,0FFh)
  P2_SYNC_ENABLE(3,0FFh)
  P2_SYNC_ENABLE(4,0FFh)
  P2_SYNC_ENABLE(5,0FFh)

  P2_SYNC_ALL(011111b) 
  

  P2_Dig_Write_Latch(1,17)
  
  
EVENT:
  P2_SYNC_ALL(0111b) 
  ProcessDelay = 1000000
  'P2_Dig_Write_Latch(1,01b)
  
  Par_3= P2_Get_Digout_Long (1)
  'P2_Dig_Write_Latch(1,01b)
  'generic_write(4097 , 1)
  if (Par_3 > 2000000000) then 
    Par_3 =0
  endif 
  
  P2_Dig_Write_Latch(1,Par_3+1)
  
  
  
  'works with SYNC_ENABLE to ensure simultaneous updates are indeed simultaneous
    
  'update Processdelay... should do at beginning of EVENT section!
  'a negative value in Data_1 indicates we should hold values for the absolute value of Data_1[event_count] number of base delays
  'we update Processdelay accordingly
  'If Data_1[event_count] is positive, the value controls the update channel loop

  


FINISH:
  
  'set to default stuff
  '  P2_Digout(d_io, 0, 1) '3D MOT single pass MOGlabs; leave on
  '  P2_Digout(d_io, 1, 0) 'camera trigger
  '  P2_Digout(d_io, 2, 0) 'just testing for now
  'P2_Dig_Write_Latch(d_io, 000b)
  
  'reset to default values; hard coded ch vals just for now for testing/until we know desired "end state" signal behaviour
  'analog_write(33, 0)  '3D mot coils are active low; this leaves them off when we finish
  'analog_write(45, 0)   '3D MOT single pass MOGlabs; keep on
    
  'analog_write(37, 0) 'Crystal Tech 3D MOT single pass; leave on
  'analog_write(40, 0) 'off; note that nominal 3D repump freq is 6.8 V
  'analog_write(39, 0) 'nominal 3D trap freq
  
  'channel tests
  'analog_write(38, 0)   'analog card 1 ch 6
  'analog_write(47, 0)   'analog card 1 ch 7
  
  'P2_SYNC_ALL(0111b) # this is wierd
  
  '***************************************************
 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
sub generic_write(type_card_channel , avalue)
  
  ' Extract x by shifting right by 12 bits
  type =Shift_Right (type_card_channel,12)And  1Fh
  card =Shift_Right(type_card_channel, 7) And  1Fh
  channel = type_card_channel And  7Fh
  P2_Dig_Write_Latch(1,01b)
  ' Extract y by masking the relevant bits and shifting right by 7 bits
  if (type = 0) then 
    
    'P2_Write_DAC(channel,card,avalue)
     
  else 
   
    ret_val = P2_Dig_Read_Latch(card)
    
    if (avalue = 1) then
      ' Use bitwise OR to set the bit
      ret_val = ret_val Or (Shift_Right (channel,1))
    else:
      ' Use bitwise AND with the complement to clear the bit
      ret_val = ret_val And  (Not ((Shift_Right (channel,1))))

    endif 
    Par_4 = ret_val
    

    
  endif 
    
  ' Extract z by masking the last 7 bits
    
endsub





  


