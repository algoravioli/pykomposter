import sys
import time

import matplotlib.pyplot as plt
import music21
import numpy as np
import pandas as pd
from tqdm import tqdm

## DECAY SOUND

#variables
up = 1
down = -1 

instrument_dict = {
    "instrument1": "Soprano Saxophone",
    "instrument2": "Alto Saxophone",
    "instrument3": "Tenor Saxophone",
    "instrument4": "Baritone Saxophone",
    "instrument5": "Piano"
}

instrument_highest_note_dict = {
    "instrument1": 86,
    "instrument2": 80,
    "instrument3": 75,
    "instrument4": 70,
    "instrument5": -1
}

def inputNote(stream,input_note,input_rhy):
    curr_note = music21.note.Note(music21.note.Note(input_note).nameWithOctave)
    curr_note.duration.quarterLength = input_rhy
    stream.append(curr_note)

def inputRest(stream,rest_duration):
    curr_note = music21.note.Rest()
    curr_note.duration.quarterLength = rest_duration
    stream.append(curr_note)

def findLeaps(array):
    leap_array = []
    for i in range(len(array)):
        if i>0:
            current_leap = abs(array[i] - array[i-1])
            leap_array = np.append(leap_array,current_leap)
    return leap_array

def genRhythmArray(length,max_events_per_beat):
    if max_events_per_beat<5:
        print("Argument <<max_events_per_beat>> should be an integer more than 5.")
        return

    event_array = []

    for i in tqdm(range(length)):
        number_events_in_beat = np.random.randint(max_events_per_beat-4,size=1)
        event_array = np.append(event_array,number_events_in_beat+4)

    # with number of events, construct acutally 
    rhythm_array = []
    for i in tqdm(range(len(event_array))):
        current_event_value = event_array[i]
        if current_event_value>0:
            for j in range(int(current_event_value)):
                rhythm_value = 1/current_event_value
                rhythm_array = np.append(rhythm_array,rhythm_value)
        else: 
            rhythm_array = np.append(rhythm_array,1)


    return rhythm_array    

def genSeedArray(units=12):
    seed_array = np.arange(units)
    np.random.shuffle(seed_array)
    return seed_array


#main function
def Ausschwingklang(direction,length_in_beats,instrument_dict,instrument_limits_dict,type=1,rhythm_array=None,seed_array=None):
    # type1 = chord with extended "resonance" decaying in some direction
    if rhythm_array == None:
        rhythm_array = genRhythmArray(length_in_beats)
    
    if seed_array == None:
        seed_array = genSeedArray()
    
    leap_array = findLeaps(seed_array)

    # variable declaration
    generated_stream = music21.stream.Stream()
    num_of_instruments = len(instrument_dict)
    instrument_list = list(instrument_dict.values())

    instrument_limit_list = list(instrument_limits_dict.values())

    # music21 parts
    for i in range(num_of_instruments):
        music21.stream.Part(id=instrument_list[i])
        
    # input notes here
    if type == 1:
        for i in range(num_of_instruments):
            current_part = music21.stream.Part(id=instrument_list[i])
            # change later!!!
            rhythm_array = genRhythmArray(length_in_beats)
            np.roll(leap_array, i)
            current_highest_note = instrument_limit_list[i]
            random_start_value = np.random.randint(0,10)
            start_note = current_highest_note - (10+random_start_value)

            for j in range(rhythm_array):
                
                return
                





    # return



# np.roll(x, -2)

a = list(instrument_dict.values())

print(np.random.randint(0,10))