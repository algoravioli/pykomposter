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
    seed_array = range(12)
    seed_array = np.random.shuffle(seed_array)
    return seed_array


#main function
def Ausschwingklang(direction,length_in_beats,instrument_array,type=1,rhythm_array=None,seed_array=None):
    # type1 = chord with extended "resonance" decaying in some direction
    if rhythm_array == None:
        rhythm_array = genRhythmArray(length_in_beats)
    
    if seed_array == None:
        seed_array = genSeedArray()