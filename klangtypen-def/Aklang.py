import sys
import time
import fractions
import matplotlib.pyplot as plt
import music21
import numpy as np
from numpy.random.mtrand import seed
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
    # "instrument5": "Piano"
}

instrument_highest_note_dict = {
    "instrument1": 86,
    "instrument2": 80,
    "instrument3": 75,
    "instrument4": 70,
    # "instrument5": -1
}

def timeSignatureLookup(number_of_beats):
    a = number_of_beats
    if a == 1:
        out = music21.meter.TimeSignature('1/4')
    if a == 2:
        out = music21.meter.TimeSignature('2/4')
    if a == 3:
        out = music21.meter.TimeSignature('3/4')
    if a == 4:
        out = music21.meter.TimeSignature('4/4')
    if a == 5:
        out = music21.meter.TimeSignature('5/4')
    if a == 6:
        out = music21.meter.TimeSignature('6/4')
    if a == 7:
        out = music21.meter.TimeSignature('7/4')
    if a == 8:
        out = music21.meter.TimeSignature('8/4')
    if a == 9:
        out = music21.meter.TimeSignature('9/4')
    if a == 10:
        out = music21.meter.TimeSignature('10/4')
    if a == 11:
        out = music21.meter.TimeSignature('11/4')
    if a == 12:
        out = music21.meter.TimeSignature('12/4')
    if a == 13:
        out = music21.meter.TimeSignature('13/4')
    if a == 14:
        out = music21.meter.TimeSignature('14/4')
    if a == 15:
        out = music21.meter.TimeSignature('15/4')
    if a == 16:
        out = music21.meter.TimeSignature('16/4')
    return out

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

def genSeedArray(units=12):
    seed_array = np.arange(units)
    np.random.shuffle(seed_array)
    return seed_array

def createExtendedLeapArray(leap_array,mult):
    array_length = mult*7
    extended_leap_array = []
    for i in range(mult):
        for j in range(len(leap_array)):
            extended_leap_array.append(leap_array[j])
        leap_array = [((foo + np.random.randint(1,5))%12) for foo in leap_array]
    return extended_leap_array

def measureLevel(curr_measure_obj,measure_length,extended_leap_array,instrument_limit,offset):
    for i in range(measure_length):
        # num_of_beats = measure_length
        # for loop describes how many events per beat\
        # PER BEAT LEVEL
        num_of_events = np.random.choice([0,1,2,3,4,5,6,7])
        rhythm_array = []

        if num_of_events in [3,5,6,7]:
            for j in range(num_of_events):
                rhythm_array.append(fractions.Fraction(1,num_of_events))
        elif num_of_events == 0:
            inputRest(curr_measure_obj,1)
        else:
            for j in range(num_of_events):
                rhythm_array.append(1/num_of_events)
        
        prev_note = instrument_limit-18 + (np.random.randint(0,11) * np.random.choice([1,-1]))        
        for k in range(len(rhythm_array)): #rhythm array is for PER BEAT
            curr_note = prev_note + extended_leap_array[k+offset]
            if curr_note > instrument_limit:
                curr_note = curr_note - 12
            inputNote(curr_measure_obj,curr_note,rhythm_array[k])

    return curr_measure_obj

def partLevel(array_of_beats,current_part,extended_leap_array,instrument_list,instrument_limit_list):
    #extend leap array
    offset = 0
    for i in range(len(array_of_beats)):
        curr_measure_length = int(array_of_beats[i])
        curr_measure_obj = music21.stream.Measure(number=i)
        curr_measure_time_signature = timeSignatureLookup(curr_measure_length)
        curr_measure_obj.append(curr_measure_time_signature)
        instrument_limit = int(instrument_limit_list[i])
        
        curr_measure = measureLevel(curr_measure_obj,curr_measure_length,extended_leap_array,instrument_limit,offset)
        current_part.append(curr_measure)
        offset = offset+curr_measure_length
    return current_part

def scoreLevel(stream,num_of_instruments,instrument_list,instrument_limit_list,array_of_beats,leap_array,seed_array):
    for i in range(num_of_instruments):
            current_part = music21.stream.Part(id=instrument_list[i])
            mult = np.sum(array_of_beats)
            extended_leap_array = createExtendedLeapArray(leap_array,mult)
            output_part = partLevel(array_of_beats,current_part,extended_leap_array,instrument_list,instrument_limit_list)
            current_part = output_part
            leap_array = findLeaps(seed_array)
            stream.insert(0,output_part)
    return stream

def Aklang(array_of_beats,instrument_dict,instrument_limits_dict,type=1,rhythm_array=None,seed_array=None):
    # type1 = chord with extended "resonance" decaying in upwards direction
    
    generated_score = music21.stream.Score(id="Score")
    if seed_array == None:
        seed_array = genSeedArray()
    leap_array = findLeaps(seed_array)
    generated_stream = music21.stream.Stream()
    num_of_instruments = len(instrument_dict)
    instrument_list = list(instrument_dict.values())
    instrument_limit_list = list(instrument_limits_dict.values())
    output = scoreLevel(generated_score,num_of_instruments,instrument_list,instrument_limit_list,array_of_beats,leap_array,seed_array)
    return output


a = Aklang([2,4,5],instrument_dict,instrument_highest_note_dict)
a.show()
