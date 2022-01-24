import sys
import time
import fractions
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
        number_events_in_beat = np.random.choice([1,2,4,8,3,5,6,7])
        event_array = np.append(event_array,number_events_in_beat)
    print(event_array)
    # with number of events, construct acutally 
    rhythm_array = []
    for i in tqdm(range(len(event_array))):
        current_event_value = int(event_array[i])
        non_float_ans = [3,5,6,7]
        reg_ans = [1,2,4,8,16]
        if current_event_value>0:
            for j in range(int(current_event_value)):
                if current_event_value in non_float_ans:
                    rhythm_value = fractions.Fraction(1,current_event_value)
                if current_event_value in reg_ans:
                    rhythm_value = 1/current_event_value
                rhythm_array.append(rhythm_value)
        else: 
            rhythm_array.append(1)


    return rhythm_array    

def genSeedArray(units=12):
    seed_array = np.arange(units)
    np.random.shuffle(seed_array)
    return seed_array


#main function
def Ausschwingklang(length_in_beats,instrument_dict,instrument_limits_dict,type=1,rhythm_array=None,seed_array=None):
    # type1 = chord with extended "resonance" decaying in some direction
    # if rhythm_array == None:
    #     rhythm_array = genRhythmArray(length_in_beats)
    
    if seed_array == None:
        seed_array = genSeedArray()
    
    leap_array = findLeaps(seed_array)
    print('this is leap array')
    print(leap_array)
    print(len(leap_array))

    # variable declaration
    generated_stream = music21.stream.Stream()
    num_of_instruments = len(instrument_dict)
    instrument_list = list(instrument_dict.values())

    instrument_limit_list = list(instrument_limits_dict.values())

    # music21 parts
    # for i in range(num_of_instruments):
    #     music21.stream.Part(id=instrument_list[i])
        
    # input notes here
    if type == 1:
        for i in range(num_of_instruments):
            current_part = music21.stream.Part(id=instrument_list[i])
            # change later!!!
            rhythm_array = genRhythmArray(length_in_beats,6)
            print(rhythm_array)
            np.roll(leap_array, i)
            current_highest_note = int(instrument_limit_list[i])
            if current_highest_note != -1:
                random_start_value = np.random.randint(0,10)
                start_note = current_highest_note - (10+random_start_value)
                print("current highest note")
                print(current_highest_note)
                print ("start note")
                print(start_note)
                for j in range(len(rhythm_array)):
                    if j == 0:
                            inputNote(current_part,start_note,rhythm_array[j])
                    if j > 0:
                        if j > len(leap_array)-1:
                            leap_array = leap_array + np.random.randint(0,5)
                            temp_leap_array = []
                            for l in range(len(leap_array)):
                                curr_leap = leap_array[l]
                                curr_leap = curr_leap%12
                                temp_leap_array = np.append(temp_leap_array,curr_leap)
                            leap_array = temp_leap_array
                            j = j%len(leap_array)
                        print("this is leap array [j]")
                        print(leap_array[j])
                        
                        current_note = int(start_note + int(leap_array[j]))
                        print("current_note no offset")
                        print(current_note)
                        if current_note > current_highest_note:
                            current_note = current_note - 12
                        print("current_note w/ offset")
                        print(current_note)
                        inputNote(current_part,current_note,rhythm_array[j])
                        start_note = current_note
            else:
                random_start_value = np.random.randint(0,10)
                start_note = 40+random_start_value
                chord_array = []
                chord_array.append(int(start_note))
                for k in range(np.random.randint(0,len(leap_array)-4)):
                    current_note = start_note + leap_array[k]
                    print("this is the piano note")
                    print(current_note)
                    if current_note-start_note > 10:
                        current_note = current_note - 12
                    chord_array.append(int(current_note))
                    start_note = current_note
                
                d = music21.duration.Duration(8.0)
                big_chord = music21.chord.Chord(chord_array, duration=d)
                current_part.append(big_chord)
                print(big_chord)
            leap_array = findLeaps(seed_array)
            generated_stream.insert(0, current_part) 
    generated_stream.makeMeasures()
    # m1 = music21.stream.Measure(number=1)
    # m1.timeSignature = music21.meter.TimeSignature('1/4')
    # m1.show('text')         
    return generated_stream

gen = Ausschwingklang(7,instrument_dict,instrument_highest_note_dict)
print(gen)
gen.show()

# print(fractions.Fraction(1,16))