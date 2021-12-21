import sys
import time

import matplotlib.pyplot as plt
import music21
import numpy as np
import pandas as pd
from tqdm import tqdm

# from prompt_toolkit import Application

# app = Application(full_screen=True)
# app.run()

def inputNote(stream,input_note,input_rhy):
    curr_note = music21.note.Note(music21.note.Note(input_note).nameWithOctave)
    curr_note.duration.quarterLength = input_rhy
    stream.append(curr_note)

def inputRest(stream,rest_duration):
    curr_note = music21.note.Rest()
    curr_note.duration.quarterLength = rest_duration
    stream.append(curr_note)

def findIntervals(array):
    pure_interval_array = []
    for i in range(len(array)):
        if array[i]>11:
            new_int = array[i] - 12
            pure_interval_array = np.append(pure_interval_array,new_int)
        else:
            pure_interval_array = np.append(pure_interval_array,array[i])
    return pure_interval_array

def findLeaps(array):
    leap_array = []
    for i in range(len(array)):
        if i>0:
            current_leap = abs(array[i] - array[i-1])
            leap_array = np.append(leap_array,current_leap)
    return leap_array

def findDirection(array): #expects input_array / sys.argv[1]
    direction_array = []
    for i in range(len(array)):
        if i>0:
            if array[i]<array[i-1]:
                current_direction = -1
            else:
                current_direction = 1
            direction_array = np.append(direction_array, current_direction)
    return direction_array

def genIA(initial_note_array, rhythm_array, direction_array, leap_array, generated_stream, num_gens=5, prob_of_note=0.5):
    generated_output = initial_note_array
    print("Generating Note Sequence:")
    time.sleep(0.3)
    for i in tqdm(range(num_gens)):
        last_value = generated_output[-1]
        for i in range(len(leap_array)):
            interval = leap_array[i]
            direction = direction_array[i]
            step = interval*direction
            next_value = last_value + step
            generated_output = np.append(generated_output,next_value)

    length_of_entire_seq = len(generated_output)/len(rhythm_array)
    
    new_rhythm_array = []
    print("Generating Rhythmic Sequences(1/2):")
    time.sleep(0.3)
    for i in tqdm(range(np.floor(length_of_entire_seq).astype(int))):
        new_rhythm_array = np.concatenate((new_rhythm_array,rhythm_array),axis=None)
        # print(len(new_rhythm_array))
    remainder = len(generated_output) - len(rhythm_array)
    print("Generating Rhythmic Sequences(2/2):")
    time.sleep(0.3)
    if remainder!=0:
        for i in tqdm(range(remainder)):
            new_rhythm_array = np.append(new_rhythm_array,new_rhythm_array[i])

    octave_corrected = []
    print("Correcting Octave Divergence:")
    time.sleep(0.3)
    for i in tqdm(range(len(generated_output))):
        current_note = generated_output[i]
        #fix required to keep contour
        while current_note>(max(initial_note_array)+12):
            current_note = current_note - 24
        while current_note<(min(initial_note_array)):
            current_note = current_note + 24
            # print(current_note)
        octave_corrected = np.append(octave_corrected, current_note)
    generated_output = octave_corrected    
    plt.plot(octave_corrected)

    print("Adding Notes to Score File (XML):")
    time.sleep(0.3)
    for i in tqdm(range(len(generated_output))):
        # print(prob_of_note)
        determine_flag = np.random.choice(2,1,p=[1-prob_of_note,prob_of_note])
        determine_flag.flatten()
        if i<len(initial_note_array):
            inputNote(generated_stream,60+generated_output[i],new_rhythm_array[i])
        else:
            if determine_flag>0:
                inputNote(generated_stream,60+generated_output[i],new_rhythm_array[i])
                prob_of_note = prob_of_note + 0.02
                if prob_of_note > 0.98:
                    prob_of_note = 0.5
            else:
                inputRest(generated_stream,new_rhythm_array[i])
                # if np.random.randint(2, size=1).flatten()>0:
                #     prob_of_note = 0
                prob_of_note = prob_of_note - 0.02
                if prob_of_note < 0.02:
                    prob_of_note = 0.5
            
    
    return generated_stream

# ingredients:
# initial note array
# note array with respect to middle C
# input array as 1,2,3,... ## 23,15,1,4,11,22 ##1st part - 11,13,14,1,2,1,8,11,4,12,11,3,-1,-2,6,9,8
input_array = sys.argv[1]
input_array = input_array.split(",")
output_array = []
for i in range(len(input_array)):
    curr_str = input_array[i]
    output_array.append(int(curr_str))
input_array = output_array

# rhythm array as 1,0.5,0.25,... ## 1,0.5,1,0.25,0.25,1,0.5,1 ##1st part - 0.5,0.5,1,0.5,0.25,0.25,0.75,0.25,0.5,2.5,1,0.5,0.25,0.25,0.75,0.25,1.5
# rhythm array
rhythm_array = sys.argv[2]
rhythm_array = rhythm_array.split(",")
output_rhythms = []
for i in range(len(rhythm_array)):
    curr_str = rhythm_array[i]
    output_rhythms.append(float(curr_str))
rhythm_array = output_rhythms

# direction array
direction_array = findDirection(input_array)

# leap array
leap_array = findLeaps(input_array)

# generated stream
generated_stream = music21.stream.Stream()

# number of generations
num_gens = 55

genIA(input_array,rhythm_array,direction_array,leap_array,generated_stream,num_gens,0.65)

generated_stream.show()
plt.show()


