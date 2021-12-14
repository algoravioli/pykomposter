import sys
import music21
import pandas as pd
import numpy as np
import time


def inputNote(stream,input_note,input_rhy):
    curr_note = music21.note.Note(music21.note.Note(input_note).nameWithOctave)
    curr_note.duration.quarterLength = input_rhy
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

def genIA(initial_note_array, rhythm_array, direction_array, leap_array, generated_stream, num_gens=5):
    generated_output = initial_note_array
    for i in range(num_gens):
        last_value = generated_output[-1]
        for i in range(len(leap_array)):
            interval = leap_array[i]
            direction = direction_array[i]
            step = interval*direction
            next_value = last_value + step
            generated_output = np.append(generated_output,next_value)
    
    print(len(generated_output))
    length_of_entire_seq = len(generated_output)/len(rhythm_array)
    
    for i in range(np.floor(length_of_entire_seq).astype(int)):
        rhythm_array = np.concatenate((rhythm_array,rhythm_array),axis=None)
    remainder = len(generated_output) - len(rhythm_array)
    
    for i in range(remainder):
        rhythm_array = np.append(rhythm_array,rhythm_array[i])

    octave_corrected = []
    for i in range(len(generated_output)):
        current_note = generated_output[i]
        #fix required to keep contour
        while current_note>(max(input_array)):
            current_note = current_note - 24
        octave_corrected = np.append(octave_corrected, current_note)
    generated_output = octave_corrected    

    for i in range(len(generated_output)):
        inputNote(generated_stream,60+generated_output[i],rhythm_array[i])
    
    return generated_stream

# ingredients:
# initial note array
# note array with respect to middle C
# input array as 1,2,3,... ## 23,15,1,4,11,20
input_array = sys.argv[1]
input_array = input_array.split(",")
output_array = []
for i in range(len(input_array)):
    curr_str = input_array[i]
    output_array.append(int(curr_str))
input_array = output_array

# rhythm array as 1,0.5,0.25,... ## 1,0.5,1,0.25,0.25,1,0.5,1
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
num_gens = 15

genIA(input_array,rhythm_array,direction_array,leap_array,generated_stream,num_gens)

generated_stream.show()


