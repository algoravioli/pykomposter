import sys
import music21
import pandas as pd
import numpy as np
import time

#variables
generated_stream = music21.stream.Stream()

#input array as ["1","2",...]
input_array = sys.argv[1]
input_array = input_array.split(",")
output_array = []
for i in range(len(input_array)):
    curr_str = input_array[i]
    output_array.append(int(curr_str))

input_array = output_array

def inputNote(stream,input_note,input_rhy):
    curr_note = music21.note.Note(music21.note.Note(input_note).nameWithOctave)
    curr_note.duration.quarterLength = input_rhy
    stream.append(curr_note)

def findIntervals(array):
    a
    return out_array

inputNote(generated_stream,61,0.25)
inputNote(generated_stream,62,0.5)
inputNote(generated_stream,63,0.25)







generated_stream.show()



