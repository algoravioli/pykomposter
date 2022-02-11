import fractions
import sys
import time

import music21
import numpy as np
import pandas as pd
import tensorflow
import tqdm


def timeSignatureLookup(number_of_beats):
    a = number_of_beats
    if a == 1:
        out = music21.meter.TimeSignature("1/4")
    if a == 2:
        out = music21.meter.TimeSignature("2/4")
    if a == 3:
        out = music21.meter.TimeSignature("3/4")
    if a == 4:
        out = music21.meter.TimeSignature("4/4")
    if a == 5:
        out = music21.meter.TimeSignature("5/4")
    if a == 6:
        out = music21.meter.TimeSignature("6/4")
    if a == 7:
        out = music21.meter.TimeSignature("7/4")
    if a == 8:
        out = music21.meter.TimeSignature("8/4")
    if a == 9:
        out = music21.meter.TimeSignature("9/4")
    if a == 10:
        out = music21.meter.TimeSignature("10/4")
    if a == 11:
        out = music21.meter.TimeSignature("11/4")
    if a == 12:
        out = music21.meter.TimeSignature("12/4")
    if a == 13:
        out = music21.meter.TimeSignature("13/4")
    if a == 14:
        out = music21.meter.TimeSignature("14/4")
    if a == 15:
        out = music21.meter.TimeSignature("15/4")
    if a == 16:
        out = music21.meter.TimeSignature("16/4")
    return out


def inputNote(measure, input_note, input_rhythm):
    curr_note = music21.note.Note(music21.note.Note(input_note).nameWithOctave)
    curr_note.duration.quarterLength = input_rhythm
    measure.append(curr_note)


def inputRest(measure, rest_duration):
    curr_note = music21.note.Rest()
    curr_note.duration.quarterLength = rest_duration
    measure.append(curr_note)


def createMeasures(content, beat_dict, rhythm_information):
    # create dictionary to contain all measures
    measures_dictionary = dict()

    for i in range(len(rhythm_information)):
        current_measure = music21.stream.Measure(number=i + 1)
        current_time_signature = timeSignatureLookup(rhythm_information[i])
        current_measure.append(current_time_signature)
        print(rhythm_information[i])

        current_measure_string = f"measure{i+1}"

        for j in range(rhythm_information[i]):
            current_beat = beat_dict[f"{j}"]
            if current_beat == [0]:
                inputRest(current_measure, 1)
            else:
                for k in range(len(current_beat)):
                    if len(content) > 0:
                        random_rest_mechanism = np.random.choice([1, 2], p=[0.05, 0.95])
                        if random_rest_mechanism == 2:
                            inputNote(current_measure, content[0], current_beat[k])
                        if random_rest_mechanism == 1:
                            inputRest(current_measure, current_beat[k])
                        content = np.delete(content, 0)

                    # print(len(content))

        measures_dictionary[current_measure_string] = current_measure

    # print(measures_dictionary)
    return measures_dictionary


def createPart(measures_dictionary, part_name):
    current_part = music21.stream.Part(id=f"{part_name}")
    for i in range(len(measures_dictionary)):
        current_measure = measures_dictionary[f"measure{i+1}"]
        current_part.append(current_measure)
    return current_part


def createScore(part_dictionary):
    for i in range(len(part_dictionary)):
        score = music21.stream.Score(id="Score")
        score.insert(0, part_dictionary[f"part{i}"])

    return score
