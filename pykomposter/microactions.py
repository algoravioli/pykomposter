import fractions
import sys
import time
import math
import music21
import numpy as np
import pandas as pd

# import tensorflow
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


def durationTypeFinder(num):
    if num == 5:
        return "16th"
    if num == 6:
        return "16th"
    if num == 7:
        return "16th"
    if num == 9:
        return "32th"
    if num == 10:
        return "32th"
    if num == 11:
        return "32th"
    if num == 12:
        return "32th"
    if num == 13:
        return "32th"
    if num == 14:
        return "32th"
    if num == 15:
        return "32th"


def Log2(x):
    y = math.log10(x) / math.log10(2)
    return y


def isPowerOfTwo(n):
    m = math.ceil(Log2(n)) == math.floor(Log2(n))
    return m


def inputNote(measure, input_note, input_rhythm):
    # print(input_rhythm)

    curr_note = music21.note.Note(music21.note.Note(input_note).nameWithOctave)
    if input_rhythm == 0.75:
        curr_note.duration.quarterLength = input_rhythm
    elif isPowerOfTwo(1 / input_rhythm) or (1 / input_rhythm) == 3:
        curr_note.duration.quarterLength = input_rhythm
    else:
        curr_tuplet = music21.duration.Tuplet((1 / input_rhythm), 4)
        curr_tuplet.setDurationType(durationTypeFinder((1 / input_rhythm)))
        print(curr_duration)
        curr_duration = music21.duration.Duration(
            durationTypeFinder((1 / input_rhythm))
        )
        curr_duration.appendTuplet(curr_tuplet)
        curr_note.duration = curr_duration

        ##TUPLET CODE
        # t = duration.Tuplet(5, 4)
        # t.setDurationType('16th')
        # d = duration.Duration('16th')
        # d.appendTuplet(t)
        # n = note.Note('E-4')
        # n.duration = d
    measure.append(curr_note)


def inputRest(measure, rest_duration):
    curr_note = music21.note.Rest()
    curr_note.duration.quarterLength = rest_duration
    measure.append(curr_note)


def createMeasures(content, beat_dict, beats_information, total_beats_information):
    # create dictionary to contain all measures
    measures_dictionary = dict()
    beat_number = 0
    # print(beat_dict)

    for i in range(len(beats_information)):
        current_measure = music21.stream.Measure(number=i + 1)
        current_time_signature = timeSignatureLookup(beats_information[i])
        current_measure.append(current_time_signature)
        current_measure_string = f"measure{i+1}"

        for j in range(beats_information[i]):
            current_beat = beat_dict[f"{beat_number}"]
            if current_beat == [0]:
                inputRest(current_measure, 1)
            else:
                for k in range(len(current_beat)):
                    if len(content) > 0:
                        random_rest_mechanism = (
                            2  # np.random.choice([1, 2], p=[0.05, 0.95])
                        )
                        if random_rest_mechanism == 2:
                            if content[0] == 0:
                                inputRest(current_measure, current_beat[k])
                            else:
                                inputNote(current_measure, content[0], current_beat[k])
                        if random_rest_mechanism == 1:
                            inputRest(current_measure, current_beat[k])
                        content = np.delete(content, 0)
            beat_number += 1
        measures_dictionary[current_measure_string] = current_measure

    return measures_dictionary


def createPart(measures_dictionary, part_name, instrument, metronome_mark):
    current_part = music21.stream.Part(id=f"{part_name}")
    if part_name == "part0":
        current_part.insert(0, metronome_mark)
    current_part.insert(instrument)
    for i in range(len(measures_dictionary)):
        current_measure = measures_dictionary[f"measure{i+1}"]
        current_part.append(current_measure)
    return current_part


def createScore(part_dictionary):
    score = music21.stream.Stream(id="Score")

    for i in range(len(part_dictionary)):
        score.insert(0, part_dictionary[f"part{i}"])

    return score
