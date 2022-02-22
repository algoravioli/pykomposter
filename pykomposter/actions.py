import fractions
import sys
import time

import music21
import numpy as np
import pandas as pd

# import tensorflow
import tqdm


def Compose(metabehaviour, behaviour_class, op_char):
    # gets time and content information
    content_information = op_char["content"]
    time_information = op_char["time"]
    instruments_information = op_char["instruments"]

    smallest_div_information = time_information["smallest_div"]
    smallest_div_print = music21.duration.Duration()
    smallest_div_print.quarterLength = smallest_div_information
    print(f"The smallest division set is {smallest_div_print.type}")
    # check if smallest_div is empty
    if smallest_div_information == None:
        smallest_div_information = 0.25
        print(
            "Smallest Division: Smallest division was not set. It has been set to 0.25 (1/16th Note)."
        )

    beats_information = time_information["beats"]
    # check if beats is empty
    if beats_information == None:
        print(
            "No beat information was given. The beat information will be generated randomly."
        )
        beats_information = []
        for event in range(np.random.randint(0, 10)):
            beats_information.append(np.random.randint(0, 5))

    rhythm_information = time_information["rhythm"]
    # check if rhythm is empty
    if rhythm_information == None:
        print("Rhythm Dictionary: Rhythm function does not exist.")
        rhythm_information_flag = 0
        total_length = np.sum(beats_information)
        print(
            f"Total Length in Beats: Total duration has been inferred, and thus, it has been set to {total_length}."
        )
        total_beats_information = total_length
    else:
        rhythm_information_flag = 1

    metronome_mark = time_information["tempo"]
    # gets a reference to behaviour_class
    behaviour_ref = behaviour_class()
    # runs the prepare function to prepare the usable content for each behaviour class
    choice_set = behaviour_ref.prepare(content_information)
    # runs the metabehaviour in each class
    metabehaviour = behaviour_ref.withMetabehaviour(metabehaviour)

    if str(metabehaviour.__class__.__name__) == "random":

        dict_of_beat_dicts = dict()
        for i in range(len(instruments_information)):
            beat_dict, total_number_of_events = metabehaviour.eventCalculator(
                smallest_div_information,
                rhythm_information,
                beats_information,
                rhythm_information_flag,
                total_beats_information,
            )
            dict_of_beat_dicts[f"part{i}"] = beat_dict
            dict_of_beat_dicts[f"total_events{i}"] = total_number_of_events

        score = metabehaviour.run(
            choice_set,
            dict_of_beat_dicts,
            content_information,
            beats_information,
            behaviour_ref,
            instruments_information,
            metronome_mark,
            rhythm_information_flag,
            total_beats_information,
            parts=len(instruments_information),
        )

    return score
