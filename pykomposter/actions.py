#%%
import fractions
import sys
import time

import music21
import numpy as np
import pandas as pd
import tensorflow
import tqdm


def Compose(metabehaviour, behaviour_class, op_char):
    # gets time and content information
    content_information = op_char["content"]
    time_information = op_char["time"]

    smallest_div_information = time_information["smallest_div"]
    print(smallest_div_information)
    # check if smallest_div is empty
    if smallest_div_information == None:
        smallest_div_information = 0.25
        print(
            "Smallest Division: Smallest division was not set. It has been set to 0.25 (1/16th Note)."
        )

    rhythm_information = time_information["rhythm"]
    # check if rhythm is empty
    if rhythm_information == []:
        print("Rhythm Dictionary: Rhythm array is empty.")

    beats_information = time_information["beats"]
    # check if beats is empty
    if beats_information == None:
        total_length = np.sum(rhythm_information)
        print(
            f"Beat Dictionary: No total duration was given, and thus, it has been set to {total_length}."
        )
        beats_information = total_length

    # gets a reference to behaviour_class
    behaviour_ref = behaviour_class()
    # runs the prepare function to prepare the usable content for each behaviour class
    choice_set = behaviour_ref.prepare(content_information)
    # runs the metabehaviour in each class
    metabehaviour = behaviour_ref.withMetabehaviour(metabehaviour)

    if str(metabehaviour.__class__.__name__) == "random":

        beat_dict, total_number_of_events = metabehaviour.eventCalculator(
            smallest_div_information, rhythm_information, beats_information
        )

        score = metabehaviour.run(
            choice_set,
            beat_dict,
            total_number_of_events,
            content_information,
            rhythm_information,
        )
        print(score)

    return score


# %%
