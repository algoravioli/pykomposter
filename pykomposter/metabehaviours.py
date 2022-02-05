#%%
import fractions
from re import L
import sys
import time

import music21
import numpy as np
import pandas as pd
import tensorflow
import tqdm
import pprint


class random:
    def __init__(self):
        super(random, self).__init__()

    def eventCalculator(self, smallest_div, rhythm_information, beats_information):
        # beats_information is the total duration of the generated material in 1/4 notes.
        beat_dict = dict()

        for i in range(beats_information):
            max_events_per_beat = np.ceil(1 / smallest_div)
            number_of_events_this_beat = np.random.randint(0, max_events_per_beat + 1)
            current_beat = []

            if number_of_events_this_beat != 0:
                for j in range(number_of_events_this_beat):
                    event_types = np.arange(1, number_of_events_this_beat + 1)
                    temp = np.array([])

                    for k in range(number_of_events_this_beat):
                        temp_num = 1 / event_types[k]
                        temp = np.append(temp, temp_num)
                    event_types = temp
                    sum_of_events_in_beat = 0

                    while sum_of_events_in_beat != 1:
                        for l in range(number_of_events_this_beat):
                            curr_event = np.random.choice(event_types)
                            current_beat.append(curr_event)
                        sum_of_events_in_beat = sum(current_beat)
                        # print(sum_of_events_in_beat)
                        if sum_of_events_in_beat != 1:
                            current_beat = []

            else:
                current_beat = [0]
            beat_dict[str(i)] = current_beat
        # pprint.pprint(beat_dict)
        total_number_of_events = 0
        for m in range(len(beat_dict)):
            curr_beat = beat_dict[f"{m}"]
            curr_beat_events = len(curr_beat)
            total_number_of_events += curr_beat_events

        return beat_dict, total_number_of_events

    def run(self, choice_set, beat_dict, total_number_of_events):
        print("a")
