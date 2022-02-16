import fractions
import math

import numpy as np
import pandas as pd
import tensorflow
from tqdm import tqdm

import microactions
import matplotlib.pyplot as plt


class random:
    def __init__(self):
        super(random, self).__init__()

    def Log2(self, x):
        y = math.log10(x) / math.log10(2)
        return y

    def isPowerOfTwo(self, n):
        m = math.ceil(self.Log2(n)) == math.floor(self.Log2(n))
        return m

    def rhythmInvert(self, number):
        if self.isPowerOfTwo(number):
            number = 1 / number
        else:
            number = fractions.Fraction(1, number)
        return number

    def eventCalculator(
        self,
        smallest_div,
        rhythm_information,
        beats_information,
        rhythm_information_flag,
        total_beats_information,
        *args,
    ):
        # beats_information is the total duration of the generated material in 1/4 notes.
        if rhythm_information_flag == 1:
            beat_dict = rhythm_information(*args)

        else:
            beat_dict = dict()

            for i in tqdm(range(total_beats_information)):
                max_events_per_beat = np.ceil(1 / smallest_div)
                number_of_events_this_beat = np.random.randint(
                    0, max_events_per_beat + 1
                )
                current_beat = []
                pow_two_array = np.array([])
                sum_of_beats = 0

                for num in range(int(max_events_per_beat)):
                    if self.isPowerOfTwo(num + 1):
                        pow_two_array = np.append(pow_two_array, 1 / (num + 1))
                        # pow_two_array = np.append(pow_two_array, [0.75, 0.375])

                if number_of_events_this_beat != 0:
                    if number_of_events_this_beat == 1:
                        current_beat = [1.0]

                    else:
                        if self.isPowerOfTwo(number_of_events_this_beat):
                            while sum_of_beats != 1.0:
                                current_beat.append(
                                    np.random.choice(pow_two_array)  # p=[0.2, 0.5, 0.3]
                                )
                                sum_of_beats = sum(current_beat)
                                if sum_of_beats > 1.0:
                                    current_beat = []
                        else:
                            for a in range(number_of_events_this_beat):
                                current_beat.append(
                                    fractions.Fraction(1, number_of_events_this_beat)
                                )

                else:
                    current_beat = [0]
                # print(current_beat, i)
                beat_dict[str(i)] = current_beat
        # print(beat_dict)
        # pprint.pprint(beat_dict)
        total_number_of_events = 0
        for m in range(len(beat_dict)):
            curr_beat = beat_dict[f"{m}"]
            curr_beat_events = len(curr_beat)
            total_number_of_events += curr_beat_events

        return beat_dict, total_number_of_events

    def generateOneSetOfMeasures(
        self,
        choice_set,
        beat_dict,  # dictionary of information per beat (rhythm)
        total_number_of_events,  # dictionary of total number of events
        content_information,
        beats_information,  # array of number of beats per measure (time signature)
        total_beats_information,  # int of sum of all the beats
        random_choice_max=3,
        clamp=True,
    ):
        # print(np.random.randint(0, 100))
        choice_stack = np.array([])
        while len(choice_stack) < total_number_of_events - len(content_information):
            current_choice = int(
                np.abs(np.floor(np.random.default_rng().normal(0, random_choice_max)))
            )
            current_choice = np.abs(np.min([random_choice_max - 1, current_choice]))
            # print(current_choice)
            aug_or_dim = np.random.randint(0, 2)

            if aug_or_dim == 1:
                aug_or_dim = "augmentation(+"
            elif aug_or_dim == 0:
                aug_or_dim = "diminution(-"
            current_choice = choice_set[f"{aug_or_dim}{current_choice})"].tolist()
            current_choice = np.delete(current_choice, 0)
            choice_stack = np.append(choice_stack, current_choice)
        # somemore preprocessing
        processed_stack = np.array([])
        # plt.plot(choice_stack)
        clamp_value = 10
        if clamp == True:
            for i in range(len(choice_stack)):
                curr_note = choice_stack[i]
                curr_note = np.max(
                    [(-1 * clamp_value), np.min([curr_note, clamp_value])]
                )
                processed_stack = np.append(processed_stack, curr_note)

        content = np.array(content_information)
        note_of_interest = content[-1]
        for i in range(len(processed_stack)):
            note_of_interest = note_of_interest + processed_stack[i]
            if note_of_interest < 52:
                note_of_interest = note_of_interest + 12
            if note_of_interest > 82:
                note_of_interest = note_of_interest - 12
            content = np.append(content, note_of_interest)

        # run your music21 code here //
        measures = microactions.createMeasures(
            content, beat_dict, beats_information, total_beats_information
        )
        return measures

    def markovMeasureGenerator(
        self,
        state_transition_matrix,
        beat_dict,  # dictionary of information per beat (rhythm)
        total_number_of_events,  # dictionary of total number of events
        beats_information,  # array of number of beats per measure (time signature)
        total_beats_information,  # int of sum of all the beats
    ):
        hi = 1

    def run(
        self,
        choice_set,
        dict_of_beat_dicts,
        content_information,
        beats_information,
        behaviour_class,
        instruments_dict,
        metronome_mark,
        rhythm_information_flag,
        total_beats_information,
        random_choice_max=1,
        clamp=True,
        parts=1,
    ):
        print(f"The behaviour class chosen is {(behaviour_class.__class__.__name__)}")

        # new if statement for intervalAnalyst
        if str(behaviour_class.__class__.__name__) == "intervalAnalyst":

            part_dict = dict()
            for i in range(parts):
                np.random.seed(i)
                measures = self.generateOneSetOfMeasures(
                    choice_set,
                    dict_of_beat_dicts[f"part{i}"],
                    dict_of_beat_dicts[f"total_events{i}"],
                    content_information,
                    beats_information,
                    total_beats_information,
                    random_choice_max=random_choice_max,
                    clamp=clamp,
                )
                part_dict[f"part{i}"] = microactions.createPart(
                    measures, f"part{i}", instruments_dict[f"{i+1}"], metronome_mark
                )

            score = microactions.createScore(part_dict)

            return score

        # new if statement for markovModeller
        if str(behaviour_class.__class__.__name__) == "markovModeller":
            state_transition_matrix = choice_set
            print("The state transition matrix of the system is:")
            print(state_transition_matrix.round(2).to_markdown())

            for i in range(parts):
                measure = self.markovMeasureGenerator()
