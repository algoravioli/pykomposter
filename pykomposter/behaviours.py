import fractions
import sys
import time

import music21
import numpy as np
import pandas as pd
import tensorflow
import tqdm

import metabehaviours


### INTERVAL ANALYST #####
"""[summary]
"""


class intervalAnalyst:
    def __init__(self):
        super(intervalAnalyst, self).__init__()

    def generateIntervals(pitch_sequence):
        length_of_sequence = len(pitch_sequence)
        interval_sequence = np.array(np.zeros(length_of_sequence))
        for i in range(length_of_sequence):
            if i == 0:
                interval_sequence[i] = 0
            else:
                if pitch_sequence[i] >= pitch_sequence[i - 1]:
                    # next pitch is higher than previous pitch
                    interval_sequence[i] = pitch_sequence[i] - pitch_sequence[i - 1]
                else:
                    # next pitch is lower than previous pitch
                    # negative indicates moving downwards
                    interval_sequence[i] = -1 * (
                        pitch_sequence[i - 1] - pitch_sequence[i]
                    )
        return interval_sequence

    def Augmentations(self, pitch_sequence, number_of_augmentations=11):
        interval_augmentations_dictionary = dict()
        interval_sequence = self.generateIntervals(pitch_sequence)

        for j in range(number_of_augmentations):
            entryname = f"augmentation(+{j})"
            temp_interval_sequence = np.array(np.zeros(len(interval_sequence)))
            for k in range(len(interval_sequence)):
                if k == 0:
                    temp_interval_sequence[k] = 0
                else:
                    if interval_sequence[k] > 0:
                        temp_interval_sequence[k] = interval_sequence[k] + j
                    elif interval_sequence[k] < 0:
                        temp_interval_sequence[k] = interval_sequence[k] - j
                    elif interval_sequence[k] == 0:
                        temp_interval_sequence[k] = 0

            interval_augmentations_dictionary[entryname] = temp_interval_sequence

        return interval_augmentations_dictionary

    def Diminutions(self, pitch_sequence, number_of_diminutions=11):
        interval_diminutions_dictionary = dict()
        interval_sequence = self.generateIntervals(pitch_sequence)

        for j in range(number_of_diminutions):
            entryname = f"diminution(-{j})"
            temp_interval_sequence = np.array(np.zeros(len(interval_sequence)))
            for k in range(len(interval_sequence)):
                if k == 0:
                    temp_interval_sequence[k] = 0
                else:
                    if interval_sequence[k] == 0:
                        temp_interval_sequence[k] = 0

                    elif interval_sequence[k] > 0:
                        temp_value = interval_sequence[k] - j
                        if temp_value < 0:
                            temp_interval_sequence[k] = 0
                        else:
                            temp_interval_sequence[k] = temp_value

                    elif interval_sequence[k] < 0:
                        temp_value = interval_sequence[k] + j
                        if temp_value > 0:
                            temp_interval_sequence[k] = 0
                        else:
                            temp_interval_sequence[k] = temp_value
            interval_diminutions_dictionary[entryname] = temp_interval_sequence

        return interval_diminutions_dictionary

    def prepare(self, content_information):
        augments = self.Augmentations(content_information)
        diminutions = self.Diminutions(content_information)
        augments_df = pd.DataFrame(data=augments)
        diminutions_df = pd.DataFrame(data=diminutions)
        output_df = pd.concat([augments_df, diminutions_df], axis=1)
        return output_df

    def withMetabehaviour(self, metabehaviour_ref):
        metabehaviour_class = metabehaviour_ref()
        return metabehaviour_class


class markovModeller:
    def __init__(self):
        super(markovModeller, self).__init__()

    def reduceToPitchClass(self, input_array):
        output_array = []
        for i in range(len(input_array)):
            output_array.append((input_array[i] % 12))
        return output_array

    def inputArrayToTransitionArray(self, input_array):
        input_array = self.reduceToPitchClass(input_array)
        input_array = np.array(input_array)
        output_array = np.array([])
        for i in range(len(input_array)):
            if i > 0:
                output_array = np.append(
                    [output_array], [input_array[i - 1], input_array[i]], axis=0
                )

        return output_array

    def generateStateTransitionMatrix(self, input_array):  # [60,53,50,52,85]
        transition_array = self.inputArrayToTransitionArray(input_array)

        # return state_transition_matrix

    # def prepare(self, etc):

    def withMetabehaviour(self, metabehaviour_ref):
        metabehaviour_class = metabehaviour_ref()
        return metabehaviour_class


class finiteStateMachine:
    def __init__(self):
        super(finiteStateMachine, self).__init__()
