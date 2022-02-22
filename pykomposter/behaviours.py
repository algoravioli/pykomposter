import fractions
import sys
import time

import music21
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import tensorflow
import tqdm

import metabehaviours


### INTERVAL ANALYST #####
"""[summary]
"""


class intervalAnalyst:
    def __init__(self):
        super(intervalAnalyst, self).__init__()

    def generateIntervals(self, pitch_sequence):
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
        for i in range(len(input_array)):
            if i > 0:
                if i == 1:
                    output_array = [[input_array[i - 1], input_array[i]]]
                if i > 1:
                    temp_array = [input_array[i - 1], input_array[i]]
                    output_array.append(temp_array)

        return output_array

    def createStateTransitionCounterUnique(self):
        all_units = []
        for i in range(12):
            for j in range(12):
                left_unit = i
                right_unit = j
                unit = [i, j]
                all_units.append(unit)
        return all_units

    def generateStateTransitionMatrix(self, input_array):  # [60,53,50,52,85]
        transition_array = self.inputArrayToTransitionArray(input_array)
        unit_array = self.createStateTransitionCounterUnique()
        count_array = []
        for i in unit_array:
            counter = 0
            for j in transition_array:
                if i == j:
                    counter += 1
            count_array.append(counter)

        state_transition_matrix = np.reshape(np.array(count_array), (12, 12))

        dummy_array = []
        stm_output = []
        for k in range(len(state_transition_matrix)):
            current_row = state_transition_matrix[k]
            row_total_occurences = np.sum(current_row)
            if row_total_occurences == 0:
                dummy_array = np.zeros(12)
                dummy_array = dummy_array.tolist()
            else:
                for l in range(len(current_row)):
                    current_num = current_row[l] / row_total_occurences
                    dummy_array.append(current_num)

            stm_output.append(dummy_array)
            dummy_array = []
        state_transition_matrix = stm_output
        return state_transition_matrix

    def showSTM(self, stm):
        plot_mat = stm
        plot_df = pd.DataFrame(data=plot_mat)
        plt.matshow(plot_df)
        plt.colorbar()
        plt.show()

    def prepare(self, content_information):
        output = self.generateStateTransitionMatrix(content_information)
        output_df = pd.DataFrame(data=output)
        self.showSTM(output)
        return output_df

    def withMetabehaviour(self, metabehaviour_ref):
        metabehaviour_class = metabehaviour_ref()
        return metabehaviour_class


class finiteStateMachine:
    def __init__(self):
        super(finiteStateMachine, self).__init__()
