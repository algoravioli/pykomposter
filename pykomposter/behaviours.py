from asyncio import events
import fractions
import sys
import time

import matplotlib.pyplot as plt
import music21
import numpy as np
import pandas as pd
import tqdm

############################
# FOR FINITE STATE MACHINE #
############################
from transitions import Machine
from transitions.extensions import GraphMachine

##########################
# FOR THE XENAKIS MODELS #
##########################

import statsmodels.api as sm
import statsmodels.sandbox.distributions.extras as extras
import scipy.interpolate as interpolate
import scipy.stats as ss

#####################
# KOMPOSTER IMPORTS #
#####################

import metabehaviours


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


class simpleMarkovModeller:
    def __init__(self):
        super(simpleMarkovModeller, self).__init__()

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
    states = ["1", "2", "3", "4"]

    def __init__(self, state_transitions=100):
        super(finiteStateMachine, self).__init__()

        self.state_transitions = state_transitions

        # self.name = name
        self.OutputNotes = 0
        self.machine = GraphMachine(
            model=self, states=finiteStateMachine.states, initial="1"
        )
        self.machine.add_transition(
            trigger="toState1", source="*", dest="1", after="giveNote1"
        )  # before and after is able to run a function

        self.machine.add_transition(
            trigger="toState2", source="*", dest="2", after="giveNote2"
        )

        self.machine.add_transition(
            trigger="toState3", source="*", dest="3", after="giveNote3"
        )

        self.machine.add_transition(
            trigger="toState4", source="1", dest="4", after="giveNote4"
        )

        self.pitchdict = {"1": [], "2": [], "3": [], "4": []}

        self.rhythmdict = {"1": [], "2": [], "3": [], "4": []}

    # def giveNote(self, affected_array_number):
    #     random_pitch = np.random.randint(48, 70)
    #     random_rhythm = np.random.choice([1, 0.75, 0.5, 0.25])

    #     # for pitch
    #     for i in range(len(self.pitchdict)):
    #         if i == affected_array_number:
    #             self.pitchdict[f"{affected_array_number}"].append(random_pitch)
    #         else:
    #             self.pitchdict[f"i"].append(0)

    #     for i in range(len(self.rhythmdict)):
    #         self.rhythmdict[f"i"].append(random_rhythm)

    def giveNote1(self):
        random_pitch = np.random.randint(60, 78)
        random_rhythm = np.random.choice([1, 0.5, 0.25])

        # for pitch
        for i in range(len(self.pitchdict)):
            if i == 1:
                self.pitchdict[f"1"].append(random_pitch)
            else:
                self.pitchdict[f"{i+1}"].append(0)

        for i in range(len(self.rhythmdict)):
            self.rhythmdict[f"{i+1}"].append(random_rhythm)

    def giveNote2(self):
        random_pitch = np.random.randint(60, 78)
        random_rhythm = np.random.choice([1, 0.5, 0.25])

        # for pitch
        for i in range(len(self.pitchdict)):
            if i == 2:
                self.pitchdict[f"2"].append(random_pitch)
            else:
                self.pitchdict[f"{i+1}"].append(0)

        for i in range(len(self.rhythmdict)):
            self.rhythmdict[f"{i+1}"].append(random_rhythm)

    def giveNote3(self):
        random_pitch = np.random.randint(60, 78)
        random_rhythm = np.random.choice([1, 0.5, 0.25])

        # for pitch
        for i in range(len(self.pitchdict)):
            if i == 3:
                self.pitchdict[f"3"].append(random_pitch)
            else:
                self.pitchdict[f"{i+1}"].append(0)

        for i in range(len(self.rhythmdict)):
            self.rhythmdict[f"{i+1}"].append(random_rhythm)

    def giveNote4(self):
        random_pitch = np.random.randint(60, 78)
        random_rhythm = np.random.choice([1, 0.5, 0.25])

        # for pitch
        for i in range(len(self.pitchdict)):
            if i == 1:
                self.pitchdict[f"4"].append(random_pitch)
            else:
                self.pitchdict[f"{i+1}"].append(0)

        for i in range(len(self.rhythmdict)):
            self.rhythmdict[f"{i+1}"].append(random_rhythm)

    def prepare(self, Machine):

        self.list_of_state1_transitions = [
            self.toState1,
            self.toState2,
            self.toState3,
            self.toState4,
        ]
        self.list_of_state2_transitions = [self.toState1, self.toState3]
        self.list_of_state3_transitions = [self.toState1, self.toState2, self.toState3]
        self.list_of_state4_transitions = [self.toState1]
        self.transition_functions = {
            "1": self.list_of_state1_transitions,
            "2": self.list_of_state2_transitions,
            "3": self.list_of_state3_transitions,
            "4": self.list_of_state4_transitions,
        }
        # print(self.state_transitions)
        for i in range(self.state_transitions):
            current_state = int(Machine.state)
            np.random.choice(self.transition_functions[f"{current_state}"])()

        output_dict = {"pitch": self.pitchdict, "rhythm": self.rhythmdict}
        return output_dict

    def withMetabehaviour(self, metabehaviour_ref):
        metabehaviour_class = metabehaviour_ref()
        return metabehaviour_class


class cubesInCubes:
    def __init__(self, cubeDict={"rotationlist": None}):
        super(cubesInCubes, self).__init__()
        self.cubeDict = cubeDict
        self.rotationList = cubeDict["rotationlist"]

    def rotateUp(self, list):
        order = [3, 4, 7, 8, 5, 6, 1, 2]
        output_list = []
        for i in range(8):
            current_order = order[i] - 1
            output_list.append(list[current_order])

        return output_list

    def rotateDown(self, list):
        order = [5, 6, 1, 2, 7, 8, 3, 4]
        output_list = []
        for i in range(8):
            current_order = order[i] - 1
            output_list.append(list[current_order])

        return output_list

    def rotateLeft(self, list):
        order = [2, 6, 4, 8, 1, 5, 3, 7]
        output_list = []
        for i in range(8):
            current_order = order[i] - 1
            output_list.append(list[current_order])

        return output_list

    def rotateRight(self, list):
        order = [5, 1, 7, 3, 6, 2, 8, 4]
        output_list = []
        for i in range(8):
            current_order = order[i] - 1
            output_list.append(list[current_order])

        return output_list

    def rhythmToBarConvert(self, number):
        if number == 0.25:
            output = [0.25, 0.25, 0.25, 0.25]
        if number == 0.5:
            output = [0.5, 0.5]
        if number == 0.75:
            output = np.random.choice([1, 2])
            if output == 1:
                output = [0.75, 0.25]
            else:
                output = [0.25, 0.75]
        if number == 1:
            output = [1]
        return output

    def eventProvider(self, number):
        if number == 0.25:
            output = np.random.choice([1, 2, 3, 4])
        if number == 0.5:
            output = np.random.choice([1, 2])
        if number == 0.75:
            output = np.random.choice([1, 2])
        if number == 1:
            output = 1
        return output

    def formPitchArray(self, rhythm, positionOfEvent, pitch):
        if rhythm == 0.25:
            array_length = 4
        if rhythm == 0.5 or 0.75:
            array_length = 2
        if rhythm == 1:
            array_length = 1

        pitch_array = []
        for i in range(array_length):
            if positionOfEvent == i:
                pitch_array.append(pitch)
            else:
                pitch_array.append(0)

        return pitch_array

    def prepare(
        self,
        beats_information,
        total_beats_information,
        instruments_information,
        smallest_div_information,
    ):
        all_parts_dict = {"pitch": {}, "rhythm": {}}
        for i in range(len(self.rotationList)):
            current_action = self.rotationList[i].split(",")
            current_cube = current_action[0]
            current_action = current_action[1]
            print(self.cubeDict[f"cube{current_cube}"])
            if current_action == "up":
                self.cubeDict[f"cube{current_cube}"] = self.rotateUp(
                    self.cubeDict[f"cube{current_cube}"]
                )
            elif current_action == "down":
                self.cubeDict[f"cube{current_cube}"] = self.rotateDown(
                    self.cubeDict[f"cube{current_cube}"]
                )
            elif current_action == "left":
                self.cubeDict[f"cube{current_cube}"] = self.rotateLeft(
                    self.cubeDict[f"cube{current_cube}"]
                )
            elif current_action == "right":
                self.cubeDict[f"cube{current_cube}"] = self.rotateRight(
                    self.cubeDict[f"cube{current_cube}"]
                )

            for j in range(len(self.cubeDict["cube3"])):
                curr_rhythm = self.cubeDict["cube2"][j]
                all_parts_dict["rhythm"][f"{j+1}"] = {
                    f"{i}": self.rhythmToBarConvert(curr_rhythm)
                }
                position_of_event = self.eventProvider(curr_rhythm)

                curr_pitch = self.cubeDict["cube1"][j]
                all_parts_dict["pitch"][f"{j+1}"] = {
                    f"{i}": self.formPitchArray(
                        curr_rhythm, position_of_event, curr_pitch
                    )
                }

        return all_parts_dict

    def withMetabehaviour(self, metabehaviour_ref):
        metabehaviour_class = metabehaviour_ref()
        return metabehaviour_class


class roidoRipsis:
    def __init__(self, mu, sigma, skew, kurt):
        super(roidoRipsis, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.skew = skew
        self.kurt = kurt

    def generate_normal_four_moments(mu, sigma, skew, kurt, sd_wide=10, size=10000):
        ###########################
        # GRAM-CHARLIER EXPANSION #
        ###########################
        f = extras.pdf_mvsk([mu, sigma, skew, kurt])
        x = np.linspace(mu - sd_wide * sigma, mu + sd_wide * sigma, num=500)
        y = [f(i) for i in x]
        yy = np.cumsum(y) / np.sum(y)
        inv_cdf = interpolate.interp1d(yy, x, fill_value="extrapolate")
        rr = np.random.rand(size)
        return inv_cdf(rr)

    def getPitchAndRhythmForBar(self, num_events, bar_length, smallest_div):
        pitch_array = []
        rhythm_array = []
        div_changer = smallest_div
        possible_event_locations = bar_length / div_changer
        exitTrigger = 0
        while possible_event_locations < num_events:
            div_changer = np.random.choice([smallest_div, 0.125, 0.0625])
            possible_event_locations = bar_length / div_changer
            exitTrigger += 1
            print(
                f"possible={possible_event_locations},num_events={num_events},{possible_event_locations< num_events}"
            )
            if exitTrigger > 10e8:
                assert (
                    exitTrigger == 10e8
                ), "Roidoripsis could not find an adequate solution to the provided distribution values. Please exit and try again, or restart the code."

            # div changer is now the smallest rhythm possible
        loop_flag = 0
        beats = []
        while loop_flag != num_events:
            possible_array = list(range(int(possible_event_locations)))
            possible_array2 = [i + 1 for i in possible_array]
            beat_chosen = np.random.choice(possible_array2)
            if beat_chosen not in beats:
                beats.append(beat_chosen)
                loop_flag = len(beats)
            # print(
            # f"loopflag={loop_flag},num_events={num_events},possible={possible_event_locations},t/f={loop_flag != num_events}"
            # )
            # print(beats)

        for i in range(int(possible_event_locations)):
            if i in beats:
                curr_pitch = np.random.choice(np.arange(60, 73))
                # print(curr_pitch)
                # time.sleep(4)
                pitch_array.append(curr_pitch)
            else:
                pitch_array.append(0)
            # print(pitch_array)
            # time.sleep(3)
            rhythm_array.append(div_changer)

        return pitch_array, rhythm_array

    def prepare(
        self,
        beats_information,
        total_beats_information,
        number_of_instruments,
        smallest_div,
    ):
        data = roidoRipsis.generate_normal_four_moments(
            self.mu, self.sigma, self.skew, self.kurt
        )
        print(f"The generated distribution is as follows...")
        plotdata = plt.hist(data)
        plt.show()
        bins_info = plotdata[0]
        event_array = np.floor(plotdata[1])
        smallest_value = np.min(event_array)
        smallest_value = (smallest_value * -1) + 1
        event_array2 = event_array.tolist()
        event_array = [i + smallest_value for i in event_array2]
        # print(event_array)

        ###########################
        # POSSIBLE EVENTS NUMBERS #
        ###########################
        # output_event_array = []
        # for i in range(len(event_array)):
        #     curr_num = event_array[i]
        #     int_num = int(curr_num)
        #     output_event_array.append(int_num)

        # event_array = output_event_array

        probabilities_of_events_per_cell = bins_info / np.sum(bins_info)

        ###############
        # SAFETY CODE #
        ###############
        if len(event_array) != len(probabilities_of_events_per_cell):
            length_event_array = len(event_array)
            length_probs = len(probabilities_of_events_per_cell)
            if length_event_array > length_probs:
                difference = length_event_array - length_probs
                event_array = event_array[: -difference or None]
            if length_event_array < length_probs:
                difference = length_probs - length_event_array
                probabilities_of_events_per_cell = probabilities_of_events_per_cell[
                    : -difference or None
                ]

        all_parts_dict = {"pitch": dict(), "rhythm": dict()}
        for h in range(number_of_instruments):
            total_pitch_array = np.array([])
            total_rhythm_array = np.array([])
            for i in range(len(beats_information)):
                number_of_events_this_bar = np.random.choice(
                    event_array, p=probabilities_of_events_per_cell
                )

                curr_bar_pitch, curr_bar_rhythm = self.getPitchAndRhythmForBar(
                    number_of_events_this_bar, beats_information[i], smallest_div
                )

                total_pitch_array = np.append(total_pitch_array, curr_bar_pitch)
                total_rhythm_array = np.append(total_rhythm_array, curr_bar_rhythm)
                # print(i)

            # print(total_pitch_array)
            all_parts_dict["pitch"][f"{h+1}"] = total_pitch_array

            all_parts_dict["rhythm"][f"{h+1}"] = total_rhythm_array
        # print(all_parts_dict)
        # time.sleep(7)
        return all_parts_dict

    def withMetabehaviour(self, metabehaviour_ref):
        metabehaviour_class = metabehaviour_ref()
        return metabehaviour_class
