import fractions
import sys
import time

import music21
import numpy as np
import pandas as pd
import tensorflow
import tqdm


class pykomposter:
    def __init__(self):
        super(pykomposter, self).__init__()
        self.outlook = {
            "tendency": None,  # tendency: how much of stated behaviour is it likely to follow. List: [2ndary behaviour, float] eg. [stochastic, 0.2]
            "behaviour": None,  # behaviour of this komposter model. String: "string" eg. "klangtypen-structure"
            "specificity": None,  # how specific the komposter model is at returning the stated behaviour. Float: [0-1] eg. 0.5
            "sensitivity": None,  # how sensitive the komposter model is at returning the stated behaviour. Float: [0-1] eg. 0.5
            "op_char": dict(),  # operational characteristics: dict={} containing time-dependencies, and pitch-dependencies.
        }

    # setters
    def setTendency(self, tendency_list):
        if len(tendency_list) == 2:
            if isinstance(tendency_list[0], str):
                if isinstance(tendency_list[1], float):
                    self.outlook["tendency"] = tendency_list
                else:
                    raise RuntimeError(
                        "ERROR: 2nd argument of tendency needs to be a float."
                    )
            else:
                raise RuntimeError(
                    "ERROR: 1st argument of tendency needs to be a string."
                )
        else:
            raise RuntimeError("ERROR: Tendency list must only contain 2 elements")

    def setBehaviour(self, behaviour_string):
        if len(behaviour_string) == 1:
            if isinstance(behaviour_string, str):
                self.outlook["behaviour"] = behaviour_string
            else:
                raise RuntimeError("ERROR: Behaviour needs to be a string.")
        else:
            raise RuntimeError("ERROR: Behaviour string must only contain 1 element.")

    def setSpecificity(self, specificity):
        if isinstance(specificity, float) and len(specificity) == 1:
            self.outlook["specificity"] = specificity
        else:
            raise RuntimeError(
                "ERROR: Specificity must be a float and only contain 1 element."
            )

    def setSensitivity(self, sensitivity):
        if isinstance(sensitivity, float) and len(sensitivity) == 1:
            self.outlook["sensitivity"] = sensitivity
        else:
            raise RuntimeError(
                "ERROR: Sensitivity must be a float and only contain 1 element."
            )

    def setOpChar(self, opchardict):
        self.outlook["op_char"] = opchardict

    def train(self):
        print("HI")


myKomposter = pykomposter()

myKomposter.setTendency([1, 0.5, 0.2])

print(myKomposter.outlook["tendency"])
