#%%
import fractions
import sys
import time

import music21
import numpy as np
import pandas as pd
import tensorflow
import tqdm

# function definitions
import actions

# behaviours:
import behaviours

# metabehaviours:
import metabehaviours


class pykomposter:
    def __init__(self):
        super(pykomposter, self).__init__()
        self.outlook = {
            "tendency": None,  # tendency: how much of stated behaviour is it likely to follow. List: [2ndary behaviour, float] eg. [stochastic, 0.2]
            "metabehaviour": None,  # how the komposter model decides the actions to take. Reference (Variable):  e.g metabehaviour.random
            "op_char": dict(),  # operational characteristics: dict={} containing time-dependencies, and content-dependencies.
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

    def setMetaBehaviour(self, metabehaviour):
        self.outlook["metabehaviour"] = metabehaviour

    def setOpChar(self, opchardict):
        self.outlook["op_char"] = opchardict

    ##########################
    # BEHAVIOUR INTERACTIONS #
    ##########################

    def withBehaviour(self, behaviour, action):
        action(self.outlook["metabehaviour"], behaviour, self.outlook["op_char"])


# tests
music21.environment.set("musicxmlPath", "/usr/bin/musescore")

myKomposter = pykomposter()

time_and_content = {
    "time": {"beats": None, "smallest_div": None, "rhythm": [5, 4, 2, 5]},
    "content": [0, 1, 6, 10, 0, -4, 3],
}

myKomposter.setOpChar(time_and_content)
myKomposter.setMetaBehaviour(metabehaviours.random)

myKomposter.withBehaviour(
    behaviours.intervalAnalyst,
    actions.Compose,
)


# %%

# %%
