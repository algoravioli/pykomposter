import music21
import numpy as np
import pandas as pd

# function definitions
import actions

# behaviours:
import behaviours

# metabehaviours:
import metabehaviours

# microactions
import microactions


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

    def withBehaviour(self, behaviour, compose, state_transitions=100):
        print(f" state = {state_transitions}")
        score = compose(
            self.outlook["metabehaviour"],
            behaviour,
            self.outlook["op_char"],
            state_transitions,
        )

        return score
