#%%
import music21
import numpy as np
import pandas as pd

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

    def withBehaviour(self, behaviour, compose):
        score = compose(
            self.outlook["metabehaviour"], behaviour, self.outlook["op_char"]
        )

        return score


# tests()
music21.environment.set("musicxmlPath", "/usr/bin/musescore")
# music21.environment.set("graphicsPath", "/usr/bin/lilypond")
us = music21.environment.UserSettings()
us["lilypondPath"] = "/usr/bin/lilypond"
myKomposter = pykomposter()

rhythm_arr = []

for i in range(50):
    choice = np.random.choice([2, 3, 4, 5, 7, 9])
    rhythm_arr.append(choice)

content_arr = []

for i in range(12):
    choice = np.random.choice(np.arange(54, 72))
    content_arr.append(choice)

time_and_content = {
    "time": {
        "beats": None,
        "smallest_div": 0.25,
        "rhythm": rhythm_arr,
    },
    "content": content_arr,
}

print(content_arr)  # [69, 70, 63, 56, 68, 70, 61, 54, 63, 56, 63, 67]
myKomposter.setOpChar(time_and_content)
myKomposter.setMetaBehaviour(metabehaviours.random)
# %%
score = myKomposter.withBehaviour(
    behaviours.intervalAnalyst,
    actions.Compose,
)

score.show("musicxml")


# %%
