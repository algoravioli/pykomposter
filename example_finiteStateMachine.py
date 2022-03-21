# %%
import music21
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("./pykomposter")

from pykomposter.lib import *

# if on linux, probably good to have
music21.environment.set("musicxmlPath", "/usr/bin/musescore")

# start a komposter object
myKomposter = pykomposter()
NumberOfBars = 30

# fill an array with time signatures
TimeSignatureArray = []
for i in range(NumberOfBars):
    Choice = np.random.choice([2, 3, 4, 1], p=[0.3, 0.3, 0.399, 0.001])
    TimeSignatureArray.append(Choice)

# fill an array with pitches
PitchContentArray = []
NumberOfStartingPitches = 20
for i in range(NumberOfStartingPitches):
    choice = np.random.choice(np.arange(54, 72))
    PitchContentArray.append(choice)

TimeAndContent = {
    "time": {
        "beats": TimeSignatureArray,
        "smallest_div": 0.25,
        "rhythm": None,
        "tempo": music21.tempo.MetronomeMark(
            "Slow", 40, music21.note.Note(type="quarter")
        ),
        "state_transitions": None,
    },
    "content": PitchContentArray,
    "instruments": {
        "1": music21.instrument.Flute(),
        "2": music21.instrument.Flute(),
        "3": music21.instrument.Flute(),
        "4": music21.instrument.Flute(),
    },
}

myKomposter.setOpChar(TimeAndContent)
myKomposter.setMetaBehaviour(metabehaviours.default)

score = myKomposter.withBehaviour(
    behaviours.finiteStateMachine(), actions.Compose, state_transitions=200
)

score.show("musicxml")
