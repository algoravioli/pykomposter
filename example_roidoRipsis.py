# %%
import music21
import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("./pykomposter")

from pykomposter.lib import *

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
        "2": music21.instrument.Oboe(),
        "3": music21.instrument.Violin(),
        "4": music21.instrument.Viola(),
        "5": music21.instrument.Violoncello(),
    },
}

myKomposter.setOpChar(TimeAndContent)
myKomposter.setMetaBehaviour(metabehaviours.default)

score = myKomposter.withBehaviour(
    behaviours.roidoRipsis(mu=2, sigma=1, skew=100, kurt=10), actions.Compose
)

score.show("musicxml")
