#%%
import fractions
import sys
import time

import music21
import numpy as np
import pandas as pd
import tensorflow
import tqdm


def Compose(metabehaviour, behaviour_class, op_char):
    # sets metabehaviour string
    metabehaviour_string = metabehaviour

    # gets pitch information
    pitch_information = op_char["pitch"]

    behaviour_ref = behaviour_class()


# %%
