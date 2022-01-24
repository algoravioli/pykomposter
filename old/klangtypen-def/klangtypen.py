import sys
import time

import matplotlib.pyplot as plt
import music21
import numpy as np
import pandas as pd
from tqdm import tqdm

#eigenzeit (distinguishing duration, how long you need in order to perceive the sound)

## _____DEFINITE EIGENZEIT_______________________________________________________________________
## Kadenzklang (cadence sound) -- main process                                                  |
## Impulsklang (impulse sound)                                                                  |
## Einschwingklang (attack sound)                                                               |
## Ausschwingklang (decay sound)                                                                |
## _____________________________________________________________________________________________|

## ______INDEFINITE EIGENZEIT_______________________________________________________________
## Farbklang (colour sound)                                                                 |
## Fluktuationsklang (fluctuation sound, outer contour remains static, internally moving)   |
## Texturklang (texture sound)                                                              |
## _________________________________________________________________________________________|

## COMBINATIONS OF ABOVE
## Strukturklang (structure sound)


