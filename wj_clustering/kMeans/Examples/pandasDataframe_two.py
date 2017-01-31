import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

with open("vocab_frame.p", "rb") as h:
    vocab_frame = pickle.load(h)
    h.close()

a = vocab_frame.ix["cms"].values[0][0]
print a
#.tolist()[0][0]


