import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a = range(5)
b = ['apple', 'banana', 'carrot', 'durian', 'elephant']
c = ['red', 'yellow', 'orange', 'green', 'grey']

d = pd.DataFrame({'words': b, 'colors': c}, index = a)

print "This is d"
print d
print

print "This is d.values"
print d.values
print 

print "This is d[['words', 'colors']]"
print d[['words', 'colors']]
print

print "This is d[['colors', 'words']]"
print d[['colors', 'words']]
print

print "This is d[['colors']]"
print d[['colors']]
print

print "This is d['colors']"
print d['colors']