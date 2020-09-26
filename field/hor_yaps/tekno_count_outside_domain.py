# For the discussion in the swim speed paper, count how many of the teknologics position
# estimates fall outside the wetted domain, compared to yaps positions.
import matplotlib.pyplot as plt
import pandas as pd

##
tekno=pd.read_csv("../tags/cleaned_half meter.csv")


##

plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
ax.plot(tekno['X_UTM'],tekno['Y_UTM'],'k.')
