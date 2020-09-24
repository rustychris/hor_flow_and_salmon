import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
## 

tags=pd.read_excel("../../../field/circulation/2018_Data/2018FriantTaggingCombined.xlsx")

##
plt.figure(1).clf()
fig,ax=plt.subplots(num=1)
fig.set_size_inches( (4.5,2.8), forward=True)

tags['release']=[s.upper()[:8] for s in tags['Rel_group']]

sns.kdeplot( tags.loc[ tags.release=='SJSCARF1', 'Length'], ax=ax,label='Upper', shade=True)
sns.kdeplot( tags.loc[ tags.release=='SJSCARF2', 'Length'], ax=ax,label='Lower', shade=True)

ax.set_ylabel("Density")
ax.set_xlabel("Fork length (mm)")
ax.axis(xmin=68,xmax=87)
ax.legend(loc='upper right',title='Release')
fig.subplots_adjust(bottom=0.17,left=0.15,top=0.95,right=0.93)

fig.savefig('fork-length.png',dpi=200)
