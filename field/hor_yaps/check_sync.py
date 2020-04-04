import pyreadr

result = pyreadr.read_r('yaps/full/20180316T0152-20180321T0003/v04/sync_save.RDS')

## 
# done! let's see what we got
# result is a dictionary where keys are the name of objects and the values python
# objects
print(result.keys()) # let's check what objects we got
df1 = result["df1"] # extract the pandas data frame for object df1
