import pandas as pd

filename = 'datas/quotes.h5'

df = pd.read_csv("D:/materials/finance/2009年后简化行情.csv")

# Save to HDF5
df.to_hdf(filename, 'quotes', mode='w', format='table')
# del df    # allow df to be garbage collected