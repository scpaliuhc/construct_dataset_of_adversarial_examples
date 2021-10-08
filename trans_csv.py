import pandas
df=pandas.read_csv('scores/CW.csv',index_col=0)
df.to_csv('scores/CW1.csv',index=False)
print('end')