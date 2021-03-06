import os 
import pandas

rows=0
dfs=[]
for file in os.listdir('./scores'):
    if 'combined' in file or "piq" in file:
        continue
    df=pandas.read_csv(os.path.join('./scores',file),encoding='gbk',header=0)
    print(file,df.shape,df.columns.values)
    rows+=df.shape[0]
    dfs.append(df)


combined=pandas.concat(dfs,ignore_index=True)
print(combined.shape,rows,combined.columns.values,)
print(combined.head)
combined.sort_values(by='file',ascending=False)
combined.to_csv('./scores/combined.csv',index=False,header=df.columns.values)