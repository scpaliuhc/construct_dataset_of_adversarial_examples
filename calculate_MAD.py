import pandas as pd
import os
import matlab.engine
import sys
import numpy as np
def calculate_MAD_append():
    csv_file='scores_1/'+sys.argv[1]+'_piq.csv'
    eng = matlab.engine.start_matlab()
    MAD=[]
    df=pd.read_csv(csv_file)

    for file in df['file']:
        
        ref_name=file[0:11]+'.jpg'
        ae_name=file
        r=eng.MAD_index(os.path.join('D:\\adata\\paper\\dataset\\adversarial examples for IQA\\REF_VOC',ref_name),
                        os.path.join('D:\\adata\\paper\\dataset\\adversarial examples for IQA',sys.argv[1],ae_name))
        MAD.append(r['MAD'])
        print(file,r['MAD'])
    MAD=np.array(MAD)

    df['mad']=MAD
    df.to_csv(f'scores_1/{sys.argv[1]}.csv',index=False)
calculate_MAD_append()