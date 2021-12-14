import pandas as pd
import os
import matlab.engine
import sys
import numpy as np
sys.argv=['','BIM','/data0/lhc/dataset/IQA_AE/REF/ref','./generated']
def calculate_MAD_append():
    csv_file='scores/'+sys.argv[1]+'_piq.csv'
    print(csv_file)
    eng = matlab.engine.start_matlab()
    MAD=[]
    df=pd.read_csv(csv_file)
    count=len(df['file'])
    for id,file in enumerate(df['file']): 
        ref_name=file[0:11]+'.jpg'
        ae_name=file
        # print(file,ref_name,ae_name)
        # print(os.path.join(sys.argv[2],ref_name))
        # print(os.path.join(sys.argv[3],sys.argv[1],ae_name))
        # # exit()
        r=eng.MAD_index(os.path.join(sys.argv[2],ref_name),
                        os.path.join(sys.argv[3],sys.argv[1],ae_name))
        MAD.append(r['MAD'])
        print(sys.argv[1],f'{(id+1)/count*100:.2f}%',file,r['MAD'])

    MAD=np.array(MAD)
    df['mad']=MAD
    df.to_csv(f'scores/{sys.argv[1]}.csv',index=False)

calculate_MAD_append()
#环境： conda ：py36
# cd /data0/lhc/construct_dataset_of_adversarial_examples/
#python calculate_MAD.py hnmPGD /data0/lhc/dataset/IQA_AE/REF/ref ./generated