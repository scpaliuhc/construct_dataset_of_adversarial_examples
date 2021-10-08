# -*- encoding: utf-8 -*-
'''
Filename         :calculate_scores_piq.py
Description      :use build-in functions (['ssim','ms_ssim','psnr','vif_p','vsi','fsim','gmsd','ms_gmsd','haarpsi','mdsi'])                 of piq to calculate scores and produce a csv file e.g. ./scores/BIM_piq.csv
Time             :2021/09/27 19:02:16
Author           :hc liu
Version          :1.0
'''
import csv
import os
import pickle
import numpy
from piq import vif
from piq.functional.filters import prewitt_filter
import torch 
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, dataloader
import PIL.Image as Image
import piq
import pandas as pd
import argparse



def get_ref_images(root):
    if not os.path.exists('./ref_names.txt'):
        print(f'No ref_names.txt, get files from {root}')
        ref_names=os.listdir(root)
        with open('./ref_names.txt','wb') as f:
            pickle.dump(ref_names,f)
        print(f'write ref_names into ./ref_names.txt')
        
    else:
        with open('./ref_names.txt','rb') as f:
            ref_names=pickle.load(f)
        print(f'load ref_names from ./ref_names.txt')
    return ref_names
class AEData(Dataset):
    def __init__(self,ref_root,ae_root,ae_method,transform):
        """[summary]

        Args:
            ref_root ([type]): [description]
            ae_root ([type]): [description]
            ae_method ([type]): [description]
            transform ([type]): [description]
        """
        
        # super(AEData,self).__init__()
        self.ref_root=ref_root
        self.ae_root=ae_root
        self.ae_mth=ae_method
        self.trans=transform
        self.ref_names=get_ref_images(ref_root)
        self.ae_names=os.listdir(os.path.join(self.ae_root,self.ae_mth))
        self.ae_names.sort()
    def __len__(self):
        return len(self.ref_names)

    def __getitem__(self, index):
        #load ref_image
        ref_img=Image.open(os.path.join(self.ref_root,self.ref_names[index]))
        ref_img=self.trans(ref_img)
        # return ref_img
        pre=self.ref_names[index][:-4]
        found=False
        sel_files=[]
        for file in self.ae_names:
            if not found and len(sel_files)>0:
                for f in sel_files:
                    self.ae_names.remove(f)
                break
            if pre in file:
                found=True
                sel_files.append(file)
            else:
                found=False
        
        ae_imgs=[]
        try:
            for file in sel_files:
                ae_img=Image.open(os.path.join(os.path.join(self.ae_root,self.ae_mth,file)))
                ae_img=self.trans(ae_img)
                ae_imgs.append(ae_img)
        except:
            print(f"An error happened on {file}, the corresponding ref file {self.ref_names[index]}")
            exit()
        return ref_img,ae_imgs,self.ref_names[index],sel_files
def calculate_score_(names,refs,aes,cols):
    global useGPU
    tmp_dic={'file':names}
    for func in cols:
        metr=func
        try:
            if "ms_" == func[0:3]:
                func="multi_scale_"+func[3:]
            if useGPU:
                tmp_dic[metr]=eval('piq.'+func)(refs,aes,reduction='none').cpu().detach().numpy()
            else:
                tmp_dic[metr]=eval('piq.'+func)(refs,aes,reduction='none').numpy()
        except:
            print(func)
            exit()
    return tmp_dic
def calculate_score(loader,csv_file,batch_size,cols,mth,init):
    global useGPU
    dic={}
    for metric in cols:
        dic[metric]=[]
    
    error_files=[]
    for id,(ref,aes,ref_name,ae_names) in enumerate(loader):
        print(mth,id,ref_name[0],len(ae_names))
        if len(ae_names)<11:
            error_files.append(ref_name[0])
        ae_count=len(ae_names)
        refs=ref.repeat(ae_count,1,1,1)
        aes=torch.cat(aes)
        if useGPU:
            aes=aes.to('cuda')
            refs=refs.to('cuda')
        for i in range(ae_count):
            ae_names[i]=ae_names[i][0]
        for i in range(0,ae_count,batch_size):
            tmp_dic=calculate_score_(ae_names[i:(i+1)*batch_size],refs[i:(i+1)*batch_size],aes[i:(i+1)*batch_size],cols)
            
            d=pd.DataFrame(tmp_dic)
            if init:
                d.to_csv(csv_file,index=False,sep=',',mode='a')
                init=False
            else:
                d.to_csv(csv_file,index=False,sep=',',mode='a',header=None)     
    print('error files \n',error_files)                 
def main(args):
    tran=transforms.Compose([transforms.ToTensor()])
    aedata=AEData(args.ref,args.adv,args.method,tran)
    loader=DataLoader(aedata,1,False)
    funcs=['ssim','ms_ssim','psnr','vif_p','vsi','fsim','gmsd','ms_gmsd','haarpsi','mdsi']
    csv_file=f'./scores/{args.method}_piq.csv'
    # init=False
    if os.path.exists(csv_file):
        print(csv_file, 'has been created. Append?[y/n]')
        yn=input()
        if yn=='y':
            init=True
        elif yn=='n':
            exit('stop calculation')
        else:
            raise('error input [y/n]')
    else:
        init=False
    calculate_score(loader,csv_file,args.batchSize,funcs,args.method,init)

if __name__=='__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
    parser.add_argument("--method", required=True ,type=str, choices=['FGSM','BIM','PGD','CW','DeepFool','OnePixel','Square','SparseFool','Boundary','SimBA','AdvPatch','GAP','NES'], help="the attack method.")
    parser.add_argument("--ref", type=str, default='D:\\adata\\论文\\dataset\\adversarial examples for IQA\REF_VOC', help="")
    parser.add_argument("--adv", type=str, default='D:\\adata\\论文\\dataset\\adversarial examples for IQA', help="")
    parser.add_argument("--batchSize", type=int, default=10, help="")
    parser.add_argument("--gpu", type=int, default=1, help="")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}' 
    useGPU=False
    if torch.cuda.is_available():
        useGPU=True
        print(f'use GPU and the id is {args.gpu}')
    main(args)

        

        