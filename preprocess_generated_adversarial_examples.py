import os
#rename
def get_files(root):
    return os.listdir(root)
def rename_(root,old,new):
    os.rename(os.path.join(root,old),os.path.join(root,new))
def re_BIM(root,file):
    pre=file[0:12]
    params=file.split('(')[-1].split(')')[0].split(',')
    med=f'BIM_({params[0]},{params[1]})'
    pos='.png'
    new_file=pre+med+pos
    rename_(root,file,new_file)
def re_Boundary(root,file):
    if '_20000.png' in file or '_25000.png' in file:
        print(file)
        os.remove(os.path.join(root,file))
def re_CW(root,file):
    if '(10,' in file or '(300,' in file:
        print(file)
        os.remove(os.path.join(root,file))
def re_DeepFool(root,file):
    if '_60.png' not in file:
        print(file)
        os.remove(os.path.join(root,file))
def re_FGSM(root,file):
    ep=float(file.split('_')[-1][:-4])
    if ep not in [0.003,0.01,0.03,0.05,0.07,0.1,0.3,0.5,0.7]:
        os.remove(os.path.join(root,file))
def re_NES(root,file):
    if 'ILSVRC2012' in file:
        os.remove(os.path.join(root,file))
        return None
    if 'NES_query_limited' in file:
        pre=file.split('NES_query_limited')[0]
        pos=file.split('NES_query_limited')[-1]
        med='NES-ql'
    elif 'NES_parial_info' in file:
        pre=file.split('NES_parial_info')[0]
        pos=file.split('NES_parial_info')[-1]
        med='NES-pi'
    new_name=pre+med+pos
    rename_(root,file,new_name)
def re_Square(root,file):
    if 'ILSVRC2012' in file:
        os.remove(os.path.join(root,file))
        return None
    if '0.6, 0.12).png' in file or '0.6, 0.06).png' in file or '0.6, 0.09).png' in file:
        print(file)
        os.remove(os.path.join(root,file))
def re_SimBA(root,file):
    if 'ILSVRC2012' in file:
        os.remove(os.path.join(root,file))
        return None
    if ').png' not in file:
        print(file)
        os.remove(os.path.join(root,file))

    # new_name=pre+med+pos
    # rename_(root,file,new_name)
def re_AdvPatch(root,file):

    if 'ILSVRC2012' in file:
        os.remove(os.path.join(root,file))
        return None
    pre=file[0:12]
    size=float(file.split('(')[-1].split(')')[0].split(',')[3].split('=')[-1])
    med=f'AdvPatch_{size}'
    pos='.png'
    new_file=pre+med+pos
    print(new_file)
    rename_(root,file,new_file)
def rename(root,scheme):
    files=get_files(root)
    if scheme=='BIM':
        for file in files:
            re_BIM(root,file)
    elif scheme=='Boundary':
        for file in files:
            re_Boundary(root,file)
    elif scheme=='CW':
        for file in files:
            re_CW(root,file)
    elif scheme=='DeepFool':
        for file in files:
            re_DeepFool(root,file)
    elif scheme=='FGSM':
        for file in files:
            re_FGSM(root,file)
    elif scheme=='NES':
        for file in files:
            re_NES(root,file)
    elif scheme=='Square':
        for file in files:
            re_Square(root,file)
    elif scheme=='SimBA':
        for file in files:
            re_SimBA(root,file)
    elif scheme=='AdvPatch':
        for file in files:
            re_AdvPatch(root,file)
if __name__=='__main__':
    rename('D:\\adata\\paper\\dataset\\adversarial examples for IQA\\AdvPatch','AdvPatch')