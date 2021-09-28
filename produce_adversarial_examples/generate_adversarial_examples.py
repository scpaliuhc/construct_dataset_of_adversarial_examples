from numpy import mod
from numpy.core.fromnumeric import argmax, clip
from scipy.stats.stats import mode
import eagerpy as ep
import foolbox.attacks as fatk
import os
import imageio
from skimage import img_as_ubyte
import torch
import torchattacks as tatk
import time
import numpy as np
from utils import load_target_model



def attack_torchattacks(model,args,images,labels,files):
    if args.tool=='torchattacks':
        print('torchattacks')
    elif args.tool=='foolbox':
        print('foolbox')
    else:
        raise NotImplementedError
    
    method=args.method
    root=f"/data0/lhc/dataset/IQA_AE/AE/CLS/{method}"
    try: os.makedirs(root)
    except: None
  

    #white-box
    if method=='FGSM':
        epsilons=[0.003,0.006,0.009,0.01,0.03,0.05,0.07,0.09,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7]#[0.003,0.006,0.009,0.01,0.03,0.05,0.07,0.09,0.1,]
        for pa in epsilons:
            print(f'start attack: {method} {pa} {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}')
            atk=tatk.FGSM(model,eps=pa)
            succ=attack_(model,atk,images,labels,files,beg=args.beg,end=args.end,method=method,pa=pa,root=root)
            print(f'end. success rate {succ*100:.2f}% {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}\n')
    elif method=='BIM':
        params=((0.05,0.005,10),(0.05,0.0025,20),
                (0.1,0.01,10),(0.1,0.005,20),
                (0.2,0.02,10),(0.2,0.01,20),
                (0.3,0.03,10),(0.3,0.015,20),)
        for pa in params:
            eps=pa[0]
            alpha=pa[1]
            steps=pa[2]
            print(f'start attack: {method} {pa} {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}')
            atk=tatk.BIM(model,eps=eps,alpha=alpha,steps=steps)
            succ=attack_(model,atk,images,labels,files,beg=args.beg,end=args.end,method=method,pa=pa,root=root)
            print(f'end. success rate {succ*100:.2f}% {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}\n')
    elif method=='PGD':
        params=((0.05,0.005,10),(0.05,0.0025,20),
                (0.1,0.01,10),(0.1,0.005,20),
                (0.2,0.02,10),(0.2,0.01,20),
                (0.3,0.03,10),(0.3,0.015,20),)
        for pa in params:
            eps=pa[0]
            alpha=pa[1]
            steps=pa[2]
            print(f'start attack: {method} {pa} {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}')
            atk=tatk.PGD(model,eps=eps,alpha=alpha,steps=steps,random_start=True)
            succ=attack_(model,atk,images,labels,files,beg=args.beg,end=args.end,method=method,pa=pa,root=root)
            print(f'end. success rate {succ*100:.2f}% {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}\n')
    elif method=='CW':
        params=((30,0.03,50),(30,0.01,50),(30,0.06,50),(300,0.03,50),(300,0.01,50),(300,0.06,50))
            #((10,0.03,50),(10,0.01,100))#(20,0.05),(30,0.09),(40,0.1),(50,0.1),(60,0.12)
        for pa in params:
            conf=pa[0]
            lr=pa[1]
            step=pa[2]
            print(f'start attack: {method} {pa} {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}')
            atk=tatk.CW(model,c=1e-4, lr=lr, steps=step, kappa=conf)
            succ=attack_(model,atk,images,labels,files,beg=args.beg,end=args.end,method=method,pa=pa,root=root,batch_size=3)
            # adv_images=atk(images,labels)
            # outputs=model(adv_images)
            # pre = torch.argmax(outputs,1)
            # succ = (pre != labels).sum()/adv_images.shape[0]
            # for id in range(images.shape[0]):
            #     filename=f'{id}_{method}_{pa}.png'
            #     img=adv_images.cpu().numpy()[id].transpose(1,2,0)
            #     imageio.imsave(os.path.join(root,filename), img_as_ubyte(img))
            print(f'end. success rate {succ*100:.2f}% {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}\n')
    elif method=='DeepFool':
        params=(20,30,40,50,60)
        for pa in params:
            print(f'start attack: {method} {pa} {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}')
            atk=tatk.DeepFool(model,steps=pa)
            succ=attack_(model,atk,images,labels,files,beg=args.beg,end=args.end,method=method,pa=pa,root=root,batch_size=args.batch)#3
            # adv_images=atk(images,labels)
            # outputs=model(adv_images)
            # pre = torch.argmax(outputs,1)
            # succ = (pre != labels).sum()/adv_images.shape[0]
            # for id in range(images.shape[0]):
            #     filename=f'{id}_{method}_{pa}.png'
            #     img=adv_images.cpu().numpy()[id].transpose(1,2,0)
            #     imageio.imsave(os.path.join(root,filename), img_as_ubyte(img))
            print(f'end. success rate {succ*100:.2f}% {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}\n')
    
    #时间很久
    # elif method=='SparseFool':
    #     params=(1,5,10,)
    #     for pa in params:
    #         print(f'start attack: {method} {pa} {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}')
    #         atk=tatk.SparseFool(model,steps=pa, lam=3, overshoot=0.02)
    #         succ=attack_(model,atk,images,labels,beg=args.beg,end=args.end,method=method,pa=pa,root=root)
    #         print(f'end. success rate {succ*100:.2f}% {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}\n')
    
    #black-box
    elif method=='OnePixel':
        params=((100,50),(300,80),(500,100))
        for pa in params:
            pi=pa[0]
            popsize=pa[1]
            print(f'start attack: {method} {pa} {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}')
            atk=tatk.OnePixel(model,pixels=pi,popsize=popsize,inf_batch=100,steps=40)
            succ=attack_(model,atk,images,labels,files,beg=args.beg,end=args.end,method=method,pa=pa,root=root,batch_size=args.batch)#5
            # beg,end=0,3
            # adv_images=atk(images[beg:end],labels[beg:end])
            # outputs=model(adv_images)
            # pre = torch.argmax(outputs,1)
            # succ = (pre != labels[beg:end]).sum()/(end-beg)#adv_images.shape[0]
            # for id in range(beg,end):
            #     filename=f'{id}_{method}_{pa}.png'
            #     img=adv_images.cpu().numpy()[id-beg].transpose(1,2,0)
            #     imageio.imsave(os.path.join(root,filename), img_as_ubyte(img))
            print(f'end. success rate {succ*100:.2f}% {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}\n')
    elif method=='Square':
        params=((2000,0.04,0.03),(2000,0.04,0.02),(2000,0.04,0.01),
                (2000,0.08,0.03),(2000,0.08,0.02),(2000,0.08,0.01))

        #((2000,0.4,0.03),(2000,0.6,0.03),(2000,0.8,0.03),
                # (2000,0.4,0.06),(2000,0.6,0.06),(2000,0.8,0.06),
                # (2000,0.4,0.09),(2000,0.6,0.09),(2000,0.8,0.09),
                # (2000,0.4,0.12),(2000,0.6,0.12),(2000,0.8,0.12),)
        for pa in params:
            qu=pa[0]
            p_init=pa[1]
            eps=pa[2]
            print(f'start attack: {method} {pa} {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}')
            atk = tatk.Square(model, norm='Linf', n_queries=qu, n_restarts=1, eps=eps, p_init=p_init, seed=0, verbose=False,  loss='margin', resc_schedule=True)
            succ=attack_(model,atk,images,labels,files,beg=args.beg,end=args.end,method=method,pa=pa,root=root,batch_size=args.batch)
            print(f'end. success rate {succ*100:.2f}% {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}\n')
    #Boundary foolbox
    elif method=='Boundary':
        params=[100,500,1000,1500,2000,2500,3000,3500,4000,5000,15000]#2000,2500,
        #[100,500,1000,5000,15000,20000,25000,30000]#[300,500,700,900,1200,1500,2000,3000,5000,10000,15000,20000]#[300,500,700,900,1200,1500,2000]
        for pa in params:
            steps=pa
            print(f'start attack: {method}\t step:{pa} {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}')
            atk=fatk.boundary_attack.BoundaryAttack(steps=steps)
            # succ=attack_f_single(model,atk,images,labels,files,method,pa,root)
            succ=attack_f(model,atk,images,labels,files,beg=args.beg,end=args.end,method=method,pa=pa,root=root)
            print(f'end. success rate {succ*100:.2f}% {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}')
    #NES
    #SimBA 
    elif 'SimBA' in method:
        #images [0,1] torchattacks
        from simba import SimBA
        atk=SimBA(model,'imagenet',512)
        # params=[0.03,0.06,0.1,0.2,0.3]
        params=[(0.3,100),(0.3,500),(3,100),(3,500),(30,100),(30,500)]
        
        files=files[args.beg:args.end]
        images=images[args.beg:args.end].cpu()
        labels=labels[args.beg:args.end].cpu()
        size=len(files)
        for pa in params:
            ep=pa[0]
            iters=pa[1]
            succs=[]
            print(f'start attack: {method}\t eps:{pa} {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}')
            for i in range(0,size,args.batch):
                if method=='SimBA':
                    adv,probs,succ,queries,l2_norms,linf_norms=atk.simba_batch(images[i:i+args.batch],labels[i:i+args.batch],iters,512,7,ep,pixel_attack=True,log_every=100)
                elif method=='SimBA-DCT':
                    adv,probs,succ,queries,l2_norms,linf_norms=atk.simba_batch(images[i:i+args.batch],labels[i:i+args.batch],iters,62,10,ep,0.0,order='strided',log_every=100)
                for id in range(adv.shape[0]): 
                    name=files[i:i+args.batch][id][:-4]
                    filename=f'{name}_{method}_{pa}.png'
                    img=adv.cpu().numpy()[id].transpose(1,2,0)
                    imageio.imsave(os.path.join(root,filename), img_as_ubyte(img))
                succs.append(succ[:,-1].mean())
            print(f'\r end. succs: {np.mean(succs)*100:.2f}% {time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())}')

def attack_(model,atk,images,labels,files,beg,end,method,pa,root,batch_size=10):
    files=files[beg:end]
    images=images[beg:end]
    labels=labels[beg:end]
    size=len(files)
    #batch_size=10
    succ=0
    for i in range(0,size,batch_size):
        adv_images=atk(images[i:i+batch_size],labels[i:i+batch_size])
        outputs=model(adv_images)
        pre = torch.argmax(outputs,1)
        succ += (pre != labels[i:i+batch_size]).sum()
        for id in range(outputs.shape[0]):
            name=files[i:i+batch_size][id][:-4]
            filename=f'{name}_{method}_{pa}.png'
            img=adv_images.cpu().numpy()[id].transpose(1,2,0)
            imageio.imsave(os.path.join(root,filename), img_as_ubyte(img))
    succ=succ.cpu().item()
    succ=succ/size
    # adv_images=atk(images[beg:end],labels[beg:end])
    # outputs=model(adv_images)
    # pre = torch.argmax(outputs,1)
    # succ = (pre != labels[beg:end]).sum()/(end-beg)#adv_images.shape[0]
    # for id in range(beg,end):
    #     filename=f'{id}_{method}_{pa}.png'
    #     img=adv_images.cpu().numpy()[id-beg].transpose(1,2,0)
    #     imageio.imsave(os.path.join(root,filename), img_as_ubyte(img))
    return succ

def attack_f(model,atk,images,labels,files,beg,end,method,pa,root,batch_size=10):
    files=files[beg:end]
    images=images[beg:end]
    labels=labels[beg:end]
    size=len(files)
    succ=[]
    for i in range(0,size,batch_size):
        raw_advs, clipped_advs, success=atk(model,images[i:i+batch_size],labels[i:i+batch_size],epsilons=[None])
    
        for id in range(clipped_advs[0].shape[0]): 
            name=files[i:i+batch_size][id][:-4]
            filename=f'{name}_{method}_{pa}.png'
            img=raw_advs[0].raw.cpu().numpy()[id].transpose(1,2,0)
            imageio.imsave(os.path.join(root,filename), img_as_ubyte(img))
        sus=success.float32().mean(axis=-1)
        succ.append(sus[0].item())
    return np.mean(succ)

def attack_f_single(model,atk,images,labels,files,method,pa,root):
    raw_advs, clipped_advs, success=atk(model,images,labels,epsilons=[None])
    name=files[0][:-4]
    filename=f'{name}_{method}_{pa}.png'
    img=raw_advs[0].raw.cpu().numpy()[0].transpose(1,2,0)
    imageio.imsave(os.path.join(root,filename), img_as_ubyte(img))
    sus=success.float32().mean(axis=-1)
    return sus[0].item()



def main(args):
    #load images
    model=load_target_model(type='torchattacks')
    from utils import load_images
    images_,labels_,files_=load_images(root=args.REF,model=model)
    
    if args.errors:
        lis_images=[]
        lis_labels=[]
        lis_files=[]
        errors=['2009_004709.jpg', '2011_000249.jpg']
        for i in range(len(files_)):
            if files_[i] in errors:
                print(f'find {files_[i]}')
                lis_images.append(images_[i].unsqueeze(0))
                lis_labels.append(labels_[i])
                lis_files.append(files_[i])
                if len(lis_files)==len(errors):
                    break
        images_=torch.cat(lis_images)
        labels_=torch.tensor(lis_labels)
        files_=lis_files


    images_=images_.to('cuda')
    labels_=labels_.to('cuda')
    
    #load model
    if args.tool=='foolbox':
        model=load_target_model(type=args.tool)
    print(type(model))

    if args.tool=='foolbox':
        def tu(images,labels):
            return images,labels
        images_, labels_ = ep.astensors(*tu(images_,labels_))
    attack_torchattacks(model,args,images_,labels_,files_)





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
    parser.add_argument("--method", required=True ,type=str, choices=['FGSM','BIM','PGD','CW','DeepFool','OnePixel','Square','SparseFool','Boundary','SimBA','SimBA-DCT'], help="the attack method.")
    parser.add_argument("--beg", type=int,default=0, help="the id of the first image")
    parser.add_argument("--end", type=int, default=387, help="the id of the last image.")
    parser.add_argument("--gpu", type=str, default='0', help="the id of the last image.")
    parser.add_argument("--tool", type=str, required=True,choices=['foolbox','torchattacks'], help="")
    parser.add_argument("--REF", type=str, default='/data0/lhc/dataset/IQA_AE/REF/REF_VOC', help="")
    parser.add_argument("--batch", type=int, default=10, help="")
    parser.add_argument("--errors",type=bool,default=False)
    args = parser.parse_args()
    os.environ ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
