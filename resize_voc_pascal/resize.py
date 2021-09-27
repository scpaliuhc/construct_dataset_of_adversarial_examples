import os
from PIL import Image
from PIL import UnidentifiedImageError
files=os.listdir('/data0/lhc/dataset/VOC2012/JPEGImages')
problem_files=[]
for id,file in enumerate(files):
    print(f'\r percent {id/len(files)*100:.2f}%',end='')
    try:
        IMG=Image.open(os.path.join('/data0/lhc/dataset/VOC2012/JPEGImages',file))
        IMG=IMG.resize((512,512))
        IMG.save(os.path.join('/data0/lhc/dataset/VOC2012/JPEGImages512',file))
    except UnidentifiedImageError:
        problem_files.append(file)

for file in problem_files:
    print(file)
print(len(problem_files),len(files),)
