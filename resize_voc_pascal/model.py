from scipy.stats.stats import mode
import torch.nn as nn
import torch
import torchvision
import torchvision.models as models
import PIL.Image as Image
img=Image.open('/data0/lhc/dataset/VOC2012/JPEGImages/2008_000703.jpg')
trans=torchvision.transforms.Compose([
            torchvision.transforms.Resize((512,512)),
            torchvision.transforms.ToTensor()])
class Normalize(nn.Module) :
            def __init__(self, mean, std) :
                super(Normalize, self).__init__()
                self.register_buffer('mean', torch.Tensor(mean))
                self.register_buffer('std', torch.Tensor(std))
            def forward(self, input):
                # Broadcasting
                mean = self.mean.reshape(1, 3, 1, 1)
                std = self.std.reshape(1, 3, 1, 1)
                return (input - mean) / std
norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = nn.Sequential(
            norm_layer,
            models.inception_v3(pretrained=True)
        )
model=model.eval()
pred=model(trans(img).unsqueeze(0))
maxk=max((1,5))
a,b=pred.topk(maxk,1)
print(a,b)
import json
class_idx=json.load(open("./imagenet_class_index.json"))
for i in range(5):
    print(class_idx[f'{b[0][i]}'][1])

