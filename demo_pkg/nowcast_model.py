
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch=4, hidden=4):
        super().__init__()
        self.hidden = hidden
        self.conv = nn.Conv2d(in_ch + hidden, 4*hidden, 3, padding=1)
    def forward(self, x, h, c):
        i,f,o,g = torch.chunk(self.conv(torch.cat([x,h],1)),4,1)
        i,f,o = torch.sigmoid(i),torch.sigmoid(f),torch.sigmoid(o)
        c = f*c + i*torch.tanh(g)
        h = o*torch.tanh(c)
        return h,c

class NowcastNet(nn.Module):
    def __init__(self, future=6):
        super().__init__()
        self.future=future
        self.cell=ConvLSTMCell()
        self.head=nn.Sequential(
            nn.Conv2d(4,4,3,padding=1), nn.ReLU(),
            nn.Conv2d(4,3,1), nn.Sigmoid()
        )
    def forward(self,x):
        B,T,C,H,W=x.shape
        h=torch.zeros(B,4,H,W,device=x.device)
        c=torch.zeros_like(h)
        for t in range(T):
            h,c=self.cell(x[:,t],h,c)
        outs=[]
        for _ in range(self.future):
            y=self.head(h); outs.append(y)
            feedback=torch.cat([
                y[:,0:1],
                0.85*y[:,0:1]+0.15*y[:,1:2],
                y[:,1:2],
                0.65*y[:,0:1]+0.35*y[:,2:3]
            ],1)
            h,c=self.cell(feedback,h,c)
        return torch.stack(outs,1)
