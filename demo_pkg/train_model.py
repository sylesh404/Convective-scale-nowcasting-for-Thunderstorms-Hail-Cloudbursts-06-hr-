
import numpy as np, torch
from nowcast_model import NowcastNet
d=np.load("synthetic_nowcasting_dataset.npz")
X=torch.tensor(d["X"][:24]); Y=torch.tensor(d["Y"][:24])
m=NowcastNet(); opt=torch.optim.Adam(m.parameters(),lr=0.004)
p=m(X); loss=torch.nn.MSELoss()(p,Y); loss.backward(); opt.step()
torch.save(m.state_dict(),"nowcast_demo.pt")
print("trained demo model; loss =",float(loss))
