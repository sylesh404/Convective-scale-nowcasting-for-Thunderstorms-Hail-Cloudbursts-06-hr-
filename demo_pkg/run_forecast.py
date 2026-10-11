
import numpy as np, torch
from nowcast_model import NowcastNet
d=np.load("synthetic_nowcasting_dataset.npz")
m=NowcastNet(); m.load_state_dict(torch.load("nowcast_demo.pt",map_location="cpu")); m.eval()
with torch.no_grad(): p=m(torch.tensor(d["X"][0:1])).numpy()[0]
np.savez_compressed("forecast_demo.npz",input=d["X"][0],forecast=p)
print("forecast shape:",p.shape)
