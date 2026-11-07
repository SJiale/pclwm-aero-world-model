"""导出 GRU 权重为列主序 .mat（每层单独文件，避免转置混乱）

约定：MATLAB 端所有向量统一为行向量 (1×N)
      权重矩阵保持 PyTorch 形状 (out, in)
      Linear: y = x * W' + b   其中 x 是 (1,in), W 是 (out,in) -> W' 是 (in,out)
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from scipy.io import savemat

sys.path[:0] = [r"C:\pclwm\pclwm_data", r"C:\pclwm\pclwm_mask",
                r"C:\pclwm\pclwm_pcrssm", r"C:\pclwm\pclwm_experiments"]

from evaluate import UnifiedData, WINDOW, HORIZON, DT

DEV = "cpu"
data = UnifiedData()
C_O, C_A = data.n_obs, data.n_act


class GRUModel(nn.Module):
    def __init__(self, c_obs, c_act, hidden=128, horizon=HORIZON,
                 mlp=256, layers=2):
        super().__init__()
        self.horizon = horizon
        self.inp = nn.Sequential(
            nn.Linear(c_obs + c_act, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.cell = nn.GRUCell(hidden, hidden)
        mods = [nn.Linear(hidden, mlp), nn.SiLU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(mlp, mlp), nn.SiLU()]
        mods += [nn.Linear(mlp, c_obs)]
        self.dec = nn.Sequential(*mods)
        self.o2h = nn.Linear(c_obs + c_act, hidden)

    def forward(self, obs, act, af):
        B, T, _ = obs.shape
        H = af.shape[1]
        x = self.inp(torch.cat([obs, act], -1))
        _, h = self.gru(x)
        h = h[-1]
        o_prev = obs[:, -1]
        outs = []
        for k in range(H):
            z = self.o2h(torch.cat([o_prev, af[:, k]], -1))
            h = self.cell(z, h)
            o_hat = self.dec(h)
            outs.append(o_hat)
            o_prev = o_hat
        return torch.stack(outs, 1)


sd = torch.load(r"C:\pclwm\pclwm_experiments\checkpoints\GRU-h128.pt",
                map_location=DEV, weights_only=True)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
m = GRUModel(C_O, C_A)
m.load_state_dict(sd, strict=True)
m.eval()

# 权重（全部转成 numpy，pyTorch 的 Linear 权重是 (out,in)，直接存）
d = {}
for k, v in m.state_dict().items():
    d["w_" + k.replace(".", "_")] = v.detach().numpy()

d["obs_mu"] = np.asarray(data.obs_mu).ravel().reshape(1, -1)
d["obs_sd"] = np.asarray(data.obs_sd).ravel().reshape(1, -1)
d["act_mu"] = np.asarray(data.act_mu).ravel().reshape(1, -1)
d["act_sd"] = np.asarray(data.act_sd).ravel().reshape(1, -1)

# 测试用例
w0 = data.windows("test_id", stride=8)[0]
o = torch.from_numpy(w0["obs"][None]).float()
a = torch.from_numpy(w0["act"][None]).float()
af = torch.from_numpy(w0["act_fut"][None]).float()
with torch.no_grad():
    p = m(o, a, af)

d["tc_obs"] = w0["obs"]           # (32,5)
d["tc_act"] = w0["act"]           # (32,3)
d["tc_af"] = w0["act_fut"]        # (32,3)
d["tc_expected"] = p[0].numpy()   # (32,5)

savemat(r"C:\pclwm\pclwm_stage3a\gru_pack2.mat", d)

print("  导出的权重形状（MATLAB 视角）:")
for k in sorted(d.keys()):
    if k.startswith("w_"):
        print(f"    {k:<28} {d[k].shape}")
print(f"\n  测试用例: tc_obs{d['tc_obs'].shape} -> tc_expected{d['tc_expected'].shape}")
print(f"  期望[0] = {d['tc_expected'][0]}")
print(f"\n  -> gru_pack2.mat")
