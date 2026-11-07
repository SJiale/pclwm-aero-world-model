import scipy.io as sio

G = sio.loadmat(r"C:\pclwm\pclwm_stage3a\gru_pack2.mat")
print("gru_pack2.mat keys:")
for k in sorted(G.keys()):
    if not k.startswith("__"):
        v = G[k]
        print(f"  {k}: {getattr(v, 'shape', type(v))}")

print()
print("=== obs/act 归一化参数 ===")
for k in ["obs_mu", "obs_sd", "act_mu", "act_sd"]:
    if k in G:
        print(f"  {k} = {G[k].ravel()}")
