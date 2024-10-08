import sys
sys.path.append(r"C:\Users\chenc\CodeProject\Neural-Dynamical-Operator")
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchdiffeq
from utility_old import get_batch
from utility import FNO_KSE, acf, DKL_estimator, predict_short_relay, DKL1d
import time



device = "cuda:1"
np.random.seed(0)
torch.manual_seed(0)

mpl.use("Qt5Agg")
plt.rcParams["agg.path.chunksize"] = 10000
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \boldmath"


##########################
### Data Preprocessing ###
##########################

#Simulation Setting: Lx = 22, Lt = 5000, dx=22/1024, dt = 0.025

#Data Setting: dt=0.25, dx=22/1024
u_data = torch.from_numpy(np.load(r"C:\Users\chenc\CodeProject\Neural-Dynamical-Operator\Data\KSE_data.npy")).to(torch.float32)
t = torch.arange(0, 5000, 0.25)
x = torch.arange(0, 22, 22/1024)

# Subsampling for different resolutions

# 1. dt=0.5, dx=22/1024
# u_data = u_data[::2] # (10000, 1024)
# t = t[::2]
# Ntrain = int(10000*0.8)
# Ntest = int(10000*0.2)
# u_train = u_data[:Ntrain]
# u_test = u_data[-Ntest:]
# t_train = t[:Ntrain]
# t_test = t[:Ntest]
# batch_time = 40 #20s
# epochs = 20000
# num_batch = 2

# # 2. dt=1, dx=22/512
# u_data = u_data[::4, ::2] # (5000, 512)
# t = t[::4]
# x = x[::2]
# Ntrain = int(5000*0.8)
# Ntest = int(5000*0.2)
# u_train = u_data[:Ntrain]
# u_test = u_data[-Ntest:]
# t_train = t[:Ntrain]
# t_test = t[:Ntest]
# batch_time = 20 # 20s
# epochs = 20000
# num_batch = 2

# 3. dt=2, dx=22/256
u_data = u_data[::8, ::4] # (2500, 256)
t = t[::8]
x = x[::4]
Ntrain = int(2500*0.8)
Ntest = int(2500*0.2)
u_train = u_data[:Ntrain]
u_test = u_data[-Ntest:]
t_train = t[:Ntrain]
t_test = t[:Ntest]
batch_time = 10 # 20s
epochs = 10000
num_batch = 2



######################
### Model Training ###
######################

loss_training_history = []
loss_test_history = []
test_freq = 100

model = FNO_KSE(24, 64).to(device)
optimizer = optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
for ep in range(1, epochs+1):
    start_time = time.time()
    batch_u0, batch_t, batch_u = get_batch(t=t_train, u=u_train, num_batch=num_batch, batch_time=batch_time)
    batch_u0, batch_t, batch_u = batch_u0.to(device), batch_t.to(device), batch_u.to(device)
    optimizer.zero_grad()
    out = torchdiffeq.odeint(model, batch_u0, batch_t, method="rk4", options={"step_size":0.1})
    loss = F.mse_loss(batch_u, out)
    # print(torch.cuda.memory_allocated(device=device) / 1024**2)
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_training_history.append(loss.item())
    end_time = time.time()
    print(ep, "Time: ", round(end_time-start_time, 4), "Loss: ",  round(loss.item(), 4) )

    # if ep % test_freq == 0:
    #     # Test Loss
    #     with torch.no_grad():
    #         state_test_error = integrate_batch(t=t_test.to(device), u=u_test.to(device), model=model, batch_time=batch_time)[0]
    #         loss_test_history.append(state_test_error)
    #     print(ep, "Test Loss: ", state_test_error)

# torch.save(model.state_dict(), r"C:\Users\chenc\CodeProject\Neural-Dynamical-Operator\Model\KSE_Models\KS_model_v3_ep10000(modes12_width32).pt")
# np.save(r"C:\Users\chenc\CodeProject\Neural-Dynamical-Operator\Model\KSE_Models\KS_model_v3_ep10000(modes12_width32)_loss_training_history.npy", loss_training_history)





#########################
### Model Application ###
#########################

u_data = torch.from_numpy(np.load(r"C:\Users\chenc\CodeProject\Neural-Dynamical-Operator\Data\KSE_data.npy")).to(torch.float32)
t = torch.linspace(0, 5000, 20001)[:-1]
x = torch.from_numpy(np.linspace(0, 22, 1024, endpoint=False))
Ntrain = int(u_data.shape[0]*0.8)
Ntest = int(u_data.shape[0]*0.2)
u_train = u_data[:Ntrain]
u_test = u_data[-Ntest:]
t_train = t[:Ntrain]
t_test = t[:Ntest]


# Pre-trained Model
model1 = FNO_KSE(24, 64)
model1.load_state_dict(torch.load(r"C:\Users\chenc\CodeProject\Neural-Dynamical-Operator\Model\KSE_Models\KS_model_v1_ep20000.pt"))
model2 = FNO_KSE(24, 64)
model2.load_state_dict(torch.load(r"C:\Users\chenc\CodeProject\Neural-Dynamical-Operator\Model\KSE_Models\KS_model_v2_ep20000.pt"))
model3 = FNO_KSE(12, 32)
model3.load_state_dict(torch.load(r"C:\Users\chenc\CodeProject\Neural-Dynamical-Operator\Model\KSE_Models\KS_model_v3_ep20000.pt"))

# Different Resolution for Test Data
u_test_res1 = u_test[::2]
u_test_res2 = u_test[::4, ::2]
u_test_res3 = u_test[::8, ::4]
t_test_res1 = t_test[::2]
t_test_res2 = t_test[::4]
t_test_res3 = t_test[::8]

#Short-term Relay Predictions & Test Error (Each Resolution)
u_test_pred_short1_res1 = predict_short_relay(u_test_res1, t_test_res1, model1, 40)
u_test_pred_short2_res2 = predict_short_relay(u_test_res2, t_test_res2, model2, 20)
u_test_pred_short3_res3 = predict_short_relay(u_test_res3, t_test_res3, model3, 10)

F.mse_loss(u_test_res1, u_test_pred_short1_res1).item()
F.mse_loss(u_test_res2, u_test_pred_short2_res2).item()
F.mse_loss(u_test_res3, u_test_pred_short3_res3).item()
torch.mean( torch.norm(u_test_res1 - u_test_pred_short1_res1, 2, 1) / torch.norm(u_test_res1, 2, 1) ).item()
torch.mean( torch.norm(u_test_res2 - u_test_pred_short2_res2, 2, 1) / torch.norm(u_test_res2, 2, 1) ).item()
torch.mean( torch.norm(u_test_res3 - u_test_pred_short3_res3, 2, 1) / torch.norm(u_test_res3, 2, 1) ).item()



#Short-term Predictions & Test Error (Same Resolution)
u_test_pred_short1 = predict_short_relay(u_test, t_test, model1, 80)
u_test_pred_short2 = predict_short_relay(u_test, t_test, model2, 80)
u_test_pred_short3 = predict_short_relay(u_test, t_test, model3, 80)

F.mse_loss(u_test, u_test_pred_short1).item()
F.mse_loss(u_test, u_test_pred_short2).item()
F.mse_loss(u_test, u_test_pred_short3).item()
torch.mean( torch.norm(u_test - u_test_pred_short1, 2, 1) / torch.norm(u_test, 2, 1) ).item()
torch.mean( torch.norm(u_test - u_test_pred_short2, 2, 1) / torch.norm(u_test, 2, 1) ).item()
torch.mean( torch.norm(u_test - u_test_pred_short3, 2, 1) / torch.norm(u_test, 2, 1) ).item()







# Long-time Prediction of Test Data (Each Resolution)
model1.to(device)
model2.to(device)
model3.to(device)
# with torch.no_grad():
#     u_test_pred1_res1 = torchdiffeq.odeint(model1, u_test_res1[[0]].to(device), t_test_res1.to(device))[:,0,:].cpu()
#     u_test_pred2_res2 = torchdiffeq.odeint(model2, u_test_res2[[0]].to(device), t_test_res2.to(device))[:,0,:].cpu()
#     u_test_pred3_res3 = torchdiffeq.odeint(model3, u_test_res3[[0]].to(device), t_test_res3.to(device))[:,0,:].cpu()

u_test_pred1_res1 = torch.from_numpy(np.load(r"C:\Users\chenc\Downloads\KSE_longSimu\u_test_pred1_res1(XD).npy"))
u_test_pred2_res2 = torch.from_numpy(np.load(r"C:\Users\chenc\Downloads\KSE_longSimu\u_test_pred2_res2(XD).npy"))
u_test_pred3_res3 = torch.from_numpy(np.load(r"C:\Users\chenc\Downloads\KSE_longSimu\u_test_pred3_res3_12_32.npy"))

# Forward DKL
DKL_res1_forward = DKL1d(u_test_res1.reshape(-1), u_test_pred1_res1.reshape(-1))
DKL_res2_forward = DKL1d(u_test_res2.reshape(-1), u_test_pred2_res2.reshape(-1))
DKL_res3_forward = DKL1d(u_test_res3.reshape(-1), u_test_pred3_res3.reshape(-1))
# Reverse DKL
DKL_res1_reverse = DKL1d(u_test_pred1_res1.reshape(-1), u_test_res1.reshape(-1))
DKL_res2_reverse = DKL1d(u_test_pred2_res2.reshape(-1), u_test_res2.reshape(-1))
DKL_res3_reverse = DKL1d(u_test_pred3_res3.reshape(-1), u_test_res3.reshape(-1))

# Long-time Prediction of Test Data (Same Resolution)
# with torch.no_grad():
#     u_test_pred1 = torchdiffeq.odeint(model1, u_test[[0]].to(device), t_test.to(device))[:,0,:].cpu()
#     u_test_pred2 = torchdiffeq.odeint(model2, u_test[[0]].to(device), t_test.to(device))[:,0,:].cpu()
#     u_test_pred3 = torchdiffeq.odeint(model3, u_test[[0]].to(device), t_test.to(device))[:,0,:].cpu()

u_test_pred1 = torch.from_numpy(np.load(r"C:\Users\chenc\Downloads\KSE_longSimu\u_test_pred1(XD).npy"))
u_test_pred2 = torch.from_numpy(np.load(r"C:\Users\chenc\Downloads\KSE_longSimu\u_test_pred2(XD).npy"))
u_test_pred3 = torch.from_numpy(np.load(r"C:\Users\chenc\Downloads\KSE_longSimu\u_test_pred3_12_32.npy"))

# Forward DKL
DKL1_forward = DKL1d(u_test.reshape(-1), u_test_pred1.reshape(-1))
DKL2_forward = DKL1d(u_test.reshape(-1), u_test_pred2.reshape(-1))
DKL3_forward = DKL1d(u_test.reshape(-1), u_test_pred3.reshape(-1))
# Reverse DKL
DKL1_reverse = DKL1d(u_test_pred1.reshape(-1), u_test.reshape(-1))
DKL2_reverse = DKL1d(u_test_pred2.reshape(-1), u_test.reshape(-1))
DKL3_reverse = DKL1d(u_test_pred3.reshape(-1), u_test.reshape(-1))



# Pcolor
vmin = min( torch.min(u_test[:2000]), torch.min(u_test_pred1[:2000]), torch.min(u_test_pred2[:2000]), torch.min(u_test_pred3[:2000]) )
vmax = max( torch.max(u_test[:2000]), torch.max(u_test_pred1[:2000]), torch.max(u_test_pred2[:2000]), torch.max(u_test_pred3[:2000]) )
fig = plt.figure(constrained_layout=True)
axs = fig.subplots(2, 2)
fig.set_size_inches(30, 20)
c = axs[0, 0].pcolor(4000+t_test[:2000], x, u_test[:2000].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[0,0].set_ylabel(r"\unboldmath$x$", fontsize=50, rotation=0)
axs[0,0].set_xticks(range(4000, 4500+100, 100))
axs[0,0].set_yticks(range(2, 23, 4))
axs[0,0].set_title(r"\textbf{True}", fontsize=40)
axs[0,0].tick_params(labelsize=40, length=15, width=3)
axs[0,1].pcolor(4000+t_test[:2000], x, u_test_pred1[:2000].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[0,1].set_xticks(range(4000, 4500+100, 100))
axs[0,1].set_yticks(range(2, 23, 4))
axs[0,1].set_title(r"\textbf{Model 1}", fontsize=40)
axs[0,1].tick_params(labelsize=40, length=15, width=3)
axs[1,0].pcolor(4000+t_test[:2000], x, u_test_pred2[:2000].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[1,0].set_xlabel(r"\unboldmath$t$", fontsize=50)
axs[1,0].set_ylabel(r"\unboldmath$x$", fontsize=50, rotation=0)
axs[1,0].set_xticks(range(4000, 4500+100, 100))
axs[1,0].set_yticks(range(2, 23, 4))
axs[1,0].set_title(r"\textbf{Model 2}", fontsize=40)
axs[1,0].tick_params(labelsize=40, length=15, width=3)
axs[1,1].pcolor(4000+t_test[:2000], x, u_test_pred3[:2000].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[1,1].set_xlabel(r"\unboldmath$t$", fontsize=50)
axs[1,1].set_xticks(range(4000, 4500+100, 100))
axs[1,1].set_yticks(range(2, 23, 4))
axs[1,1].set_title(r"\textbf{Model 3}", fontsize=40)
axs[1,1].tick_params(labelsize=40, length=15, width=3)
cbar = fig.colorbar(c, ax=axs)
cbar.ax.tick_params(labelsize=40, length=15, width=3)



# Profile1 (t=4000, 4005, 4010, 4020s)
# Profile2 (t=4500, 4505, 4510, 4520s)
# Profile2 (t=4900, 4905, 4910, 4920s)
model1.cpu()
model2.cpu()
model3.cpu()
with torch.no_grad():
    u_test_pred1_4000 = torchdiffeq.odeint(model1, u_test[[0]], t_test[:80])[:,0,:]
    u_test_pred2_4000 = torchdiffeq.odeint(model2, u_test[[0]], t_test[:80])[:,0,:]
    u_test_pred3_4000 = torchdiffeq.odeint(model3, u_test[[0]], t_test[:80])[:,0,:]
    u_test_pred1_4500 = torchdiffeq.odeint(model1, u_test[[2000]], t_test[:80])[:,0,:]
    u_test_pred2_4500 = torchdiffeq.odeint(model2, u_test[[2000]], t_test[:80])[:,0,:]
    u_test_pred3_4500 = torchdiffeq.odeint(model3, u_test[[2000]], t_test[:80])[:,0,:]
    u_test_pred1_4900 = torchdiffeq.odeint(model1, u_test[[3600]], t_test[:80])[:,0,:]
    u_test_pred2_4900 = torchdiffeq.odeint(model2, u_test[[3600]], t_test[:80])[:,0,:]
    u_test_pred3_4900 = torchdiffeq.odeint(model3, u_test[[3600]], t_test[:80])[:,0,:]
# Profile Visualization
fig = plt.figure(figsize=(30, 18))
axs = fig.subplots(3, 4, sharex=True, sharey=True)
axs[0,0].plot(x, u_test[0], linewidth=5)
axs[0,0].set_ylabel(r"\unboldmath$u$", fontsize=50, rotation=0)
axs[0,0].set_title(r"\unboldmath$t = 4000$", fontsize=50)
axs[0,0].tick_params(labelsize=40, width=3, length=15)
axs[0,1].plot(x, u_test[20], linewidth=5, label=r"\textbf{True}")
axs[0,1].plot(x, u_test_pred1_4000[20], linewidth=5, linestyle="dashed", label=r"\textbf{Model 1}")
axs[0,1].plot(x, u_test_pred2_4000[20], linewidth=5, linestyle="dashed", label=r"\textbf{Model 2}")
axs[0,1].plot(x, u_test_pred3_4000[20], linewidth=5, linestyle="dashed", label=r"\textbf{Model 3}")
axs[0,1].set_title(r"\unboldmath$t = 4005$", fontsize=50)
axs[0,1].tick_params(labelsize=40, width=3, length=15)
axs[0,2].plot(x, u_test[40], linewidth=5)
axs[0,2].plot(x, u_test_pred1_4000[40], linewidth=5, linestyle="dashed")
axs[0,2].plot(x, u_test_pred2_4000[40], linewidth=5, linestyle="dashed")
axs[0,2].plot(x, u_test_pred3_4000[40], linewidth=5, linestyle="dashed")
axs[0,2].set_title(r"\unboldmath$t = 4010$", fontsize=50)
axs[0,2].tick_params(labelsize=40, width=3, length=15)
axs[0,3].plot(x, u_test[80-1], linewidth=5)
axs[0,3].plot(x, u_test_pred1_4000[-1], linewidth=5, linestyle="dashed")
axs[0,3].plot(x, u_test_pred2_4000[-1], linewidth=5, linestyle="dashed")
axs[0,3].plot(x, u_test_pred3_4000[-1], linewidth=5, linestyle="dashed")
axs[0,3].set_title(r"\unboldmath$t = 4020$", fontsize=50)
axs[0,3].tick_params(labelsize=40, width=3, length=15)
axs[1,0].plot(x, u_test[2000], linewidth=5)
axs[1,0].set_ylabel(r"\unboldmath$u$", fontsize=50, rotation=0)
axs[1,0].set_title(r"\unboldmath$t = 4500$", fontsize=50)
axs[1,0].tick_params(labelsize=40, width=3, length=15)
axs[1,1].plot(x, u_test[20+2000], linewidth=5)
axs[1,1].plot(x, u_test_pred1_4500[20], linewidth=5, linestyle="dashed")
axs[1,1].plot(x, u_test_pred2_4500[20], linewidth=5, linestyle="dashed")
axs[1,1].plot(x, u_test_pred3_4500[20], linewidth=5, linestyle="dashed")
axs[1,1].set_title(r"\unboldmath$t = 4505$", fontsize=50)
axs[1,1].tick_params(labelsize=40, width=3, length=15)
axs[1,2].plot(x, u_test[40+2000], linewidth=5)
axs[1,2].plot(x, u_test_pred1_4500[40], linewidth=5, linestyle="dashed")
axs[1,2].plot(x, u_test_pred2_4500[40], linewidth=5, linestyle="dashed")
axs[1,2].plot(x, u_test_pred3_4500[40], linewidth=5, linestyle="dashed")
axs[1,2].set_title(r"\unboldmath$t = 4510$", fontsize=50)
axs[1,2].tick_params(labelsize=40, width=3, length=15)
axs[1,3].plot(x, u_test[80+2000-1], linewidth=5)
axs[1,3].plot(x, u_test_pred1_4500[-1], linewidth=5, linestyle="dashed")
axs[1,3].plot(x, u_test_pred2_4500[-1], linewidth=5, linestyle="dashed")
axs[1,3].plot(x, u_test_pred3_4500[-1], linewidth=5, linestyle="dashed")
axs[1,3].set_title(r"\unboldmath$t = 4520$", fontsize=50)
axs[1,3].tick_params(labelsize=40, width=3, length=15)
axs[2,0].plot(x, u_test[3600], linewidth=5)
axs[2,0].set_xlabel(r"\unboldmath$x$", fontsize=50)
axs[2,0].set_ylabel(r"\unboldmath$u$", fontsize=50, rotation=0)
axs[2,0].set_title(r"\unboldmath$t = 4900$", fontsize=50)
axs[2,0].tick_params(labelsize=40, width=3, length=15)
axs[2,1].plot(x, u_test[20+3600], linewidth=5)
axs[2,1].plot(x, u_test_pred1_4900[20], linewidth=5, linestyle="dashed")
axs[2,1].plot(x, u_test_pred2_4900[20], linewidth=5, linestyle="dashed")
axs[2,1].plot(x, u_test_pred3_4900[20], linewidth=5, linestyle="dashed")
axs[2,1].set_xlabel(r"\unboldmath$x$", fontsize=50)
axs[2,1].set_title(r"\unboldmath$t = 4905$", fontsize=50)
axs[2,1].tick_params(labelsize=40, width=3, length=15)
axs[2,2].plot(x, u_test[40+3600], linewidth=5)
axs[2,2].plot(x, u_test_pred1_4900[40], linewidth=5, linestyle="dashed")
axs[2,2].plot(x, u_test_pred2_4900[40], linewidth=5, linestyle="dashed")
axs[2,2].plot(x, u_test_pred3_4900[40], linewidth=5, linestyle="dashed")
axs[2,2].set_xlabel(r"\unboldmath$x$", fontsize=50)
axs[2,2].set_title(r"\unboldmath$t = 4910$", fontsize=50)
axs[2,2].tick_params(labelsize=40, width=3, length=15)
axs[2,3].plot(x, u_test[80+3600-1], linewidth=5)
axs[2,3].plot(x, u_test_pred1_4900[-1], linewidth=5, linestyle="dashed")
axs[2,3].plot(x, u_test_pred2_4900[-1], linewidth=5, linestyle="dashed")
axs[2,3].plot(x, u_test_pred3_4900[-1], linewidth=5, linestyle="dashed")
axs[2,3].set_xlabel(r"\unboldmath$x$", fontsize=50)
axs[2,3].set_title(r"\unboldmath$t = 4920$", fontsize=50)
axs[2,3].tick_params(labelsize=40, width=3, length=15)
for ax in axs.flatten():
    for spine in ax.spines.values():
        spine.set_linewidth(3)
lege = fig.legend(fontsize=40, loc="upper center", ncol=4, fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(3)
fig.tight_layout()
fig.subplots_adjust(top=0.87)


## Long-Term Statistics
u_long_pred_smooth1 = torch.fft.irfft(torch.fft.rfft(u_test_pred1)[:, :8], n=u_test_pred1.shape[1])
u_long_pred_smooth2 = torch.fft.irfft(torch.fft.rfft(u_test_pred2)[:, :8], n=u_test_pred2.shape[1])
u_long_pred_smooth3 = torch.fft.irfft(torch.fft.rfft(u_test_pred3)[:, :8], n=u_test_pred3.shape[1])

# subsampling(1024 --> 256) to reduce computational cost
u_test_sub = u_test[:, ::4]
u_long_pred_smooth1 = u_long_pred_smooth1[:, ::4]
u_long_pred_smooth2 = u_long_pred_smooth2[:, ::4]
u_long_pred_smooth3 = u_long_pred_smooth3[:, ::4]

# Numerical 1st & 2nd Derivative
dx = 22/256
u_x = (u_test_sub[:, 2:] - u_test_sub[:, :-2]) / (2 * dx)
u_xx = (u_test_sub[:, 2:] + u_test_sub[:, :-2] - 2 * u_test_sub[:, 1:-1]) / (dx ** 2)
u_x1 = (u_long_pred_smooth1[:, 2:] - u_long_pred_smooth1[:, :-2]) / (2*dx)
u_xx1 = (u_long_pred_smooth1[:, 2:] + u_long_pred_smooth1[:, :-2] - 2*u_long_pred_smooth1[:, 1:-1]) / (dx**2)
u_x2 = (u_long_pred_smooth2[:, 2:] - u_long_pred_smooth2[:, :-2]) / (2*dx)
u_xx2 = (u_long_pred_smooth2[:, 2:] + u_long_pred_smooth2[:, :-2] - 2*u_long_pred_smooth2[:, 1:-1]) / (dx**2)
u_x3 = (u_long_pred_smooth3[:, 2:] - u_long_pred_smooth3[:, :-2]) / (2*dx)
u_xx3 = (u_long_pred_smooth3[:, 2:] + u_long_pred_smooth3[:, :-2] - 2*u_long_pred_smooth3[:, 1:-1]) / (dx**2)

# Joint Data
u_jointPDF = np.stack( [u_x.reshape(-1), u_xx.reshape(-1)] ).T
u_jointPDF1 = np.stack( [u_x1.reshape(-1), u_xx1.reshape(-1)] ).T
u_jointPDF2 = np.stack( [u_x2.reshape(-1), u_xx2.reshape(-1)] ).T
u_jointPDF3 = np.stack( [u_x3.reshape(-1), u_xx3.reshape(-1)] ).T


# Visualization of Marginal PDF
fig = plt.figure(figsize=(20, 8))
ax = fig.subplots(1, 3)
sns.kdeplot(u_test.reshape(-1), ax=ax[0], label=r"\textbf{True}", linewidth=5)
sns.kdeplot(u_test_pred1.reshape(-1), ax=ax[0], label=r"\textbf{Model 1}", linewidth=5)
sns.kdeplot(u_test_pred2.reshape(-1), ax=ax[0], label=r"\textbf{Model 2}", linewidth=5)
sns.kdeplot(u_test_pred3.reshape(-1), ax=ax[0], label=r"\textbf{Model 3}", linewidth=5)
ax[0].set_xlabel(r"\unboldmath$u$", fontsize=35)
ax[0].set_ylabel("")
ax[0].set_title(r"\textbf{PDF of} \unboldmath$u$", fontsize=30, pad=10)
ax[0].tick_params(labelsize=30, length=15, width=3)
sns.kdeplot(u_jointPDF[:, 0], ax=ax[1], linewidth=5)
sns.kdeplot(u_jointPDF1[:, 0], ax=ax[1], linewidth=5)
sns.kdeplot(u_jointPDF2[:, 0], ax=ax[1], linewidth=5)
sns.kdeplot(u_jointPDF3[:, 0], ax=ax[1], linewidth=5)
ax[1].set_xlabel(r"\unboldmath$u_x$", fontsize=35)
ax[1].set_ylabel("")
ax[1].set_xlim([-2.8, 2.5])
ax[1].set_title(r"\textbf{PDF of} \unboldmath$u_x$", fontsize=30, pad=10)
ax[1].tick_params(labelsize=30, length=15, width=3)
sns.kdeplot(u_jointPDF[:, 1], ax=ax[2], linewidth=5)
sns.kdeplot(u_jointPDF1[:, 1], ax=ax[2], linewidth=5)
sns.kdeplot(u_jointPDF2[:, 1], ax=ax[2], linewidth=5)
sns.kdeplot(u_jointPDF3[:, 1], ax=ax[2], linewidth=5)
ax[2].set_xlabel(r"\unboldmath$u_{xx}$", fontsize=35)
ax[2].set_ylabel("")
ax[2].set_xlim([-3, 3])
ax[2].set_title(r"\textbf{PDF of} \unboldmath$u_{xx}$", fontsize=30, pad=10)
ax[2].tick_params(labelsize=30, length=15, width=3)
for a in ax.flatten():
    for spine in a.spines.values():
        spine.set_linewidth(3)
lege = fig.legend(fontsize=30, loc="upper center", ncol=4, fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(3)
fig.tight_layout()
fig.subplots_adjust(top=0.77)

# Visualization of Spatial & Temporal ACF
acf_accumulator = 0
acf_accumulator1 = 0
acf_accumulator2 = 0
acf_accumulator3 = 0
for j in range(u_test_sub.shape[1]):
    acf_accumulator += acf(u_test_sub[:, j])[1]
    acf_accumulator1 += acf(u_long_pred_smooth1[:, j])[1]
    acf_accumulator2 += acf(u_long_pred_smooth2[:, j])[1]
    acf_accumulator3 += acf(u_long_pred_smooth3[:, j])[1]
u_acf = acf_accumulator / u_test_sub.shape[1]
u_acf1 = acf_accumulator1/u_long_pred_smooth1.shape[1]
u_acf2 = acf_accumulator2/u_long_pred_smooth2.shape[1]
u_acf3 = acf_accumulator3/u_long_pred_smooth3.shape[1]
sacf_accumulator = 0
sacf_accumulator1 = 0
sacf_accumulator2 = 0
sacf_accumulator3 = 0
for i in range(u_test_sub.shape[0]):
    sacf_accumulator += acf(np.tile(u_test_sub[i], 2), 255)[1]
    sacf_accumulator1 += acf(np.tile(u_long_pred_smooth1[i],2), 255)[1]
    sacf_accumulator2 += acf(np.tile(u_long_pred_smooth2[i],2), 255)[1]
    sacf_accumulator3 += acf(np.tile(u_long_pred_smooth3[i],2), 255)[1]
u_sacf = sacf_accumulator / u_test_sub.shape[0]
u_sacf1 = sacf_accumulator1/u_long_pred_smooth1.shape[0]
u_sacf2 = sacf_accumulator2/u_long_pred_smooth2.shape[0]
u_sacf3 = sacf_accumulator3/u_long_pred_smooth3.shape[0]
lags = np.linspace(0, 50, 201)
slags = np.linspace(0, 255 * 22/256, 256)


fig = plt.figure(layout="constrained", figsize=(30, 10))
ax = fig.subplots(1, 2)
ax[0].plot(lags, u_acf, linewidth=6, label=r"\textbf{True}")
ax[0].plot(lags, u_acf1, linewidth=6, label=r"\textbf{Model 1}")
ax[0].plot(lags, u_acf2, linewidth=6, label=r"\textbf{Model 2}")
ax[0].plot(lags, u_acf3, linewidth=6, label=r"\textbf{Model 3}")
ax[0].set_xlabel(r"\unboldmath$t$ \textbf{lags}", fontsize=60)
ax[0].set_xticks(range(0,51,10))
ax[0].set_title(r"\textbf{Temporal ACF of} \unboldmath$u$", fontsize=60, pad=10)
ax[0].tick_params(labelsize=50, length=18, width=5)
ax[1].plot(slags, u_sacf, linewidth=6)
ax[1].plot(slags, u_sacf1, linewidth=6)
ax[1].plot(slags, u_sacf2, linewidth=6)
ax[1].plot(slags, u_sacf3, linewidth=6)
ax[1].set_xlabel(r"\unboldmath$x$ \textbf{lags}", fontsize=60)
ax[1].set_xticks(range(0,26,5))
ax[1].set_title(r"\textbf{Spatial ACF of} \unboldmath$u$", fontsize=60, pad=10)
ax[1].tick_params(labelsize=50, length=18, width=5)
for a in ax.flatten():
    for spine in a.spines.values():
        spine.set_linewidth(5)


# Visualization of Joint PDF
def bold_formatter(x, pos):
    return r"$\mathbf{{{:.2f}}}$".format(x)
fig = plt.figure(figsize=(30, 30), layout="constrained")
ax = fig.subplots(2, 2)
sns.kdeplot(x=u_jointPDF[:, 0], y=u_jointPDF[:, 1], fill=True, thresh=0.0, levels=50, cmap="flare", ax=ax[0,0], cbar=True)
ax[0,0].set_xlabel(r"\unboldmath$u_x$", fontsize=40)
ax[0,0].set_ylabel(r"\unboldmath$u_{xx}$", fontsize=40)
ax[0,0].set_title(r"\textbf{True}", fontsize=40)
ax[0,0].set_xlim([-2, 2])
ax[0,0].set_ylim([-2, 2])
ax[0,0].set_xticks(range(-2,3))
ax[0,0].set_yticks(range(-2,3))
ax[0,0].tick_params(labelsize=40, length=15, width=3)
cbar = fig.axes[-1]
cbar.tick_params(labelsize=40, length=15, width=3)
cbar.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(bold_formatter))
sns.kdeplot(x=u_jointPDF1[:, 0], y=u_jointPDF1[:, 1], fill=True, thresh=0.0, levels=50, cmap="flare", ax=ax[0,1], cbar=True)
ax[0,1].set_xlabel(r"\unboldmath$u_x$", fontsize=40)
ax[0,1].set_ylabel(r"\unboldmath$u_{xx}$", fontsize=40)
ax[0,1].set_title(r"\textbf{Model 1}", fontsize=40)
ax[0,1].set_xlim([-2, 2])
ax[0,1].set_ylim([-2, 2])
ax[0,1].set_xticks(range(-2,3))
ax[0,1].set_yticks(range(-2,3))
ax[0,1].tick_params(labelsize=40, length=15, width=3)
cbar = fig.axes[-1]
cbar.tick_params(labelsize=40, length=15, width=3)
cbar.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(bold_formatter))
sns.kdeplot(x=u_jointPDF2[:, 0], y=u_jointPDF2[:, 1], fill=True, thresh=0.0, levels=50, cmap="flare", ax=ax[1,0], cbar=True)
ax[1,0].set_xlabel(r"\unboldmath$u_x$", fontsize=40)
ax[1,0].set_ylabel(r"\unboldmath$u_{xx}$", fontsize=40)
ax[1,0].set_title(r"\textbf{Model 2}", fontsize=40)
ax[1,0].set_xlim([-2, 2])
ax[1,0].set_ylim([-2, 2])
ax[1,0].set_xticks(range(-2,3))
ax[1,0].set_yticks(range(-2,3))
ax[1,0].tick_params(labelsize=40, length=15, width=3)
cbar = fig.axes[-1]
cbar.tick_params(labelsize=40, length=15, width=3)
cbar.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(bold_formatter))
sns.kdeplot(x=u_jointPDF3[:, 0], y=u_jointPDF3[:, 1], fill=True, thresh=0.0, levels=50, cmap="flare", ax=ax[1,1], cbar=True)
ax[1,1].set_xlabel(r"\unboldmath$u_x$", fontsize=40)
ax[1,1].set_ylabel(r"\unboldmath$u_{xx}$", fontsize=40)
ax[1,1].set_title(r"\textbf{Model 3}", fontsize=40)
ax[1,1].set_xlim([-2, 2])
ax[1,1].set_ylim([-2, 2])
ax[1,1].set_xticks(range(-2,3))
ax[1,1].set_yticks(range(-2,3))
ax[1,1].tick_params(labelsize=40, length=15, width=3)
cbar = fig.axes[-1]
cbar.tick_params(labelsize=40, length=15, width=3)
cbar.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(bold_formatter))





# Visualization of KL-Divergence
# u1_DKL = DKL1d(u_test_sub.reshape(-1, 1), u_long_pred_smooth1.reshape(-1, 1))
# u2_DKL = DKL1d(u_test_sub.reshape(-1, 1), u_long_pred_smooth2.reshape(-1, 1))
# u3_DKL = DKL1d(u_test_sub.reshape(-1, 1), u_long_pred_smooth3.reshape(-1, 1))

ux1_DKL_forward = DKL1d(u_jointPDF[:, 0], u_jointPDF1[:, 0])
ux2_DKL_forward = DKL1d(u_jointPDF[:, 0], u_jointPDF2[:, 0])
ux3_DKL_forward = DKL1d(u_jointPDF[:, 0], u_jointPDF3[:, 0])
uxx1_DKL_forward = DKL1d(u_jointPDF[:, 1], u_jointPDF1[:, 1])
uxx2_DKL_forward = DKL1d(u_jointPDF[:, 1], u_jointPDF2[:, 1])
uxx3_DKL_forward = DKL1d(u_jointPDF[:, 1], u_jointPDF3[:, 1])
u_joint1_DKL_forward = DKL_estimator(u_jointPDF, u_jointPDF1)
u_joint2_DKL_forward = DKL_estimator(u_jointPDF, u_jointPDF2)
u_joint3_DKL_forward = DKL_estimator(u_jointPDF, u_jointPDF3)


ux1_DKL_reverse = DKL1d(u_jointPDF1[:, 0], u_jointPDF[:, 0])
ux2_DKL_reverse = DKL1d(u_jointPDF2[:, 0], u_jointPDF[:, 0])
ux3_DKL_reverse = DKL1d(u_jointPDF3[:, 0], u_jointPDF[:, 0])
uxx1_DKL_reverse = DKL1d(u_jointPDF1[:, 1], u_jointPDF[:, 1])
uxx2_DKL_reverse = DKL1d(u_jointPDF2[:, 1], u_jointPDF[:, 1])
uxx3_DKL_reverse = DKL1d(u_jointPDF3[:, 1], u_jointPDF[:, 1])
u_joint1_DKL_reverse = DKL_estimator(u_jointPDF1, u_jointPDF)
u_joint2_DKL_reverse = DKL_estimator(u_jointPDF2, u_jointPDF)
u_joint3_DKL_reverse = DKL_estimator(u_jointPDF3, u_jointPDF)



# Forward DKL
fig = plt.figure(figsize=(25, 10))
axs = fig.subplots(1, 2)
axs[0].scatter(np.arange(1, 7, 2), [DKL1_forward, DKL2_forward, DKL3_forward], s=400, marker="o", label=r"\unboldmath$u$")
axs[0].scatter(np.arange(1, 7, 2), [ux1_DKL_forward, ux2_DKL_forward, ux3_DKL_forward], s=400, marker="^", label=r"\unboldmath$u_x$")
axs[0].scatter(np.arange(1, 7, 2), [uxx1_DKL_forward, uxx2_DKL_forward, uxx3_DKL_forward], s=400, marker="*", label=r"\unboldmath$u_{xx}$")
axs[0].scatter(np.arange(1, 7, 2), [u_joint1_DKL_forward, u_joint2_DKL_forward, u_joint3_DKL_forward], s=400, marker="X", label=r"\unboldmath$(u_x, u_{xx})$")
axs[0].plot(np.arange(1, 7, 2), [DKL1_forward, DKL2_forward, DKL3_forward], marker="o", markersize=25, linewidth=3)
axs[0].plot(np.arange(1, 7, 2), [ux1_DKL_forward, ux2_DKL_forward, ux3_DKL_forward], marker="^", markersize=25)
axs[0].plot(np.arange(1, 7, 2), [uxx1_DKL_forward, uxx2_DKL_forward, uxx3_DKL_forward], marker="*", markersize=25)
axs[0].plot(np.arange(1, 7, 2), [u_joint1_DKL_forward, u_joint2_DKL_forward, u_joint3_DKL_forward], marker="X", markersize=25)
axs[0].set_xlim([0, 6])
axs[0].set_ylabel(r"\textbf{Forward} \unboldmath$D_{\textrm{KL}}$", fontsize=35, labelpad=15)
axs[0].tick_params(labelsize=30, length=15, width=3)
axs[0].set_xticks([1,3,5], [r"\textbf{Model 1}", r"\textbf{Model 2}", r"\textbf{Model 3}"])
axs[1].scatter(np.arange(1, 7, 2), [DKL1_reverse, DKL2_reverse, DKL3_reverse], s=400, marker="o")
axs[1].scatter(np.arange(1, 7, 2), [ux1_DKL_reverse, ux2_DKL_reverse, ux3_DKL_reverse], s=400, marker="^")
axs[1].scatter(np.arange(1, 7, 2), [uxx1_DKL_reverse, uxx2_DKL_reverse, uxx3_DKL_reverse], s=400, marker="*")
axs[1].scatter(np.arange(1, 7, 2), [u_joint1_DKL_reverse, u_joint2_DKL_reverse, u_joint3_DKL_reverse], s=400, marker="X")
axs[1].plot(np.arange(1, 7, 2), [DKL1_reverse, DKL2_reverse, DKL3_reverse], marker="o", markersize=25, linewidth=3)
axs[1].plot(np.arange(1, 7, 2), [ux1_DKL_reverse, ux2_DKL_reverse, ux3_DKL_reverse], marker="^", markersize=25)
axs[1].plot(np.arange(1, 7, 2), [uxx1_DKL_reverse, uxx2_DKL_reverse, uxx3_DKL_reverse], marker="*", markersize=25)
axs[1].plot(np.arange(1, 7, 2), [u_joint1_DKL_reverse, u_joint2_DKL_reverse, u_joint3_DKL_reverse], marker="X", markersize=25)
axs[1].set_xlim([0, 6])
axs[1].set_ylabel(r"\textbf{Reverse} \unboldmath$D_{\textrm{KL}}$", fontsize=35, labelpad=15)
axs[1].tick_params(labelsize=30, length=15, width=3)
axs[1].set_xticks([1,3,5], [r"\textbf{Model 1}", r"\textbf{Model 2}", r"\textbf{Model 3}"])
axs[0].set_ylim([0.0, 0.10])
axs[0].set_ylim([0.0, 0.10])
axs[0].set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.1])
axs[1].set_yticks([0.0, 0.02, 0.04, 0.06, 0.08, 0.1])
lege = fig.legend(fontsize=35, loc="upper center", ncol=4, fancybox=False, edgecolor="black", columnspacing=0.25, handletextpad=0, bbox_to_anchor=(0.51, 1))
lege.get_frame().set_linewidth(3)
fig.tight_layout()
fig.subplots_adjust(top=0.82, wspace=0.25)
for ax in axs:
    for spine in ax.spines.values():
        spine.set_linewidth(3)

