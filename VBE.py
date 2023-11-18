import sys
sys.path.append("/home/cc/CodeProjects/NeuralDynamicalOperator")
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchdiffeq
import mat73
from utility import FNO_VBE, tke_spectrum_1d1d
import time


device = "cuda:1"
np.random.seed(0)
torch.manual_seed(0)

matplotlib.use("Qt5Agg")
plt.rcParams["agg.path.chunksize"] = 10000
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


##########################
### Data Preprocessing ###
##########################

# Simulation Setting: Lx = 1, Lt = 5, dx = 1/1024, dt = 0.005

# Data setting: dx = 1/1024, dt = 0.005
u_data = torch.from_numpy(mat73.loadmat("/home/cc/PythonProjects/NeuralDynamicalOperator/Data/burgers_data.mat")["output"]).to(torch.float32).permute(1, 0, 2)[1:]
x = torch.linspace(0, 1, 1024)
t = torch.linspace(0, 5, 1001)[1:]


# Subsampling for different resolutions

#1. dt = 0.05, dx = 1/512
# u_data = u_data[::10, :, ::2] # (100, 1100, 512)
# x = x[::2]
# t = t[::10]

#2. dt = 0.1, dx = 1/256
# u_data = u_data[::20, :, ::4] # (50, 1100, 256)
# x = x[::4]
# t = t[::20]

#3. dt = 0.5, dx = 1/64
u_data = u_data[::100, :, ::16] # (10, 1100, 64)
x = x[::16]
t = t[::100]

# Train/Test
Ntrain = 1000
Ntest = 100
u_train = u_data[:, :Ntrain, :]
u_test = u_data[:, -Ntest:, :]
del u_data

######################
### Model Training ###
######################

epochs = 10**3
iterations = epochs
loss_training_history = []
loss_test_history = []
test_freq = 10

model = FNO_VBE(24, 64).to(device)
optimizer = optim.Adam(model.parameters(), lr=10**-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
for ep in range(1, epochs+1):
    sim_idx = np.random.choice(Ntrain, 10)

    batch_y0 = u_train[0, sim_idx, :].to(device)
    batch_y = u_train[:, sim_idx, :].to(device)
    batch_t = t.to(device)

    optimizer.zero_grad()
    out = torchdiffeq.odeint(model, batch_y0, batch_t, method="rk4", options={"step_size":0.05})
    loss = F.mse_loss(batch_y, out)
    print(torch.cuda.memory_allocated(device=device) / 1024**2)
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_training_history.append(loss.item())
    print(ep, loss.item())

    if ep % test_freq == 0:
        # Test Loss
        with torch.no_grad():
            pred_test = torchdiffeq.odeint(model, u_test[0].to(device), batch_t)
        loss_test = F.mse_loss(u_test.to(device), pred_test).item()
        loss_test_history.append(loss_test)
        print(ep, "Loss Test: ", loss_test)

# torch.save(model.state_dict(), "/home/cc/PythonProjects/Neural_Dynamical_Operator/Model/Burgers_model_v5.pt")
# with open("/home/cc/PythonProjects/Neural_Dynamical_Operator/Model/Burgers_loss_training_v5.npy", "wb") as f:
#     np.save(f, loss_training_history)
# with open("/home/cc/PythonProjects/Neural_Dynamical_Operator/Model/Burgers_loss_test_v5.npy", "wb") as f:
#     np.save(f, loss_test_history)


#########################
### Model Application ###
#########################

u_data = torch.from_numpy(mat73.loadmat("/home/cc/CodeProjects/NeuralDynamicalOperator/Data/burgers_data.mat")["output"]).to(torch.float32).permute(1, 0, 2)[1:]
x = torch.linspace(0, 1, 1024)
t = torch.linspace(0, 5, 1001)[1:]
Ntrain = 1000
Ntest = 100
u_train = u_data[:, :Ntrain, :]
u_test = u_data[:, -Ntest:, :]
del u_data

# Pre-trained Model
model1 = FNO_VBE(24, 64).to(device)
model1.load_state_dict(torch.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Model/Burgers_model_v1.pt"))
model2 = FNO_VBE(24, 64).to(device)
model2.load_state_dict(torch.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Model/Burgers_model_v2.pt"))
model3 = FNO_VBE(24, 64).to(device)
model3.load_state_dict(torch.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Model/Burgers_model_v3.pt"))



u_test_res1 = u_test[::10, :, ::2] # (100, 1100, 512)
u_test_res2 = u_test[::20, :, ::4] # (50, 1100, 256)
u_test_res3 = u_test[::100, :, ::16] # (10, 1100, 64)
t_res1 = t[::10]
t_res2 = t[::20]
t_res3 = t[::100]

def rel_err(u, u_pred):
    u = u[1:]
    u_pred = u_pred[1:]
    shape0 = u.shape[0]
    shape1 = u.shape[1]
    shape2 = u.shape[2]
    u = u.reshape(shape0*shape1, shape2)
    u_pred = u_pred.reshape(shape0*shape1, shape2)
    return torch.mean(torch.norm(u-u_pred, 2, 1)/torch.norm(u, 2, 1)).item()


# Predictions & Test Error (Each Resolution)
with torch.no_grad():
    u_test_pred1_res1 = torchdiffeq.odeint(model1, u_test_res1[0, :, :].to(device), t_res1.to(device)).to("cpu")
    u_test_pred2_res2 = torchdiffeq.odeint(model2, u_test_res2[0, :, :].to(device), t_res2.to(device)).to("cpu")
    u_test_pred3_res3 = torchdiffeq.odeint(model3, u_test_res3[0, :, :].to(device), t_res3.to(device)).to("cpu")

torch.mean((u_test_res1[1:]-u_test_pred1_res1[1:])**2).item()
torch.mean((u_test_res2[1:]-u_test_pred2_res2[1:])**2).item()
torch.mean((u_test_res3[1:]-u_test_pred3_res3[1:])**2).item()
rel_err(u_test_res1, u_test_pred1_res1)
rel_err(u_test_res2, u_test_pred2_res2)
rel_err(u_test_res3, u_test_pred3_res3)


# Predictions & Test Error  (Same Resolution)
with torch.no_grad():
    u_test_pred1 = torchdiffeq.odeint(model1, u_test[0, :, :].to(device), t.to(device)).to("cpu")
    u_test_pred2 = torchdiffeq.odeint(model2, u_test[0, :, :].to(device), t.to(device)).to("cpu")
    u_test_pred3 = torchdiffeq.odeint(model3, u_test[0, :, :].to(device), t.to(device)).to("cpu")

torch.mean( (u_test[1:] - u_test_pred1[1:])**2 ).item()
torch.mean( (u_test[1:] - u_test_pred2[1:])**2 ).item()
torch.mean( (u_test[1:] - u_test_pred3[1:])**2 ).item()
rel_err(u_test, u_test_pred1)
rel_err(u_test, u_test_pred2)
rel_err(u_test, u_test_pred3)






## Pcolor (4th test simulation)
vmin = min([torch.min(u_test[:, 3, :]).item(), torch.min(u_test_pred1[:, 3, :]).item(), torch.min(u_test_pred2[:, 3, :]).item(), torch.min(u_test_pred3[:, 3, :]).item()])
vmax = max([torch.max(u_test[:, 3, :]).item(), torch.max(u_test_pred1[:, 3, :]).item(), torch.max(u_test_pred2[:, 3, :]).item(), torch.max(u_test_pred3[:, 3, :]).item()])
vmin_diff = 0
vmax_diff = max([torch.max(torch.abs(u_test[:, 3, :]-u_test_pred1[:, 3, :])).item(), torch.max(torch.abs(u_test[:, 3, :]-u_test_pred2[:, 3, :])).item(), torch.max(torch.abs(u_test[:, 3, :]-u_test_pred3[:, 3, :])).item()])
fig = plt.figure(layout="constrained")
fig.set_size_inches(30, 15)
axs = fig.subplots(3, 3)
axs[0,0].set_axis_off()
axs[2,0].set_axis_off()
# Truth
c = axs[1,0].pcolor(t, x, u_test[:, 3, :].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[1,0].set_xticks(range(6))
axs[1,0].set_yticks(np.arange(0, 1+0.5, 0.5))
axs[1,0].set_title(r"\textbf{Truth}", fontsize=40)
axs[1,0].set_xlabel(r"$t$", fontsize=50)
axs[1,0].set_ylabel(r"$x$", fontsize=50, rotation=0, labelpad=15)
axs[1,0].tick_params(labelsize=40, length=15, width=3)
axs[1,0].set_xticklabels([r'\boldmath${}$'.format(label) for label in axs[1,0].get_xticks()])
axs[1,0].set_yticklabels([r'\boldmath${}$'.format(label) for label in axs[1,0].get_yticks()])
# Prediction
axs[0,1].pcolor(t, x, u_test_pred1[:, 3, :].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[0,1].set_xticks(range(6))
axs[0,1].set_yticks(np.arange(0, 1+0.5, 0.5))
axs[0,1].set_title(r"\textbf{Model 1}", fontsize=40)
axs[0,1].set_xlabel(r"$ $", fontsize=50, labelpad=15)
axs[0,1].set_ylabel(r"$x$", fontsize=50, rotation=0, labelpad=15)
axs[0,1].tick_params(length=15, width=3, labelsize=40)
axs[0,1].set_xticklabels([r'\boldmath${}$'.format(label) for label in axs[0,1].get_xticks()])
axs[0,1].set_yticklabels([r'\boldmath${}$'.format(label) for label in axs[0,1].get_yticks()])
axs[1,1].pcolor(t, x, u_test_pred2[:, 3, :].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[1,1].set_xticks(range(6))
axs[1,1].set_yticks(np.arange(0, 1+0.5, 0.5))
axs[1,1].set_title(r"\textbf{Model 2}", fontsize=40)
axs[1,1].set_xlabel(r"$ $", fontsize=50, labelpad=15)
axs[1,1].set_ylabel(r"$x$", fontsize=50, rotation=0, labelpad=15)
axs[1,1].tick_params(length=15, width=3, labelsize=40)
axs[1,1].set_xticklabels([r'\boldmath${}$'.format(label) for label in axs[1,1].get_xticks()])
axs[1,1].set_yticklabels([r'\boldmath${}$'.format(label) for label in axs[1,1].get_yticks()])
axs[2,1].pcolor(t, x, u_test_pred3[:, 3, :].T, cmap='magma', vmin=vmin, vmax=vmax)
axs[2,1].set_xticks(range(6))
axs[2,1].set_yticks(np.arange(0, 1+0.5, 0.5))
axs[2,1].set_title(r"\textbf{Model 3}", fontsize=40)
axs[2,1].set_xlabel(r"$t$", fontsize=50)
axs[2,1].set_ylabel(r"$x$", fontsize=50, rotation=0, labelpad=15)
axs[2,1].tick_params(length=15, width=3, labelsize=40)
axs[2,1].set_xticklabels([r'\boldmath${}$'.format(label) for label in axs[2,1].get_xticks()])
axs[2,1].set_yticklabels([r'\boldmath${}$'.format(label) for label in axs[2,1].get_yticks()])
cbar = fig.colorbar(c, ax=axs[:,1])
cbar.ax.tick_params(length=15, width=3, labelsize=40)
cbar.ax.set_yticklabels([r'\boldmath${}$'.format(label.round(2)) for label in cbar.ax.get_yticks()])
# Difference
c = axs[0,2].pcolor(t, x, u_test[:, 3, :].T - u_test_pred1[:, 3, :].T, cmap='magma', vmin=vmin_diff, vmax=vmax_diff)
axs[0,2].set_xticks(range(6))
axs[0,2].set_yticks(np.arange(0, 1+0.5, 0.5))
axs[0,2].set_title(r"\textbf{Error 1}", fontsize=40)
axs[0,2].set_xlabel(r"$ $", fontsize=50)
axs[0,2].set_ylabel(r"$x$", fontsize=50, rotation=0, labelpad=15)
axs[0,2].tick_params(length=15, width=3, labelsize=40)
axs[0,2].set_xticks(range(6))
axs[0,2].set_yticks(np.arange(0, 1+0.5, 0.5))
axs[0,2].set_xticklabels([r'\boldmath${}$'.format(label) for label in axs[0,2].get_xticks()])
axs[0,2].set_yticklabels([r'\boldmath${}$'.format(label) for label in axs[0,2].get_yticks()])
axs[1,2].pcolor(t, x, u_test[:, 3, :].T - u_test_pred2[:, 3, :].T, cmap='magma', vmin=vmin_diff, vmax=vmax_diff)
axs[1,2].set_xticks(range(6))
axs[1,2].set_yticks(np.arange(0, 1+0.5, 0.5))
axs[1,2].set_title(r"\textbf{Error 2}", fontsize=40)
axs[1,2].set_xlabel(r"$ $", fontsize=50)
axs[1,2].set_ylabel(r"$x$", fontsize=50, rotation=0, labelpad=15)
axs[1,2].tick_params(length=15, width=3, labelsize=40)
axs[1,2].set_xticklabels([r'\boldmath${}$'.format(label) for label in axs[1,2].get_xticks()])
axs[1,2].set_yticklabels([r'\boldmath${}$'.format(label) for label in axs[1,2].get_yticks()])
axs[2,2].pcolor(t, x, u_test[:, 3, :].T - u_test_pred3[:, 3, :].T, cmap='magma', vmin=vmin_diff, vmax=vmax_diff)
axs[2,2].set_xticks(range(6))
axs[2,2].set_yticks(np.arange(0, 1+0.5, 0.5))
axs[2,2].set_title(r"\textbf{Error 3}", fontsize=40)
axs[2,2].set_xlabel(r"$t$", fontsize=50)
axs[2,2].set_ylabel(r"$x$", fontsize=50, rotation=0, labelpad=15)
axs[2,2].tick_params(length=15, width=3, labelsize=40)
axs[2,2].set_xticklabels([r'\boldmath${}$'.format(label) for label in axs[2,2].get_xticks()])
axs[2,2].set_yticklabels([r'\boldmath${}$'.format(label) for label in axs[2,2].get_yticks()])
cbar = fig.colorbar(c, ax=axs[:, 2])
cbar.ax.tick_params(length=15, width=3, labelsize=40)
cbar.ax.set_yticklabels([r'\boldmath${}$'.format(label.round(3)) for label in cbar.ax.get_yticks()])
plt.show()




## Time Series (t=0, 1, 3, 5s)
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
fig = plt.figure(figsize=(30, 18))
axs = fig.subplots(3, 4, sharex=True, sharey=True)
# First Profile
axs[0,0].plot(x, u_test[0, 3, :], linewidth=6)
axs[0,0].set_title(r"$t = 0.0$", fontsize=50)
axs[0,0].set_ylabel(r"$u$", fontsize=50, rotation=0)
axs[0,0].tick_params(labelsize=40, width=3, length=10)
axs[0,1].plot(x, u_test[200, 3, :], linewidth=6, label=r"\textbf{True}")
axs[0,1].plot(x, u_test_pred1[200, 3, :], linewidth=8, linestyle="dotted", label=r"\textbf{Model}")
axs[0,1].set_title(r"$t = 1.0$", fontsize=50)
axs[0,1].tick_params(labelsize=40, width=3, length=10)
axs[0,2].plot(x, u_test[600, 3, :], linewidth=6)
axs[0,2].plot(x, u_test_pred3[600, 3, :], linewidth=8, linestyle="dotted")
axs[0,2].set_title(r"$t = 3.0$", fontsize=50)
axs[0,2].tick_params(labelsize=40, width=3, length=10)
axs[0,3].plot(x, u_test[-1, 3, :], linewidth=6)
axs[0,3].plot(x, u_test_pred3[-1, 3, :], linewidth=8, linestyle="dotted")
axs[0,3].set_title(r"$t = 5.0$", fontsize=50)
axs[0,3].tick_params(labelsize=40, width=3, length=10)
# Second Profile
axs[1,0].plot(x, u_test[0, 50, :], linewidth=6)
axs[1,0].set_ylabel(r"$u$", fontsize=50, rotation=0)
axs[1,0].tick_params(labelsize=40, width=3, length=10)
axs[1,1].plot(x, u_test[200, 50, :], linewidth=6)
axs[1,1].plot(x, u_test_pred3[200, 50, :],linewidth=8, linestyle="dotted")
axs[1,1].tick_params(labelsize=40, width=3, length=10)
axs[1,2].plot(x, u_test[600, 50, :], linewidth=6)
axs[1,2].plot(x, u_test_pred3[600, 50, :],linewidth=8, linestyle="dotted")
axs[1,2].tick_params(labelsize=40, width=3, length=10)
axs[1,3].plot(x, u_test[-1, 50, :], linewidth=6)
axs[1,3].plot(x, u_test_pred3[-1, 50, :],linewidth=8, linestyle="dotted")
axs[1,3].tick_params(labelsize=40, width=3, length=10)
# Third Profile
axs[2,0].plot(x, u_test[0, -1, :], linewidth=6)
axs[2,0].set_xlabel(r"$x$", fontsize=50)
axs[2,0].set_ylabel(r"$u$", fontsize=50, rotation=0)
axs[2,0].tick_params(labelsize=40, width=3, length=10)
axs[2,1].plot(x, u_test[200, -1, :], linewidth=6)
axs[2,1].plot(x, u_test_pred3[200, -1, :],linewidth=8, linestyle="dotted")
axs[2,1].set_xlabel(r"$x$", fontsize=50)
axs[2,1].tick_params(labelsize=40, width=3, length=10)
axs[2,2].plot(x, u_test[600, -1, :], linewidth=6)
axs[2,2].plot(x, u_test_pred3[600, -1, :],linewidth=8, linestyle="dotted")
axs[2,2].set_xlabel(r"$x$", fontsize=50)
axs[2,2].tick_params(labelsize=40, width=3, length=10)
axs[2,3].plot(x, u_test[-1, -1, :], linewidth=6)
axs[2,3].plot(x, u_test_pred3[-1, -1, :], linewidth=8, linestyle="dotted")
axs[2,3].set_xlabel(r"$x$", fontsize=50)
axs[2,3].tick_params(labelsize=40, width=3, length=10)
for ax in axs.flatten():
    ax.set_xticks(np.arange(0, 1+0.5, 0.5))
    ax.set_yticks([0.2, 0.1, 0.0, -0.1])
    ax.set_xticklabels([r'\boldmath${}$'.format(label) for label in ax.get_xticks()])
    ax.set_yticklabels([r'\boldmath${}$'.format(label) for label in ax.get_yticks()])
    for spine in ax.spines.values():
        spine.set_linewidth(3)
handles, labels = axs[0,1].get_legend_handles_labels()
lege = fig.legend(fontsize=40, loc="upper center", ncol=2, fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(3)
fig.tight_layout()
fig.subplots_adjust(top=0.87)
plt.show()





# Energy Spectrum Profile
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \boldmath"
fig = plt.figure(figsize=(30, 18))
axs = fig.subplots(3, 4)
# First Energy Spectrum
axs[0,0].plot(tke_spectrum_1d1d(u_test[0, 3, :])[1], tke_spectrum_1d1d(u_test[0, 3, :])[2], linewidth=6)
axs[0,0].set_title(r"\unboldmath$t = 0.0$", fontsize=50)
axs[0,0].set_ylabel(r"\unboldmath$E$", fontsize=50, rotation=0, labelpad=15)
axs[0,1].plot(tke_spectrum_1d1d(u_test[200, 3, :])[1], tke_spectrum_1d1d(u_test[200, 3, :])[2], linewidth=6, label=r"\textbf{True}")
axs[0,1].plot(tke_spectrum_1d1d(u_test_pred1[200, 3, :])[1], tke_spectrum_1d1d(u_test_pred1[200, 3, :])[2], linewidth=4, label=r"\textbf{Model 1}")
axs[0,1].plot(tke_spectrum_1d1d(u_test_pred2[200, 3, :])[1], tke_spectrum_1d1d(u_test_pred2[200, 3, :])[2], linewidth=4, label=r"\textbf{Model 2}")
axs[0,1].plot(tke_spectrum_1d1d(u_test_pred3[200, 3, :])[1], tke_spectrum_1d1d(u_test_pred3[200, 3, :])[2], linewidth=4, label=r"\textbf{Model 3}")
xx = np.linspace(10**0.5, 10**2.5, 10000)
yy = xx**(-2)*np.exp(-1)
axs[0,1].plot(xx, yy, linewidth=6, color="black", linestyle="dashed")
axs[0,1].annotate(r"$k^{-2}$", xy=(xx[-1], yy[-1]), fontsize=40)
axs[0,1].set_title(r"\unboldmath$t = 1.0$", fontsize=50)
axs[0,2].plot(tke_spectrum_1d1d(u_test[600, 3, :])[1], tke_spectrum_1d1d(u_test[600, 3, :])[2], linewidth=6)
axs[0,2].plot(tke_spectrum_1d1d(u_test_pred1[600, 3, :])[1], tke_spectrum_1d1d(u_test_pred1[600, 3, :])[2],linewidth=4 )
axs[0,2].plot(tke_spectrum_1d1d(u_test_pred2[600, 3, :])[1], tke_spectrum_1d1d(u_test_pred2[600, 3, :])[2],linewidth=4 )
axs[0,2].plot(tke_spectrum_1d1d(u_test_pred3[600, 3, :])[1], tke_spectrum_1d1d(u_test_pred3[600, 3, :])[2],linewidth=4 )
xx = np.linspace(10**0.5, 10**2.5, 10000)
yy = xx**(-2)*np.exp(-1.5)
axs[0,2].plot(xx, yy, linewidth=6, color="black", linestyle="dashed")
axs[0,2].annotate(r"$k^{-2}$", xy=(xx[-1], yy[-1]), fontsize=40)
axs[0,2].set_title(r"\unboldmath$t = 3.0$", fontsize=50)
axs[0,3].plot(tke_spectrum_1d1d(u_test[-1, 3, :])[1], tke_spectrum_1d1d(u_test[-1, 3, :])[2], linewidth=6)
axs[0,3].plot(tke_spectrum_1d1d(u_test_pred1[-1, 3, :])[1], tke_spectrum_1d1d(u_test_pred1[-1, 3, :])[2], linewidth=4)
axs[0,3].plot(tke_spectrum_1d1d(u_test_pred2[-1, 3, :])[1], tke_spectrum_1d1d(u_test_pred2[-1, 3, :])[2], linewidth=4)
axs[0,3].plot(tke_spectrum_1d1d(u_test_pred3[-1, 3, :])[1], tke_spectrum_1d1d(u_test_pred3[-1, 3, :])[2], linewidth=4)
xx = np.linspace(10**0.5, 10**2.5, 10000)
yy = xx**(-2)*np.exp(-1.5)
axs[0,3].plot(xx, yy, linewidth=6, color="black", linestyle="dashed")
axs[0,3].annotate(r"$k^{-2}$", xy=(xx[-1], yy[-1]), fontsize=40)
axs[0,3].set_title(r"\unboldmath$t = 5.0$", fontsize=50)
# Second Energy Spectrum
axs[1,0].plot(tke_spectrum_1d1d(u_test[0, 50, :])[1], tke_spectrum_1d1d(u_test[0, 50, :])[2], linewidth=6)
axs[1,0].set_ylabel(r"\unboldmath$E$", fontsize=50, rotation=0, labelpad=15)
axs[1,1].plot(tke_spectrum_1d1d(u_test[200, 50, :])[1], tke_spectrum_1d1d(u_test[200, 50, :],1 , True)[2], linewidth=6)
axs[1,1].plot(tke_spectrum_1d1d(u_test_pred1[200, 50, :])[1], tke_spectrum_1d1d(u_test_pred1[200, 50, :])[2], linewidth=4)
axs[1,1].plot(tke_spectrum_1d1d(u_test_pred2[200, 50, :])[1], tke_spectrum_1d1d(u_test_pred2[200, 50, :])[2], linewidth=4)
axs[1,1].plot(tke_spectrum_1d1d(u_test_pred3[200, 50, :])[1], tke_spectrum_1d1d(u_test_pred3[200, 50, :])[2], linewidth=4)
xx = np.linspace(10**0.5, 10**2.5, 10000)
yy = xx**(-2)
axs[1,1].plot(xx, yy, linewidth=6, color="black", linestyle="dashed")
axs[1,1].annotate(r"$k^{-2}$", xy=(xx[-1], yy[-1]), fontsize=40)
axs[1,2].plot(tke_spectrum_1d1d(u_test[600, 50, :])[1], tke_spectrum_1d1d(u_test[600, 50, :])[2], linewidth=6)
axs[1,2].plot(tke_spectrum_1d1d(u_test_pred1[600, 50, :])[1], tke_spectrum_1d1d(u_test_pred1[600, 50, :])[2],linewidth=4)
axs[1,2].plot(tke_spectrum_1d1d(u_test_pred2[600, 50, :])[1], tke_spectrum_1d1d(u_test_pred2[600, 50, :])[2],linewidth=4)
axs[1,2].plot(tke_spectrum_1d1d(u_test_pred3[600, 50, :])[1], tke_spectrum_1d1d(u_test_pred3[600, 50, :])[2],linewidth=4)
xx = np.linspace(10**0.5, 10**2.5, 10000)
yy = xx**(-2)*np.exp(-1.5)
axs[1,2].plot(xx, yy, linewidth=6, color="black", linestyle="dashed")
axs[1,2].annotate(r"$k^{-2}$", xy=(xx[-1], yy[-1]), fontsize=40)
axs[1,3].plot(tke_spectrum_1d1d(u_test[-1, 50, :])[1], tke_spectrum_1d1d(u_test[-1, 50, :])[2], linewidth=6)
axs[1,3].plot(tke_spectrum_1d1d(u_test_pred1[-1, 50, :])[1], tke_spectrum_1d1d(u_test_pred1[-1, 50, :])[2], linewidth=4)
axs[1,3].plot(tke_spectrum_1d1d(u_test_pred2[-1, 50, :])[1], tke_spectrum_1d1d(u_test_pred2[-1, 50, :])[2], linewidth=4)
axs[1,3].plot(tke_spectrum_1d1d(u_test_pred3[-1, 50, :])[1], tke_spectrum_1d1d(u_test_pred3[-1, 50, :])[2], linewidth=4)
xx = np.linspace(10**0.5, 10**2.5, 10000)
yy = xx**(-2)*np.exp(-2)
axs[1,3].plot(xx, yy, linewidth=6, color="black", linestyle="dashed")
axs[1,3].annotate(r"$k^{-2}$", xy=(xx[-1], yy[-1]), fontsize=40)
# Third Energy Spectrum
axs[2,0].plot(tke_spectrum_1d1d(u_test[0, -1, :])[1], tke_spectrum_1d1d(u_test[0, -1, :])[2], linewidth=6)
axs[2,0].set_xlabel(r"\unboldmath$k$", fontsize=50, rotation=0)
axs[2,0].set_ylabel(r"\unboldmath$E$", fontsize=50, rotation=0, labelpad=15)
axs[2,1].plot(tke_spectrum_1d1d(u_test[200, -1, :])[1], tke_spectrum_1d1d(u_test[200, -1, :],1 , True)[2], linewidth=6)
axs[2,1].plot(tke_spectrum_1d1d(u_test_pred1[200, -1, :])[1], tke_spectrum_1d1d(u_test_pred1[200, -1, :])[2], linewidth=4)
axs[2,1].plot(tke_spectrum_1d1d(u_test_pred2[200, -1, :])[1], tke_spectrum_1d1d(u_test_pred2[200, -1, :])[2], linewidth=4)
axs[2,1].plot(tke_spectrum_1d1d(u_test_pred3[200, -1, :])[1], tke_spectrum_1d1d(u_test_pred3[200, -1, :])[2], linewidth=4)
xx = np.linspace(10**0.5, 10**2.5, 10000)
yy = xx**(-2)
axs[2,1].plot(xx, yy, linewidth=6, color="black", linestyle="dashed")
axs[2,1].annotate(r"$k^{-2}$", xy=(xx[-1], yy[-1]), fontsize=40)
axs[2,1].set_xlabel(r"\unboldmath$k$", fontsize=50, rotation=0)
axs[2,2].plot(tke_spectrum_1d1d(u_test[600, -1, :])[1], tke_spectrum_1d1d(u_test[600, -1, :])[2], linewidth=6)
axs[2,2].plot(tke_spectrum_1d1d(u_test_pred1[600, -1, :])[1], tke_spectrum_1d1d(u_test_pred1[600, -1, :])[2],linewidth=4)
axs[2,2].plot(tke_spectrum_1d1d(u_test_pred2[600, -1, :])[1], tke_spectrum_1d1d(u_test_pred2[600, -1, :])[2],linewidth=4)
axs[2,2].plot(tke_spectrum_1d1d(u_test_pred3[600, -1, :])[1], tke_spectrum_1d1d(u_test_pred3[600, -1, :])[2],linewidth=4)
xx = np.linspace(10**0.5, 10**2.5, 10000)
yy = xx**(-2)*np.exp(-1.5)
axs[2,2].plot(xx, yy, linewidth=6, color="black", linestyle="dashed")
axs[2,2].annotate(r"$k^{-2}$", xy=(xx[-1], yy[-1]), fontsize=40)
axs[2,2].set_xlabel(r"\unboldmath$k$", fontsize=50, rotation=0)
axs[2,3].plot(tke_spectrum_1d1d(u_test[-1, -1, :])[1], tke_spectrum_1d1d(u_test[-1, -1, :])[2], linewidth=6)
axs[2,3].plot(tke_spectrum_1d1d(u_test_pred1[-1, -1, :])[1], tke_spectrum_1d1d(u_test_pred1[-1, -1, :])[2], linewidth=4)
axs[2,3].plot(tke_spectrum_1d1d(u_test_pred2[-1, -1, :])[1], tke_spectrum_1d1d(u_test_pred2[-1, -1, :])[2], linewidth=4)
axs[2,3].plot(tke_spectrum_1d1d(u_test_pred3[-1, -1, :])[1], tke_spectrum_1d1d(u_test_pred3[-1, -1, :])[2], linewidth=4)
xx = np.linspace(10**0.5, 10**2.5, 10000)
yy = xx**(-2)*np.exp(-2)
axs[2,3].plot(xx, yy, linewidth=6, color="black", linestyle="dashed")
axs[2,3].annotate(r"$k^{-2}$", xy=(xx[-1], yy[-1]), fontsize=40)
axs[2,3].set_xlabel(r"\unboldmath$k$", fontsize=50, rotation=0)
for ax in axs.flatten():
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(labelsize=40, width=3, length=15)
    ax.tick_params(which="minor", width=3, length=5)
    ax.set_xlim([None, 10**3])
    for spine in ax.spines.values():
        spine.set_linewidth(3)
lege = fig.legend(fontsize=40, loc="upper center", ncol=4, fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(3)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()


