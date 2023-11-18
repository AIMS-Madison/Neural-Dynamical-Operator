import sys
sys.path.append("/home/cc/CodeProjects/NeuralDynamicalOperator")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchdiffeq
from utility import FNO_NSE, solve_poisson_equation_2d_periodic, stream2velocity, tke_spectrum_2d2d, tke_spectrum_1d1d
import mat73

device = "cuda:0"
np.random.seed(0)
torch.manual_seed(0)


matplotlib.use("Qt5Agg")
plt.rcParams["agg.path.chunksize"] = 10000
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

###########################
#### Data Preprocessing ###
###########################

# Simulation Setting: x = (0, 1)^2, Lt = 20, dx1 = 1/256, dx2 = 1/256, dt = 1e-4, nu=1e-3

# Data Setting: dx = 1/64, dy = 1/64, dt = 0.2
u_data = torch.from_numpy(np.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Data/NS_data6464.npy")) # (100, 1100, 64, 64)
t = torch.linspace(0, 20, 101)[1:]


# subsampling for different resolutions

#1. dx = 1/64, dy = 1/64, dt = 0.2
# (100, 1100, 64, 64)
# FNO2d(12,12,32)

#2. dx = 1/64,dy = 1/64, dt = 0.4
# u_data = u_data[::2, :]  # (50, 1100, 64, 64)
# t = t[::2]
# FNO2d(12,12,32)

#3. dx = 1/64, dy = 1/64, dt = 1
# u_data = u_data[::5, :]  # (20, 1100, 64, 64)
# t = t[::5]
# FNO2d(12,12,32)

#4. 1/32, dy=1/32, dt = 0.2
# u_data = u_data[:, :, ::2, ::2]  # (100, 1100, 32, 32)
# FNO2d(12,12,32)

#5. dx = 1/16, dy=1/16, dt = 0.2
u_data = u_data[:, :, ::4, ::4]  # (100, 1100, 16, 16)
# FNO2d(8,8,24)


# Train/Test
Ntrain = 1000
Ntest = 100
u_train = u_data[:, :Ntrain, :, :]
u_test = u_data[:, -Ntest:, :, :]
del u_data


#######################
### Model Trainning ###
#######################

epochs = 20000
numbatch = 1
test_feq = 200
loss_training_history = []
loss_test_history = []
iterations = epochs

model = FNO_NSE(12, 12, 32).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
model.train()
for ep in range(1, epochs+1):

    sim_idx = np.random.choice(Ntrain, numbatch)

    batch_u0 = u_train[0, sim_idx, :, :].to(device)
    batch_u = u_train[:, sim_idx, :, :].to(device)
    batch_t = t.to(device)
    optimizer.zero_grad()
    out = torchdiffeq.odeint(model, batch_u0, batch_t, method="rk4", options={"step_size":0.1})
    loss = F.mse_loss(out, batch_u)
    print(torch.cuda.memory_allocated(device=device) / 1024**2)
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_training_history.append(loss.item())
    print(ep, loss.item())

    if ep % test_feq == 0:
        # Test Loss
        with torch.no_grad():
            pred_test = torchdiffeq.odeint(model, u_test[0].to(device), batch_t)
        loss_test = F.mse_loss(u_test.to(device), pred_test).item()
        loss_test_history.append(loss_test)
        print(ep, "Loss Test: ", loss_test)


# torch.save(model.state_dict(), "/home/cc/PythonProjects/Neural_Dynamical_Operator/Model/NS_model_v5.pt")
# with open("/home/cc/PythonProjects/Neural_Dynamical_Operator/Model/NS_loss_training_v5.npy", "wb") as f:
#     np.save(f, loss_training_history)
# with open("/home/cc/PythonProjects/Neural_Dynamical_Operator/Model/NS_loss_test_v5.npy", "wb") as f:
#     np.save(f, loss_test_history)


#########################
### Model Application ###
#########################
u_data = torch.from_numpy(np.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Data/NS_data6464.npy")) # (100, 1100, 64, 64)
t = torch.linspace(0, 20, 101)[1:]
Ntrain = 1000
Ntest = 100
u_test = u_data[:, -Ntest:, :, :]
del u_data


# Pre-trained Model
model1 = FNO_NSE(12, 12, 32).to(device)
model1.load_state_dict(torch.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Model/NS_model_v1.pt"))
model2 = FNO_NSE(12, 12, 32).to(device)
model2.load_state_dict(torch.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Model/NS_model_v2.pt"))
model3 = FNO_NSE(12, 12, 32).to(device)
model3.load_state_dict(torch.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Model/NS_model_v3.pt"))
model4 = FNO_NSE(12, 12, 32).to(device)
model4.load_state_dict(torch.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Model/NS_model_v4.pt"))
model5 = FNO_NSE(8, 8, 24).to(device)
model5.load_state_dict(torch.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Model/NS_model_v5.pt"))


# (t, N, x, y)
u_test_res1 = u_test[:] # (100, 100, 64, 64)
u_test_res2 = u_test[::2] # (50, 100, 64, 64)
u_test_res3 = u_test[::5] # (20, 100, 64, 64)
u_test_res4 = u_test[:, :, ::2, ::2 ] # (100, 100, 32, 32)
u_test_res5 = u_test[:, :, ::4, ::4 ] # (100, 100, 16, 16)
t_res1 = t[:]
t_res2 = t[::2]
t_res3 = t[::5]
t_res4 = t[:]
t_res5 = t[:]

def rel_err(u, u_pred):
    u = u[1:]
    u_pred = u_pred[1:]
    shape0 = u.shape[0]
    shape1 = u.shape[1]
    shape2 = u.shape[2]
    shape3 = u.shape[3]
    u = u.reshape(shape0*shape1, shape2*shape3)
    u_pred = u_pred.reshape(shape0*shape1, shape2*shape3)
    return torch.mean(torch.norm(u-u_pred, 2, 1)/torch.norm(u, 2, 1)).item()


# Predictions & Test Error (Each Resolution)

with torch.no_grad():
    u_test_pred1_res1 = torchdiffeq.odeint(model1, u_test_res1[0].to(device), t_res1.to(device)).to("cpu")
    u_test_pred2_res2 = torchdiffeq.odeint(model2, u_test_res2[0].to(device), t_res2.to(device)).to("cpu")
    u_test_pred3_res3 = torchdiffeq.odeint(model3, u_test_res3[0].to(device), t_res3.to(device)).to("cpu")
    u_test_pred4_res4 = torchdiffeq.odeint(model4, u_test_res4[0].to(device), t_res4.to(device)).to("cpu")
    u_test_pred5_res5 = torchdiffeq.odeint(model5, u_test_res5[0].to(device), t_res5.to(device)).to("cpu")

torch.mean((u_test_res1[1:] - u_test_pred1_res1[1:])**2).item()
torch.mean((u_test_res2[1:] - u_test_pred2_res2[1:])**2).item()
torch.mean((u_test_res3[1:] - u_test_pred3_res3[1:])**2).item()
torch.mean((u_test_res4[1:] - u_test_pred4_res4[1:])**2).item()
torch.mean((u_test_res5[1:] - u_test_pred5_res5[1:])**2).item()
rel_err(u_test_res1, u_test_pred1_res1)
rel_err(u_test_res2, u_test_pred2_res2)
rel_err(u_test_res3, u_test_pred3_res3)
rel_err(u_test_res4, u_test_pred4_res4)
rel_err(u_test_res5, u_test_pred5_res5)


# Predictions & Test Error (Same Resolution)
with torch.no_grad():
    u_test_pred1 = torchdiffeq.odeint(model1, u_test[0].to(device), t.to(device)).to("cpu")
    u_test_pred2 = torchdiffeq.odeint(model2, u_test[0].to(device), t.to(device)).to("cpu")
    u_test_pred3 = torchdiffeq.odeint(model3, u_test[0].to(device), t.to(device)).to("cpu")
    u_test_pred4 = torchdiffeq.odeint(model4, u_test[0].to(device), t.to(device)).to("cpu")
    u_test_pred5 = torchdiffeq.odeint(model5, u_test[0].to(device), t.to(device)).to("cpu")

torch.mean((u_test[1:]-u_test_pred1[1:])**2).item()
torch.mean((u_test[1:]-u_test_pred2[1:])**2).item()
torch.mean((u_test[1:]-u_test_pred3[1:])**2).item()
torch.mean((u_test[1:]-u_test_pred4[1:])**2).item()
torch.mean((u_test[1:]-u_test_pred5[1:])**2).item()
rel_err(u_test, u_test_pred1)
rel_err(u_test, u_test_pred2)
rel_err(u_test, u_test_pred3)
rel_err(u_test, u_test_pred4)
rel_err(u_test, u_test_pred5)





# 51-th simulation Flow Heatmap  (Each Resolution)
fig = plt.figure(layout="constrained")
fig.set_size_inches(25, 10)
ax = fig.subplots(2, 6)
for i in range(6):
    ax[0,i].set_title(r"$t={}$".format(4*i), fontsize=50)
for i in range(5):
    sns.heatmap(u_test_res5[20*i, 50, :, :], ax=ax[0, i], cbar=False, xticklabels=False, yticklabels=False)
for i in range(5):
    sns.heatmap(u_test_pred5_res5[20*i, 50, :, :], ax=ax[1, i], cbar=False, xticklabels=False, yticklabels=False)
sns.heatmap(u_test_res5[-1, 50, :, :], ax=ax[0, -1], cbar=False, xticklabels=False, yticklabels=False)
sns.heatmap(u_test_pred5_res5[-1, 50, :, :], ax=ax[1, -1], cbar=False, xticklabels=False, yticklabels=False)
ax[0,0].set_ylabel(r"\textbf{True}", fontsize=40)
ax[1,0].set_ylabel(r"\textbf{Model 5}", fontsize=40)




# 51-th simulation Flow Heatmap  (Same Resolution)
fig = plt.figure(layout="constrained")
fig.set_size_inches(25, 15)
ax = fig.subplots(4, 6)
for i in range(6):
    ax[0,i].set_title(r"$t={}$".format(4*i), fontsize=50)
for i in range(5):
    sns.heatmap(u_test[20*i, 50, :, :], ax=ax[0, i], cbar=False, xticklabels=False, yticklabels=False)
for i in range(5):
    sns.heatmap(u_test_pred3[20*i, 50, :, :], ax=ax[1, i], cbar=False, xticklabels=False, yticklabels=False)
for i in range(5):
    sns.heatmap(u_test_pred4[20*i, 50, :, :], ax=ax[2, i], cbar=False, xticklabels=False, yticklabels=False)
for i in range(5):
    sns.heatmap(u_test_pred5[20*i, 50, :, :], ax=ax[3, i], cbar=False, xticklabels=False, yticklabels=False)
sns.heatmap(u_test[-1, 50, :, :], ax=ax[0, -1], cbar=False, xticklabels=False, yticklabels=False)
sns.heatmap(u_test_pred3[-1, 50, :, :], ax=ax[1, -1], cbar=False, xticklabels=False, yticklabels=False)
sns.heatmap(u_test_pred4[-1, 50, :, :], ax=ax[2, -1], cbar=False, xticklabels=False, yticklabels=False)
sns.heatmap(u_test_pred5[-1, 50, :, :], ax=ax[3, -1], cbar=False, xticklabels=False, yticklabels=False)
ax[0,0].set_ylabel(r"\textbf{True}", fontsize=40)
ax[1,0].set_ylabel(r"\textbf{Model 3}", fontsize=40)
ax[2,0].set_ylabel(r"\textbf{Model 4}", fontsize=40)
ax[3,0].set_ylabel(r"\textbf{Model 5}", fontsize=40)





# Energy Spectrum
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \boldmath"

omega = u_test[0, 50, :, :]
stream = solve_poisson_equation_2d_periodic(-omega, 1, 64)
v1, v2 = stream2velocity(stream)
wave_numbers, tke_spectrum = tke_spectrum_2d2d(v1, v2)[1:]
fig, axs = plt.subplots(2, 3, figsize=[30, 15])
axs[0,0].plot(wave_numbers, tke_spectrum, linewidth=6)
xx = np.linspace(10, 10**1.5, 10000)
yy = xx**(-3)*np.exp(-5)
axs[0,0].plot(xx, yy, linewidth=6, color="black", linestyle="dashed")
axs[0,0].annotate(r"\unboldmath$k^{-3}$", xy=(xx[-1], yy[-1]), fontsize=40)
axs[0,0].set_title(r"\unboldmath$t = 0$", fontsize=50)
axs[0,0].set_ylabel(r"\unboldmath$E$", fontsize=40, rotation=0)
axs = axs.flatten()
idx = 1
t_title = range(0, 20+4, 4)
for omega, omega1, omega2, omega3, omega4, omega5 in zip(u_test[[20, 40, 60, 80, -1], 50],
                                                         u_test_pred1[[20, 40, 60, 80, -1], 50],
                                                         u_test_pred2[[20, 40, 60, 80, -1], 50],
                                                         u_test_pred3[[20, 40, 60, 80, -1], 50],
                                                         u_test_pred4[[20, 40, 60, 80, -1], 50],
                                                         u_test_pred5[[20, 40, 60, 80, -1], 50]):

    stream = solve_poisson_equation_2d_periodic(-omega, 1, 64)
    stream1 = solve_poisson_equation_2d_periodic(-omega1, 1, 64)
    stream2 = solve_poisson_equation_2d_periodic(-omega2, 1, 64)
    stream3 = solve_poisson_equation_2d_periodic(-omega3, 1, 64)
    stream4 = solve_poisson_equation_2d_periodic(-omega4, 1, 64)
    stream5 = solve_poisson_equation_2d_periodic(-omega5, 1, 64)

    v1, v2 = stream2velocity(stream)
    v1_1, v1_2 = stream2velocity(stream1)
    v2_1, v2_2 = stream2velocity(stream2)
    v3_1, v3_2 = stream2velocity(stream3)
    v4_1, v4_2 = stream2velocity(stream4)
    v5_1, v5_2 = stream2velocity(stream5)

    wave_numbers, tke_spectrum = tke_spectrum_2d2d(v1, v2)[1:]
    axs[idx].plot(wave_numbers, tke_spectrum, linewidth=6, label=r"\textbf{True}")
    # wave_numbers, tke_spectrum = tke_spectrum_2d2d(v1_1, v1_2)[1:]
    # axs[idx].plot(wave_numbers, tke_spectrum, linewidth=4, label=r"\textbf{Model 1}")
    # wave_numbers, tke_spectrum = tke_spectrum_2d2d(v2_1, v2_2)[1:]
    # axs[idx].plot(wave_numbers, tke_spectrum, linewidth=6, label=r"\textbf{Model 2}")
    wave_numbers, tke_spectrum = tke_spectrum_2d2d(v3_1, v3_2)[1:]
    axs[idx].plot(wave_numbers, tke_spectrum, linewidth=6, label=r"\textbf{Model 3}", linestyle="dotted")
    wave_numbers, tke_spectrum = tke_spectrum_2d2d(v4_1, v4_2)[1:]
    axs[idx].plot(wave_numbers, tke_spectrum, linewidth=4, label=r"\textbf{Model 4}")
    wave_numbers, tke_spectrum = tke_spectrum_2d2d(v5_1, v5_2)[1:]
    axs[idx].plot(wave_numbers, tke_spectrum, linewidth=4, label=r"\textbf{Model 5}")

    xx = np.linspace(10, 10**1.5, 10000)
    if idx==1:
        yy = xx**-3*np.exp(-4.8)
        axs[idx].plot(xx, yy, linewidth=6, color="black", linestyle="dashed")
        axs[idx].annotate(r"\unboldmath$k^{-3}$", xy=(xx[-1], yy[-1]), fontsize=40)
    elif idx==4 or idx==5:
        yy = xx**-3*np.exp(-3)
        axs[idx].plot(xx, yy, linewidth=6, color="black", linestyle="dashed")
        axs[idx].annotate(r"\unboldmath$k^{-3}$", xy=(xx[-1], yy[-1]), fontsize=40)
    else:
        yy = xx**-3*np.exp(-4)
        axs[idx].plot(xx, yy, linewidth=6, color="black", linestyle="dashed")
        axs[idx].annotate(r"\unboldmath$k^{-3}$", xy=(xx[-1], yy[-1]), fontsize=40)
    axs[idx].set_title(r"\unboldmath$t = "+str(t_title[idx]) + "$", fontsize=50)
    idx += 1
axs[3].set_ylabel(r"\unboldmath$E$", fontsize=40, rotation=0)
axs[3].set_xlabel(r"\unboldmath$k$", fontsize=45, rotation=0)
axs[4].set_xlabel(r"\unboldmath$k$", fontsize=45, rotation=0)
axs[5].set_xlabel(r"\unboldmath$k$", fontsize=45, rotation=0)
for ax in axs:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([10**(-10), 10**(-2)])
    ax.tick_params(labelsize=40, width=3, length=15)
    ax.tick_params(which="minor", width=3, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
handles, labels = axs[1].get_legend_handles_labels()
lege = fig.legend(handles, labels, fontsize=40, loc="upper center", ncol=4, fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(3)
fig.tight_layout()
fig.subplots_adjust(top=0.84)



## Energy Spectrum Comparison
# VBE
u_data = torch.from_numpy(mat73.loadmat("/home/cc/CodeProjects/NeuralDynamicalOperator/Data/burgers_data.mat")["output"]).to(torch.float32).permute(1, 0, 2)[1:]
u64 = u_data[0, 0, ::16]
u256 = u_data[0, 0, ::4]
u512 = u_data[0, 0, ::2]
del u_data
#NSE
omega64 = torch.from_numpy(np.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Data/NS_data6464.npy"))[0, 50] # (100, 1100, 64, 64)
omega256 = torch.from_numpy(np.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Data/NS_data.npy")).permute(3,0,1,2)[0, 50]
omega32 = omega64[::2, ::2]
omega16 = omega64[::4, ::4]
stream16 = solve_poisson_equation_2d_periodic(-omega16, 1,16)
v1, v2 = stream2velocity(stream16)
wave_numbers16, tke_spectrum16 = tke_spectrum_2d2d(v1, v2)[1:]
stream32 = solve_poisson_equation_2d_periodic(-omega32, 1, 32)
v1, v2 = stream2velocity(stream32)
wave_numbers32, tke_spectrum32 = tke_spectrum_2d2d(v1, v2)[1:]
stream64 = solve_poisson_equation_2d_periodic(-omega64, 1, 64)
v1, v2 = stream2velocity(stream64)
wave_numbers64, tke_spectrum64 = tke_spectrum_2d2d(v1, v2)[1:]
stream256 = solve_poisson_equation_2d_periodic(-omega256, 1, 256)
v1, v2 = stream2velocity(stream256)
wave_numbers256, tke_spectrum256 = tke_spectrum_2d2d(v1, v2)[1:]

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath} \boldmath"
fig = plt.figure(figsize=(20, 8), layout="constrained")
axs = fig.subplots(1, 2)
axs[0].plot(tke_spectrum_1d1d(u64)[1], tke_spectrum_1d1d(u64)[2], linewidth=6, label=r"\textbf{64}")
axs[0].plot(tke_spectrum_1d1d(u256)[1], tke_spectrum_1d1d(u256)[2], linewidth=6, label=r"\textbf{256}", linestyle="dashed")
axs[0].plot(tke_spectrum_1d1d(u512)[1], tke_spectrum_1d1d(u512)[2], linewidth=6, label=r"\textbf{512}", linestyle="dotted")
axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[0].set_title(r"\textbf{(a) viscous Burgers' Equation}", fontsize=35)
axs[0].set_xlabel(r"\unboldmath$k$", fontsize=45, rotation=0)
axs[0].set_ylabel(r"\unboldmath$E$", fontsize=45, rotation=0, labelpad=15)
axs[0].legend(fontsize=35)
axs[0].tick_params(labelsize=35, width=3, length=15)
axs[0].tick_params(which="minor", width=3, length=5)
axs[0].set_xlim([None, 10**3])
axs[0].set_ylim([10**-10, 10**-0.9])
axs[0].set_yticks([10**(-i)for i in range(2, 11, 2)])
axs[1].plot(wave_numbers16, tke_spectrum16, linewidth=6, label=r"\textbf{16}")
axs[1].plot(wave_numbers32, tke_spectrum32, linewidth=6, label=r"\textbf{32}")
axs[1].plot(wave_numbers64, tke_spectrum64, linewidth=6, label=r"\textbf{64}")
axs[1].plot(wave_numbers256, tke_spectrum256, linewidth=6, label=r"\textbf{256}")
axs[1].set_xscale("log")
axs[1].set_yscale("log")
axs[1].set_title(r"\textbf{(b) Navier-Stokes Equation}", fontsize=35)
axs[1].set_xlabel(r"\unboldmath$k$", fontsize=45, rotation=0)
axs[1].set_ylabel(r"\unboldmath$E$", fontsize=45, rotation=0, labelpad=15)
axs[1].tick_params(labelsize=35, width=3, length=15)
axs[1].tick_params(which="minor", width=3, length=5)
axs[1].legend(fontsize=35)
axs[1].set_xlim([None, 10**3])
axs[1].set_ylim([10**-10, 10**-0.9])
axs[1].set_yticks([10**(-i)for i in range(2, 11, 2)])
for ax in axs.flatten():
    for spine in ax.spines.values():
        spine.set_linewidth(3)

