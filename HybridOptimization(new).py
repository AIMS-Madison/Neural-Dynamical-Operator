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
import torchdiffeq
import time
from utility import FNO_KSE, get_batchs, predict_short_relay, var_ux, var_uxx, kurtosis, kurtosis_ux, kurtosis_uxx, FNO_get_params, FNO_set_params, cross_cov


device = "cuda:0"
np.random.seed(42)
torch.manual_seed(42)

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

# Resolution 3: dt=2, dx=22/256
u_data = u_data[::8, ::4] # (2500, 256)
t = t[::8]
x = x[::4]
dt = (t[1] - t[0]).item()
dx = (x[1] - x[0]).item()
Ntrain = int(2500*0.8)
Ntest = int(2500*0.2)
train_u = u_data[:Ntrain]
test_u = u_data[-Ntest:]
train_t = t[:Ntrain]
test_t = t[-Ntest:]
short_steps = 2  # 4s
long_steps = 100 # 200s
test_short_steps = 10
test_long_steps = long_steps


##############################################################
################ Data for Hybrid Optimization ################
##############################################################

# Trajectory Data

# Setting 1
sample_size = 40
_, _, train_u_part = get_batchs(train_u, train_t, sample_size, short_steps)
train_t_part = train_t[:short_steps]
#  # Setting 2
# train_u_part = train_u[1000:1040]
# train_t_part = train_t[1000:1040]

# Statistics Data
train_idx_u0_longSim = np.arange(long_steps, Ntrain, long_steps) # Known initial point and long-term statistics. Unknown trajectory
train_var_u = torch.stack([torch.var(train_u[n:n + long_steps].to(device)) for n in train_idx_u0_longSim])
train_var_ux = torch.stack([var_ux(train_u[n:n + long_steps].to(device), dx) for n in train_idx_u0_longSim])
train_var_uxx = torch.stack([var_uxx(train_u[n:n + long_steps].to(device), dx) for n in train_idx_u0_longSim])
train_kurtosis_u = torch.stack([kurtosis(train_u[n:n + long_steps].to(device)) for n in train_idx_u0_longSim])
train_kurtosis_ux = torch.stack([kurtosis_ux(train_u[n:n + long_steps].to(device), dx) for n in train_idx_u0_longSim])
train_kurtosis_uxx = torch.stack([kurtosis_uxx(train_u[n:n + long_steps].to(device), dx) for n in train_idx_u0_longSim])

# Not used, only for check
test_idx_u0_longSim = np.arange(0, Ntest, test_long_steps)
test_var_u = torch.stack([torch.var(test_u[n:n + long_steps].to(device)) for n in test_idx_u0_longSim])
test_var_ux = torch.stack([var_ux(test_u[n:n + long_steps].to(device), dx) for n in test_idx_u0_longSim])
test_var_uxx = torch.stack([var_uxx(test_u[n:n + long_steps].to(device), dx) for n in test_idx_u0_longSim])
test_kurtosis_u = torch.stack([kurtosis(test_u[n:n + long_steps].to(device)) for n in test_idx_u0_longSim])
test_kurtosis_ux = torch.stack([kurtosis_ux(test_u[n:n + long_steps].to(device), dx) for n in test_idx_u0_longSim])
test_kurtosis_uxx = torch.stack([kurtosis_uxx(test_u[n:n + long_steps].to(device), dx) for n in test_idx_u0_longSim])

#########################################################
################ Model Training with SGD ################
#########################################################
epochs = 3000
train_batch_size = 10
train_loss_history = []
model_init = FNO_KSE(8, 16).to(device)
optimizer = torch.optim.Adam(model_init.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
for ep in range(1, epochs+1):
    start_time = time.time()
    optimizer.zero_grad()

    # Setting 1
    head_idx = np.random.choice(range(sample_size), size=train_batch_size, replace=False)
    u_short = train_u_part[:, head_idx]
    # #Setting 2
    # head_idx = np.random.choice(len(train_u_part)-short_steps+1, size=train_batch_size, replace=False)
    # u_short = torch.stack([train_u_part[head_idx+i] for i in range(short_steps)])

    out = torchdiffeq.odeint(model_init, u_short[0].to(device), train_t[:short_steps].to(device), method="rk4", options={"step_size":0.1})
    loss = F.mse_loss(u_short.to(device), out)
    # print(torch.cuda.memory_allocated(device=device) / 1024**2)
    loss.backward()
    optimizer.step()
    scheduler.step()
    train_loss_history.append(loss.item())
    end_time = time.time()
    print(ep, "Time: ", round(end_time-start_time, 4), "Loss: ",  round(loss.item(), 4) )


model_init = torch.load(r"C:\Users\chenc\CodeProject\Neural-Dynamical-Operator\Model_init\FNO(8_16)_SF2s_ep3000_samplesize40.pt")


# model_init Short-term
model_init.cpu()
test_u_shortPred_init = predict_short_relay(test_u, test_t, model_init, test_short_steps)
F.mse_loss(test_u, test_u_shortPred_init)


# model_init Long-term
model_init.to(device)
with torch.no_grad():
    train_u_longSim_init = torchdiffeq.odeint(model_init, train_u[train_idx_u0_longSim].to(device), train_t[:long_steps].to(device), method="euler", options={"step_size":0.2})
    test_u_longSim_init = torchdiffeq.odeint(model_init, test_u[test_idx_u0_longSim].to(device), test_t[:test_long_steps].to(device), method="euler", options={"step_size":0.2})

train_var_u_pred_init = torch.stack([torch.var(train_u_longSim_init[:, i, :]) for i in range(len(train_idx_u0_longSim))])
train_var_ux_pred_init = torch.stack([var_ux(train_u_longSim_init[:, i, :], dx, True) for i in range(len(train_idx_u0_longSim))])
train_var_uxx_pred_init = torch.stack([var_uxx(train_u_longSim_init[:, i, :], dx, True) for i in range(len(train_idx_u0_longSim))])
train_kurtosis_u_pred_init = torch.stack([kurtosis(train_u_longSim_init[:, i, :]) for i in range(len(train_idx_u0_longSim))])
train_kurtosis_ux_pred_init = torch.stack([kurtosis_ux(train_u_longSim_init[:, i, :], dx, True) for i in range(len(train_idx_u0_longSim))])
train_kurtosis_uxx_pred_init = torch.stack([kurtosis_uxx(train_u_longSim_init[:, i, :], dx, True) for i in range(len(train_idx_u0_longSim))])

test_var_u_pred_init = torch.stack([torch.var(test_u_longSim_init[:, i, :]) for i in range(len(test_idx_u0_longSim))])
test_var_ux_pred_init = torch.stack([var_ux(test_u_longSim_init[:, i, :], dx, True) for i in range(len(test_idx_u0_longSim))])
test_var_uxx_pred_init = torch.stack([var_uxx(test_u_longSim_init[:, i, :], dx, True) for i in range(len(test_idx_u0_longSim))])
test_kurtosis_u_pred_init = torch.stack([kurtosis(test_u_longSim_init[:, i, :]) for i in range(len(test_idx_u0_longSim))])
test_kurtosis_ux_pred_init = torch.stack([kurtosis_ux(test_u_longSim_init[:, i, :], dx, True) for i in range(len(test_idx_u0_longSim))])
test_kurtosis_uxx_pred_init = torch.stack([kurtosis_uxx(test_u_longSim_init[:, i, :], dx, True) for i in range(len(test_idx_u0_longSim))])

F.mse_loss(train_var_u, train_var_u_pred_init)
F.mse_loss(train_var_ux, train_var_ux_pred_init)
F.mse_loss(train_var_uxx, train_var_uxx_pred_init)
F.mse_loss(train_kurtosis_u, train_kurtosis_u_pred_init)
F.mse_loss(train_kurtosis_ux, train_kurtosis_ux_pred_init)
F.mse_loss(train_kurtosis_uxx, train_kurtosis_uxx_pred_init)

F.mse_loss(test_var_u, test_var_u_pred_init)
F.mse_loss(test_var_ux, test_var_ux_pred_init)
F.mse_loss(test_var_uxx, test_var_uxx_pred_init)
F.mse_loss(test_kurtosis_u, test_kurtosis_u_pred_init)
F.mse_loss(test_kurtosis_ux, test_kurtosis_ux_pred_init)
F.mse_loss(test_kurtosis_uxx, test_kurtosis_uxx_pred_init)

# model_init for PDF
test_u_longSim_init_smooth = torch.fft.irfft(torch.fft.rfft(test_u_longSim_init)[:, :, :8], n=256).cpu()
test_ux = (test_u[:, 2:] -  test_u[:, :-2])/(2*dx)
test_ux_init = (test_u_longSim_init_smooth[:,:, 2:] - test_u_longSim_init_smooth[:,:, :-2]) / (2*dx)
test_uxx = (test_u[:, 2:] - 2*test_u[:, 1:-1] + test_u[:, :-2])/dx**2
test_uxx_init = (test_u_longSim_init_smooth[:,:, 2:] - 2*test_u_longSim_init_smooth[:,:, 1:-1] + test_u_longSim_init_smooth[:,:, :-2]) / dx**2


fig = plt.figure(figsize=(30, 7))
axs = fig.subplots(1, 5)
for i in range(5):
    axs[i].set_title(r"\unboldmath\textbf{ $t$ in [" + str(4000+i*200)+ ", " +str(4000+(i+1)*200)+"]}", fontsize=30)
    # sns.kdeplot(test_u[100*i:100*i+100].flatten(), ax=axs[i], linewidth=3, label = r"\textbf{True}")
    # sns.kdeplot(test_u_longSim_init[:, i, :].flatten().cpu(), ax=axs[i], linewidth=3, label = r"\textbf{Model with classical optimization}")
    # sns.kdeplot(test_ux[100*i:100*i+100].flatten(), ax=axs[i], linewidth=3, label = r"\textbf{True}")
    # sns.kdeplot(test_ux_init[:, i, :].flatten().cpu(), ax=axs[i], linewidth=3, label = r"\textbf{Model with classical optimization}")
    sns.kdeplot(test_uxx[100*i:100*i+100].flatten(), ax=axs[i], linewidth=3, label = r"\textbf{True}")
    sns.kdeplot(test_uxx_init[:, i, :].flatten(), ax=axs[i], linewidth=3, label = r"\textbf{Model with classical optimization}")
    axs[i].tick_params(labelsize=30, length=5, width=2)
    axs[i].set_ylabel("")
    # axs[i].set_yticks([0, 0.2, 0.4, 0.6])
    # axs[i].set_ylim([0, 0.65])
    # axs[i].set_xticks([-4, -2, 0, 2, 4])
    # axs[i].set_xlim([-4.2, 4.2])
    axs[i].set_xlabel(r"\unboldmath$u_{xx}$", fontsize=30)
lege_handle, lege_label = axs[0].get_legend_handles_labels()
lege = fig.legend(handles=lege_handle, labels=lege_label, fontsize=30, loc="upper center", ncol=3, fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(1)
for ax in axs.flatten():
    for spine in ax.spines.values():
        spine.set_linewidth(1)
fig.tight_layout()
fig.subplots_adjust(top=0.78)


#############################################################
################ Model Training with SGD+EKI ################
#############################################################

model = FNO_KSE(8, 16).to(device)
model.load_state_dict(model_init.state_dict())

Nhybrid = 10
# SGD setup
epochs = 300
train_batch_size = 10
train_loss_history_sgd = []
# EKI setup
Nit = 20
J = 100
d = len(train_kurtosis_uxx)
p = len(torch.nn.utils.parameters_to_vector(model.parameters()))
c = 0.01  # Hyper-Parameter
COV_eta = c*torch.eye(d).to(device)
theta_bar_history_eki = []
train_state_loss_history_eki = np.zeros((Nhybrid, Nit+1)) # 2-D (#EKI, Nit+1)
train_kurtosis_loss_history_eki = np.zeros((Nhybrid, Nit+1)) # 2-D (#EKI, Nit+1)


def computing_hybrid_error():
    # This function is not Self-Contained, it needs other data from the script
    device = next(model.parameters()).device
    with torch.no_grad():
        train_u_longSim = torchdiffeq.odeint(model, train_u[train_idx_u0_longSim].to(device), train_t[:long_steps].to(device), method="euler", options={"step_size":0.2})
    train_kurtosis_uxx_pred = torch.stack([kurtosis_uxx(train_u_longSim[:,i,:], dx, True) for i in range(len(train_idx_u0_longSim))])
    train_kurtosis_loss = F.mse_loss(train_kurtosis_uxx, train_kurtosis_uxx_pred).item()
    # model.cpu()
    # train_u_shortPred = predict_short_relay(train_u, train_t, model, test_short_steps)
    # model.to(device)
    with torch.no_grad():
        train_u_part_shortPred = torchdiffeq.odeint(model, train_u_part[0].to(device), train_t[:short_steps].to(device)).cpu()
    train_state_loss = F.mse_loss(train_u_part, train_u_part_shortPred).item()
    return train_kurtosis_loss, train_state_loss


for nh in range(Nhybrid):
    # Computing Error (After SGD & Before EKI)
    train_kurtosis_loss, train_state_loss = computing_hybrid_error()
    train_kurtosis_loss_history_eki[nh, 0] = train_kurtosis_loss
    train_state_loss_history_eki[nh, 0] = train_state_loss
    print("long kurtosis error: ", round(train_kurtosis_loss, 4))
    print("short state error: ", round(train_state_loss, 4))

    print("EKI", nh+1, "Start")
    for nit in range(1, 1+Nit):
        print("Nit: ", nit)
        theta = FNO_get_params(model) # theta from SGD
        ensemble_theta = theta.repeat(J, 1)
        ensemble_theta[ensemble_theta.imag!=0] += 0.01*torch.randn_like(ensemble_theta[ensemble_theta.imag!=0])
        ensemble_theta[ensemble_theta.imag==0] += 0.001*torch.randn_like(ensemble_theta[ensemble_theta.imag==0].real)
        ensemble_g = torch.zeros(J, d).to(device)
        # 1) Ensemble Forward Mapping
        start_time = time.time()
        for j in range(J):
            FNO_set_params(model, ensemble_theta[j])
            with torch.no_grad():
                u_longSim = torchdiffeq.odeint(model, train_u[train_idx_u0_longSim].to(device), train_t[:long_steps].to(device), method="euler", options={"step_size":0.2})
            ensemble_g[j] = torch.stack([kurtosis_uxx(u_longSim[:,i,:], dx, True) for i in range(len(train_idx_u0_longSim))])
        end_time = time.time()
        print("Ensemble Simulation Time:", end_time - start_time)

        # 2) Updating ensemble_theta
        CCOV = cross_cov(ensemble_theta, ensemble_g)
        COV_gg = cross_cov(ensemble_g, ensemble_g)
        ensemble_y = train_kurtosis_uxx + c * torch.randn(J, d).to(device)
        ensemble_theta += (CCOV@torch.linalg.inv(COV_gg + COV_eta).to(torch.complex64)@(ensemble_y-ensemble_g).T.to(torch.complex64)).T
        theta_bar = ensemble_theta.mean(0)
        theta_bar_history_eki.append(theta_bar)

        # 3) Computing Error (Inside EKI)
        FNO_set_params(model, theta_bar)
        train_kurtosis_loss, train_state_loss = computing_hybrid_error()
        train_kurtosis_loss_history_eki[nh, nit] = train_kurtosis_loss
        train_state_loss_history_eki[nh, nit] = train_state_loss
        print("long kurtosis error: ", round(train_kurtosis_loss, 4) )
        print("short state error: ", round(train_state_loss, 4) )

    print("SGD Start")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for ep in range(1, epochs+1):
        start_time = time.time()
        optimizer.zero_grad()

        # Setting 1
        head_idx = np.random.choice(range(sample_size), size=train_batch_size, replace=False)
        u_short = train_u_part[:, head_idx].to(device)
        # #Setting 2
        # head_idx = np.random.choice(len(train_u_part)-short_steps+1, size=train_batch_size, replace=False)
        # u_short = torch.stack([train_u_part[head_idx+i] for i in range(short_steps)])

        out = torchdiffeq.odeint(model, u_short[0], train_t[:short_steps].to(device), method="rk4", options={"step_size":0.1})
        loss = F.mse_loss(u_short, out)
        # print(torch.cuda.memory_allocated(device=device) / 1024**2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss_history_sgd.append(loss.item())
        end_time = time.time()
        print( ep, "Time: ", round(end_time-start_time, 4), "Loss: ",  round(loss.item(), 4) )

torch.save(theta_bar_history_eki, r'C:\Users\chenc\CodeProject\Neural-Dynamical-Operator\ModelHybrid\(kurtosis)longSim200s_theta_bar_history.pt')
np.save(r'C:\Users\chenc\CodeProject\Neural-Dynamical-Operator\ModelHybrid\(kurtosis)longSim200s_train_loss_history_sgd.npy', train_loss_history_sgd)
np.save(r'C:\Users\chenc\CodeProject\Neural-Dynamical-Operator\ModelHybrid\(kurtosis)longSim200s_train_state_loss_history_eki.npy', train_state_loss_history_eki)
np.save(r'C:\Users\chenc\CodeProject\Neural-Dynamical-Operator\ModelHybrid\(kurtosis)longSim200s_train_kurtosis_loss_history_eki.npy', train_kurtosis_loss_history_eki)


####################
## Model Testing ###
####################

# Figure 1.
fig = plt.figure(figsize=(34, 12))
axs = fig.subplots(2, 5)
for i in range(Nhybrid):
    ax = axs.flatten()[i]
    ax2 = ax.twinx()
    ax.plot(train_state_loss_history_eki[i], label=r"\textbf{Short-term state loss}", color="C0", marker="o", markersize=8)
    ax2.plot(train_kurtosis_loss_history_eki[i], label=r"\textbf{Long-term statistics loss}",color="C1", marker="s", markersize=8)
    ax.set_title(r"\textbf{EKI " + str(i+1) + "}", fontsize=30)
    ax.set_xlim([0, 20])
    ax.set_ylim([np.min(train_state_loss_history_eki), np.max(train_state_loss_history_eki)])
    ax2.set_ylim([0, 0.1])
    # ax.set_yticks([0.00, 0.01, 0.02, 0.03])
    # ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.tick_params(labelsize=25, length=6, width=2)
    ax2.tick_params(labelsize=25, length=6, width=2)
    if i>4:
        ax.set_xlabel(r"\unboldmath$N_{\text{it}}$", fontsize=30)
axs[0,0].set_ylabel(r"\textbf{MSE}", fontsize=30)
axs[1,0].set_ylabel(r"\textbf{MSE}", fontsize=30)
lege_handle, lege_label = ax.get_legend_handles_labels()
lege_handle.extend(ax2.get_legend_handles_labels()[0])
lege_label.extend(ax2.get_legend_handles_labels()[1])
lege = fig.legend(handles=lege_handle, labels=lege_label, fontsize=30, loc="upper center", ncol=2, fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(1)
for ax in axs.flatten():
    for spine in ax.spines.values():
        spine.set_linewidth(1)
fig.tight_layout()
fig.subplots_adjust(hspace=0.3, top=0.85)



# Figure 2.
fig = plt.figure(figsize=(32, 12))
axs = fig.subplots(2, 5)
for i in range(Nhybrid):
    ax = axs.flatten()[i]
    ax.scatter(train_state_loss_history_eki[i], train_var_loss_history_eki[i], s=120)
    ax.set_title(r"\textbf{EKI " + str(i+1) + "}", fontsize=30)
    ax.tick_params(labelsize=25, length=6, width=2)
    if i>4:
        ax.set_xlabel(r"\textbf{Short-term state loss}", fontsize=30)
axs[0,0].set_ylabel(r"\textbf{Long-term statistics loss}", fontsize=30)
for ax in axs.flatten():
    ax.set_xlim([0, 0.03])
    ax.set_ylim([0, 0.9])
    ax.set_xticks([0.00, 0.01, 0.02, 0.03])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    for spine in ax.spines.values():
        spine.set_linewidth(1)
axs[0, 0].yaxis.set_label_coords(-0.15, 0.0)
fig.tight_layout()
fig.subplots_adjust(hspace=0.3, left=0.05)


# Model with "optimal" parameters
train_var_loss_history_eki[9, 2]
theta_bar_optimal = theta_bar_history_eki[181]
FNO_set_params(model, theta_bar_optimal)
# Predictive kurtosis after SGD+EKI training
with torch.no_grad():
    train_u_longSim_new = torchdiffeq.odeint(model, train_u[train_idx_u0_longSim].to(device), train_t[:long_steps].to(device), method="euler", options={"step_size":0.2})
    test_u_longSim_new = torchdiffeq.odeint(model, test_u[test_idx_u0_longSim].to(device), test_t[:test_long_steps].to(device), method="euler", options={"step_size":0.2})
train_var_uxx_pred_new = torch.stack([var_uxx(train_u_longSim_new[:,i,:], dx, True) for i in range(len(train_idx_u0_longSim))])
test_var_uxx_pred_new = torch.stack([var_uxx(test_u_longSim_new[:,i,:], dx, True) for i in range(len(test_idx_u0_longSim))])
F.mse_loss(train_var_uxx, train_var_uxx_pred_init)
F.mse_loss(train_var_uxx, train_var_uxx_pred_new)
F.mse_loss(test_var_uxx, test_var_uxx_pred_init)
F.mse_loss(test_var_uxx, test_var_uxx_pred_new)


test_u_longSim_init_smooth = torch.fft.irfft(torch.fft.rfft(test_u_longSim_init)[:, :, :8], n=256)
test_u_longSim_new_smooth = torch.fft.irfft( torch.fft.rfft(test_u_longSim_new)[:, :, :8], n=256)
test_uxx = (test_u[:, 2:] - 2*test_u[:, 1:-1] + test_u[:, :-2])/dx**2
test_uxx_init = (test_u_longSim_init_smooth[:,:, 2:] - 2*test_u_longSim_init_smooth[:,:, 1:-1] + test_u_longSim_init_smooth[:,:, :-2]) / dx**2
test_uxx_new = (test_u_longSim_new_smooth[:,:, 2:] - 2*test_u_longSim_new_smooth[:,:, 1:-1] + test_u_longSim_new_smooth[:,:, :-2]) / dx**2
test_uxx = test_uxx.cpu()
test_uxx_init = test_uxx_init.cpu()
test_uxx_new = test_uxx_new.cpu()
# Figure 3.
fig = plt.figure(figsize=(30, 8))
axs = fig.subplots(1, 5)
for i in range(5):
    axs[i].set_title(r"\unboldmath\textbf{ $t$ in [" + str(4000+i*200)+ ", " +str(4000+(i+1)*200)+"]}", fontsize=30)
    sns.kdeplot(test_uxx[100*i:100*i+100].flatten(), ax=axs[i], linewidth=3, label = r"\textbf{True}")
    sns.kdeplot(test_uxx_init[:, i, :].flatten(), ax=axs[i], linewidth=3, label = r"\textbf{Model with classical optimization}")
    sns.kdeplot(test_uxx_new[:, i, :].flatten(), ax=axs[i], linewidth=3, label = r"\textbf{Model with hybird optimization}")
    axs[i].tick_params(labelsize=30, length=5, width=2)
    axs[i].set_ylabel("")
    axs[i].set_yticks([0, 0.2, 0.4, 0.6])
    axs[i].set_ylim([0, 0.65])
    axs[i].set_xticks([-4, -2, 0, 2, 4])
    axs[i].set_xlim([-4.2, 4.2])
    axs[i].set_xlabel(r"\unboldmath$u_{xx}$", fontsize=30)
lege_handle, lege_label = axs[0].get_legend_handles_labels()
lege = fig.legend(handles=lege_handle, labels=lege_label, fontsize=30, loc="upper center", ncol=3, fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(1)
for ax in axs.flatten():
    for spine in ax.spines.values():
        spine.set_linewidth(1)
fig.tight_layout()
fig.subplots_adjust(top=0.78)




model_init.cpu()
model.cpu()
test_u_shortpred_init = predict_short_relay(test_u, test_t, model_init, 10)
test_u_shortpred_new = predict_short_relay(test_u, test_t, model, 10)
F.mse_loss(test_u, test_u_shortpred_init)
F.mse_loss(test_u, test_u_shortpred_new)
# Figure 4.
fig = plt.figure(figsize=(30, 18))
axs = fig.subplots(3, 4, sharex=True, sharey=True)
axs[0,0].plot(x, test_u[0], linewidth=5)
axs[0,0].set_ylabel(r"\unboldmath$u$", fontsize=50, rotation=0)
axs[0,0].set_title(r"\unboldmath$t = 4000$", fontsize=50)
axs[0,0].tick_params(labelsize=40, width=3, length=15)
axs[0,1].plot(x, test_u[2], linewidth=5, label=r"\textbf{True}")
axs[0,1].plot(x, test_u_shortpred_init[2], linewidth=5, linestyle="dashed", label=r"\textbf{Model with classical optimization}")
axs[0,1].plot(x, test_u_shortpred_new[2], linewidth=5, linestyle="dashed", label=r"\textbf{Model with hybrid optimization}")
axs[0,1].set_title(r"\unboldmath$t = 4004$", fontsize=50)
axs[0,1].tick_params(labelsize=40, width=3, length=15)
axs[0,2].plot(x, test_u[5], linewidth=5)
axs[0,2].plot(x, test_u_shortpred_init[5], linewidth=5, linestyle="dashed")
axs[0,2].plot(x, test_u_shortpred_new[5], linewidth=5, linestyle="dashed")
axs[0,2].set_title(r"\unboldmath$t = 4010$", fontsize=50)
axs[0,2].tick_params(labelsize=40, width=3, length=15)
axs[0,3].plot(x, test_u[9], linewidth=5)
axs[0,3].plot(x, test_u_shortpred_init[9], linewidth=5, linestyle="dashed")
axs[0,3].plot(x, test_u_shortpred_new[9], linewidth=5, linestyle="dashed")
axs[0,3].set_title(r"\unboldmath$t = 4020$", fontsize=50)
axs[0,3].tick_params(labelsize=40, width=3, length=15)
axs[1,0].plot(x, test_u[250], linewidth=5)
axs[1,0].set_ylabel(r"\unboldmath$u$", fontsize=50, rotation=0)
axs[1,0].set_title(r"\unboldmath$t = 4500$", fontsize=50)
axs[1,0].tick_params(labelsize=40, width=3, length=15)
axs[1,1].plot(x, test_u[252], linewidth=5)
axs[1,1].plot(x, test_u_shortpred_init[252], linewidth=5, linestyle="dashed")
axs[1,1].plot(x, test_u_shortpred_new[252], linewidth=5, linestyle="dashed")
axs[1,1].set_title(r"\unboldmath$t = 4504$", fontsize=50)
axs[1,1].tick_params(labelsize=40, width=3, length=15)
axs[1,2].plot(x, test_u[255], linewidth=5)
axs[1,2].plot(x, test_u_shortpred_init[255], linewidth=5, linestyle="dashed")
axs[1,2].plot(x, test_u_shortpred_new[255], linewidth=5, linestyle="dashed")
axs[1,2].set_title(r"\unboldmath$t = 4510$", fontsize=50)
axs[1,2].tick_params(labelsize=40, width=3, length=15)
axs[1,3].plot(x, test_u[259], linewidth=5)
axs[1,3].plot(x, test_u_shortpred_init[259], linewidth=5, linestyle="dashed")
axs[1,3].plot(x, test_u_shortpred_new[259], linewidth=5, linestyle="dashed")
axs[1,3].set_title(r"\unboldmath$t = 4520$", fontsize=50)
axs[1,3].tick_params(labelsize=40, width=3, length=15)
axs[2,0].plot(x, test_u[450], linewidth=5)
axs[2,0].set_xlabel(r"\unboldmath$x$", fontsize=50)
axs[2,0].set_ylabel(r"\unboldmath$u$", fontsize=50, rotation=0)
axs[2,0].set_title(r"\unboldmath$t = 4900$", fontsize=50)
axs[2,0].tick_params(labelsize=40, width=3, length=15)
axs[2,1].plot(x, test_u[452], linewidth=5)
axs[2,1].plot(x, test_u_shortpred_init[452], linewidth=5, linestyle="dashed")
axs[2,1].plot(x, test_u_shortpred_new[452], linewidth=5, linestyle="dashed")
axs[2,1].set_xlabel(r"\unboldmath$x$", fontsize=50)
axs[2,1].set_title(r"\unboldmath$t = 4904$", fontsize=50)
axs[2,1].tick_params(labelsize=40, width=3, length=15)
axs[2,2].plot(x, test_u[455], linewidth=5)
axs[2,2].plot(x, test_u_shortpred_init[455], linewidth=5, linestyle="dashed")
axs[2,2].plot(x, test_u_shortpred_new[455], linewidth=5, linestyle="dashed")
axs[2,2].set_xlabel(r"\unboldmath$x$", fontsize=50)
axs[2,2].set_title(r"\unboldmath$t = 4910$", fontsize=50)
axs[2,2].tick_params(labelsize=40, width=3, length=15)
axs[2,3].plot(x, test_u[459], linewidth=5)
axs[2,3].plot(x, test_u_shortpred_init[459], linewidth=5, linestyle="dashed")
axs[2,3].plot(x, test_u_shortpred_new[459], linewidth=5, linestyle="dashed")
axs[2,3].set_xlabel(r"\unboldmath$x$", fontsize=50)
axs[2,3].set_title(r"\unboldmath$t = 4920$", fontsize=50)
axs[2,3].tick_params(labelsize=40, width=3, length=15)
for ax in axs.flatten():
    for spine in ax.spines.values():
        spine.set_linewidth(3)
lege = fig.legend(fontsize=40, loc="upper center", ncol=3, fancybox=False, edgecolor="black", bbox_to_anchor=(0.52, 1))
lege.get_frame().set_linewidth(2)
fig.tight_layout()
fig.subplots_adjust(top=0.87)


# Figure 5.
fig = plt.figure(figsize=(12, 12))
ax = fig.subplots()
sns.kdeplot(test_uxx.flatten(), ax=ax, linewidth=5, label = r"\textbf{True}")
sns.kdeplot(test_uxx_init.flatten(), ax=ax, linewidth=5, label = r"\textbf{Model with classical optimization}")
sns.kdeplot(test_uxx_new.flatten(), ax=ax, linewidth=5, label = r"\textbf{Model with hybrid optimization}")
ax.set_ylabel("")
ax.set_xlabel(r"\unboldmath$u_{xx}$", fontsize=35)
ax.set_title(r"\textbf{PDF of} \unboldmath$u_{xx}$", fontsize=30, pad=10)
ax.tick_params(labelsize=35, length=15, width=3)
for spine in ax.spines.values():
    spine.set_linewidth(3)
lege = fig.legend(fontsize=30, loc="upper center", ncol=1, fancybox=False, edgecolor="black", bbox_to_anchor=(0.52, 1))
lege.get_frame().set_linewidth(2)
fig.tight_layout()
fig.subplots_adjust(top=0.77)

