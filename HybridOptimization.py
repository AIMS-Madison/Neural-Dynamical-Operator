import sys
sys.path.append("/home/cc/CodeProjects/NeuralDynamicalOperator")
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchdiffeq
import time
from utility import FNO_KSE, FNO_KSE_EKI, get_batch, integrate_batch_eki, acf, DKL_estimator, tke_spectrum_1d1d, kurtosis_uxx, kurtosis


device = "cuda:0"
np.random.seed(1000)
torch.manual_seed(1000)

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
u_data = torch.from_numpy(np.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Data/KS_data.npy")).to(torch.float32)
t = torch.arange(0, 5000, 0.25)
x = torch.arange(0, 22, 22/1024)


# Subsampling for different resolutions

#1. dt=0.5, dx=22/1024
# u_data = u_data[::2] # (10000, 1024)
# t = t[::2]
# Ntrain = int(10000*0.8)
# Ntest = int(10000*0.2)
# u_train = u_data[:Ntrain]
# u_test = u_data[-Ntest:]
# t_train = t[:Ntrain]
# t_test = t[:Ntest]
# batch_time = 40 #20s
# epochs = 10000
# num_batch = 2

#2. dt=1, dx=22/512
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
# epochs = 10000
# num_batch = 2

#3. dt=2, dx=22/256
u_data = u_data[::8, ::4] # (2500, 256)
t = t[::8]
x = x[::4]
dt = t[1] - t[0]
dx = x[1] - x[0]
Ntrain = int(2500*0.8)
Ntest = int(2500*0.2)
u_train = u_data[:Ntrain]
u_test = u_data[-Ntest:]
t_train = t[:Ntrain]
t_test = t[:Ntest]
batch_time = 10 # 20s
epochs = 10000
num_batch = 2


eki_u0_idx = torch.tensor([500, 1000, 1500]) # !!varying with different resolution
eki_t = torch.arange(0, 1000, 2.)

# True kurtosis
kurtosis_true_train = torch.tensor([kurtosis_uxx(u_train[k:k + 500], dx, False).item() for k in eki_u0_idx])
kurtosis_cov = torch.diag(torch.clip((kurtosis_true_train * 0.05) ** 2, 0.01, None))
kurtosis_true_test = kurtosis_uxx(u_test, dx, False).item()


###############################
### Model Training: SGD+EKI ###
###############################

# Sufficient-SGD-Training FNO
model_sgd = FNO_KSE(24, 64).to(device)
model_sgd.load_state_dict(torch.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Model/KS_model_v3.pt"))
model_eki = FNO_KSE_EKI(24, 64)
model_eki.load_state_dict(torch.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Model/KS_model_v3.pt"))


# setup
epochs = 10000  # 3000 # overwrite previous epochs
loss_training_history = []
eki_freq = 500 # 300 # ratio of epochs (sgd / eki)
num_EKI = 0

d = 3
model_param = torch.nn.utils.parameters_to_vector(model_eki.parameters()).detach()
p = len(model_param)
Nit = 10
J = 100
theta_bar_history_eki = []
error_state_history_eki = np.zeros((epochs // eki_freq, Nit+1)) # 2-D (#EKI, Nit+1)
error_kurtosis_history_eki = np.zeros((epochs // eki_freq, Nit+1)) # 2-D (#EKI, Nit+1)

def G(theta):
    """
    :param theta: tensor; FNO parameters
    :return: tensor; three kurtosis of uxx for train data
    """
    torch.nn.utils.vector_to_parameters(theta, model_eki.parameters())
    for m in model_eki.children():
        if not isinstance(m, type(model_eki.conv0)):
            with torch.no_grad():
                for param in m.parameters():
                    param.data = param.data.real
    pred1 = sp.integrate.solve_ivp(fun=model_eki, t_span=[eki_t[0], eki_t[-1]], y0=u_train[eki_u0_idx[0]], t_eval=eki_t).y.T
    pred2 = sp.integrate.solve_ivp(fun=model_eki, t_span=[eki_t[0], eki_t[-1]], y0=u_train[eki_u0_idx[1]], t_eval=eki_t).y.T
    pred3 = sp.integrate.solve_ivp(fun=model_eki, t_span=[eki_t[0], eki_t[-1]], y0=u_train[eki_u0_idx[2]], t_eval=eki_t).y.T
    kurtosis1 = kurtosis_uxx(pred1, dx, True).item()
    kurtosis2 = kurtosis_uxx(pred2, dx, True).item()
    kurtosis3 = kurtosis_uxx(pred3, dx, True).item()
    return torch.tensor([kurtosis1, kurtosis2, kurtosis3])

#SGD + EKI (3000 sgd with 10 eki)
for ep in range(eki_freq, epochs+1):
    if ep > eki_freq:
        # SGD
        # optimizer = torch.optim.Adam(model_sgd.parameters(), lr=1e-4)
        batch_u0, batch_t, batch_u = get_batch(t=t_train, u=u_train, num_batch=num_batch, batch_time=batch_time)
        batch_u0, batch_t, batch_u = batch_u0.to(device), batch_t.to(device), batch_u.to(device)
        optimizer.zero_grad()
        out = torchdiffeq.odeint(model_sgd, batch_u0, batch_t, method="rk4", options={"step_size":0.1})
        loss = F.mse_loss(batch_u, out)
        # print(torch.cuda.memory_allocated(device=device) / 1024**2)
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_training_history.append(loss.item())
        print(ep, loss.item())

    # EKI
    if ep % eki_freq == 0:
        num_EKI += 1
        print("EKI Process Start")

        model_eki = FNO_KSE_EKI(24, 64)
        model_eki.load_state_dict(model_sgd.state_dict())
        # Computing Error (Test Data, Before EKI)
        u_test_long_pred = sp.integrate.solve_ivp(fun=model_eki, y0=u_test[0], t_span=[t_test[0].item(), t_test[-1].item()], t_eval=t_test).y.T
        kurtosis_pred_test = kurtosis_uxx(u_test_long_pred, dx, True).item()
        err_kurtosis = (kurtosis_true_test - kurtosis_pred_test)**2
        error_kurtosis_history_eki[num_EKI-1, 0] = err_kurtosis
        print("long kurtosis error: ", err_kurtosis)
        err_state = integrate_batch_eki(t_test, u_test, model_eki, 10)[0]
        error_state_history_eki[num_EKI-1, 0] = err_state
        print("short state error: ", err_state)

        model_param = torch.nn.utils.parameters_to_vector(model_eki.parameters()).detach()
        ensemble_theta = model_param + torch.normal(mean=0, std=0.01, size=(J, p)) # randomize parameters from model_sgd
        ensemble_g = torch.zeros(J, d)
        for n in range(1, Nit+1):
            print("Nit: ", n)
            # 1) Ensemble Forward Mapping
            # parallel
            start_time = time.time()
            with mp.Pool(processes=10) as pool:
                res_lst = []
                for j in range(J):
                    res_lst.append(pool.apply_async(G, (ensemble_theta[j],)))
                j = 0
                for res in res_lst:
                    ensemble_g[j] = res.get()
                    j = j+1
            end_time = time.time()
            end_time - start_time
            print("Parallel Time: ", end_time - start_time)

            # 2) Updating ensemble_theta
            theta_bar = torch.zeros(p)
            g_bar = torch.zeros(d)
            c_up = torch.zeros((p, d))
            c_pp = torch.zeros((d, d))
            for j in range(J):
                theta_j = ensemble_theta[j]
                g_j = ensemble_g[j]
                # Means
                theta_bar = theta_bar + theta_j
                g_bar = g_bar + g_j
                # Covariance matrices
                c_up = c_up + np.tensordot(theta_j, g_j, axes=0) # np version tensordot
                c_pp = c_pp + np.tensordot(g_j, g_j, axes=0)
            theta_bar = theta_bar / J
            g_bar = g_bar / J
            c_up = c_up / J - np.tensordot(theta_bar, g_bar, axes=0)
            c_pp = c_pp / J - np.tensordot(g_bar, g_bar, axes=0)
            eta = np.array([np.random.multivariate_normal(np.zeros(d), kurtosis_cov) for _ in range(J)]) # float64
            kurtosis_sample = kurtosis_true_train + eta # torch.float64
            tmp = np.linalg.solve(c_pp + kurtosis_cov, np.transpose(kurtosis_sample - ensemble_g)) # float64
            tmp = torch.from_numpy(tmp).to(torch.cfloat)
            ensemble_theta += (c_up @ tmp).T

            # 3) Computing Error (Test, Inside EKI)
            theta_bar = ensemble_theta.mean(0)
            theta_bar_history_eki.append(theta_bar) # Record the theta_bar history,then pick the optimal.
            torch.nn.utils.vector_to_parameters(theta_bar, model_eki.parameters())
            for m in model_eki.children():
                if not isinstance(m, type(model_eki.conv0)):
                    with torch.no_grad():
                        for param in m.parameters():
                            param.data = param.data.real
            u_test_long_pred = sp.integrate.solve_ivp(fun=model_eki, y0=u_test[0], t_span=[t_test[0].item(), t_test[-1].item()], t_eval=t_test).y.T
            kurtosis_pred_test = kurtosis_uxx(u_test_long_pred, dx, True).item()
            err_kurtosis = (kurtosis_true_test - kurtosis_pred_test)**2
            error_kurtosis_history_eki[num_EKI-1, n] = err_kurtosis
            print("long kurtosis error: ", err_kurtosis)
            err_state = integrate_batch_eki(t_test, u_test, model_eki, 10)[0]
            error_state_history_eki[num_EKI-1, n] = err_state
            print("short state error: ", err_state)

        # Updating model_sgd parameters
        torch.nn.utils.vector_to_parameters(theta_bar.to(device), model_sgd.parameters())
        for m in model_sgd.children():
            if not isinstance(m, type(model_sgd.conv0)):
                with torch.no_grad():
                    for param in m.parameters():
                        param.data = param.data.real
        optimizer = torch.optim.Adam(model_sgd.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=eki_freq)

# torch.save(theta_bar, "/home/cc/CodeProjects/NeuralDynamicalOperator/SGD_EKI/test03/theta_eki.pt")
# with open("/home/cc/CodeProjects/NeuralDynamicalOperator/SGD_EKI/test03/loss_training_history.npy", "wb") as f:
#     np.save(f, loss_training_history)
# with open("/home/cc/CodeProjects/NeuralDynamicalOperator/SGD_EKI/test03/error_state_history_eki.npy", "wb") as f:
#     np.save(f, error_state_history_eki)
# with open("/home/cc/CodeProjects/NeuralDynamicalOperator/SGD_EKI/test03/error_kurtosis_history_eki.npy", "wb") as f:
#     np.save(f, error_kurtosis_history_eki)
# np.save('/home/cc/CodeProjects/NeuralDynamicalOperator/SGD_EKI/test03/theta_bar_history.npy', np.array(theta_bar_history_eki, dtype=object), allow_pickle=True)

####################
## Model Testing ###
####################
#
# eki_bar_history_eki = np.load("/home/cc/CodeProjects/NeuralDynamicalOperator/SGD_EKI/test02/theta_bar_history.npy", allow_pickle=True)
# error_state_history_eki = np.load("/home/cc/CodeProjects/NeuralDynamicalOperator/SGD_EKI/test02/error_state_history_eki.npy")
# error_kurtosis_history_eki = np.load("/home/cc/CodeProjects/NeuralDynamicalOperator/SGD_EKI/test02/error_kurtosis_history_eki.npy")
# loss_training_history = np.load("/home/cc/CodeProjects/NeuralDynamicalOperator/SGD_EKI/test02/loss_training_history.npy")
# eki_theta = eki_bar_history_eki[180]
#
#
#
# fig = plt.figure(figsize=(40, 10))
# axs = fig.subplots(1, 3)
# i = 0
# lst = [0, 4, 9]
# for ax in axs.flatten():
#     ax.set_title(r"\textbf{EKI " + str(lst[i]+1) + "}", fontsize=45)
#     ax.plot(error_state_history_eki[lst[i]], linewidth=3, label=r"\textbf{Short-term State Error}", color="C0", marker="o", markersize=13)
#     ax2 = ax.twinx()
#     ax2.plot(error_kurtosis_history_eki[lst[i]], linewidth=3, label=r"\textbf{Long-term Statistics Error}",color="C1", marker="s", markersize=13)
#     ax.tick_params(labelsize=40, length=15, width=3)
#     ax.set_xticks(np.arange(0,21,4))
#     ax.set_xlabel(r"\unboldmath$N_{\text{it}}$", fontsize=45)
#     ax2.tick_params(labelsize=40, length=15, width=3)
#     ax2.set_xticks(np.arange(0,21,4))
#     ax.set_ylabel(r"\textbf{Short-term MSE}", fontsize=40)
#     ax2.set_ylabel(r"\textbf{Long-term MSE}", fontsize=40)
#     ax.set_ylim([0, 1.5])
#     if i==0:
#         ax2.set_ylim([0, 4000])
#     elif i==1:
#         ax2.set_ylim([0, 2000])
#     else:
#         ax2.set_ylim([0, 100])
#     i += 1
# handle1, label1 = ax.get_legend_handles_labels()
# handle2, label2 = ax2.get_legend_handles_labels()
# handle1.extend(handle2)
# label1.extend(label2)
# lege = fig.legend(handles=handle1, labels=label1, fontsize=40, loc="upper center", ncol=2, fancybox=False, edgecolor="black")
# lege.get_frame().set_linewidth(3)
# for ax in axs.flatten():
#     for spine in ax.spines.values():
#         spine.set_linewidth(3)
# axs[2].plot(1, 0.11, color="black", marker="o", markersize=40, markerfacecolor="none", markeredgecolor="black", mew=4)
# # axs[2].scatter(1, 0.11, color="red", marker="o", s=1350, facecolor="none", edgecolor="red", linestyle="dotted", lw=5)
# fig.tight_layout()
# fig.subplots_adjust(top=0.8)
#
#
#
#
# # Pred kurtosis before SGD+EKI training
# model_eki = FNO_KSE_EKI(24, 64)
# model_eki.load_state_dict(torch.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Model/KS_model_v3.pt"))
# eki_u_train_pred1_init = sp.integrate.solve_ivp(fun=model_eki, y0=u_train[eki_u0_idx[0]], t_span=[eki_t[0], eki_t[-1]], t_eval=eki_t).y.T
# eki_u_train_pred2_init = sp.integrate.solve_ivp(fun=model_eki, y0=u_train[eki_u0_idx[1]], t_span=[eki_t[0], eki_t[-1]], t_eval=eki_t).y.T
# eki_u_train_pred3_init = sp.integrate.solve_ivp(fun=model_eki, y0=u_train[eki_u0_idx[2]], t_span=[eki_t[0], eki_t[-1]], t_eval=eki_t).y.T
# kurtosis_train_pred_init = torch.tensor([kurtosis_uxx(e, dx, True).item() for e in [eki_u_train_pred1_init, eki_u_train_pred2_init, eki_u_train_pred3_init]])
# eki_u_test_pred_init = sp.integrate.solve_ivp(fun=model_eki, y0=u_test[0], t_span=[t_test[0], t_test[-1]], t_eval=t_test).y.T
# kurtosis_test_pred_init = kurtosis_uxx(eki_u_test_pred_init, dx, True).item()
#
# # Pred kurtosis after SGD+EKI training
# kurtosis_train_pred_eki = G(eki_theta)  # model_eki will be updated in-place inside G
# eki_u_test_pred_eki = sp.integrate.solve_ivp(fun=model_eki, y0=u_test[0], t_span=[t_test[0], t_test[-1]], t_eval=t_test).y.T
# kurtosis_test_pred_eki = kurtosis_uxx(eki_u_test_pred_eki, torch.tensor([dx]), True).item()
#
#
# # long-term statistics
# (kurtosis_true_test - kurtosis_test_pred_eki)**2
# # short-term trajectory
# test_error_abs, test_error_rel = integrate_batch_eki(t_test, u_test, model_eki, 10)[:2]
#
#
#
#
# # Long-term statistics visualization
# u_xx = (u_test[:, 2:] + u_test[:, :-2] - 2*u_test[:, 1:-1]) / (dx**2)
# eki_u_test_pred_init_smoother = sp.signal.savgol_filter(sp.signal.savgol_filter(eki_u_test_pred_init, 100, 3), 100, 3)
# u_xx_init = (eki_u_test_pred_init_smoother[:, 2:] + eki_u_test_pred_init_smoother[:, :-2] - 2*eki_u_test_pred_init_smoother[:, 1:-1]) / (dx**2)
# eki_u_test_pred_eki_smoother = sp.signal.savgol_filter(sp.signal.savgol_filter(eki_u_test_pred_eki, 100, 3), 100, 3)
# u_xx_eki = (eki_u_test_pred_eki_smoother[:, 2:] + eki_u_test_pred_eki_smoother[:, :-2] - 2*eki_u_test_pred_eki_smoother[:, 1:-1]) / (dx**2)
#
# fig = plt.figure(figsize=(12, 12))
# ax = fig.subplots(1, 1)
# sns.kdeplot(u_xx.reshape(-1), ax=ax, linewidth=5, label=r"\textbf{True}")
# sns.kdeplot(u_xx_init.reshape(-1), ax=ax, linewidth=5, label=r"\textbf{Model with Classical Optimization}")
# sns.kdeplot(u_xx_eki.reshape(-1), ax=ax, linewidth=5, label=r"\textbf{Model with Hybrid Optimization")
# ax.set_xlabel(r"\unboldmath$u_{xx}$", fontsize=35)
# ax.set_ylabel("")
# ax.set_xlim([-3, 3])
# ax.set_title(r"\textbf{PDF of} \unboldmath$u_{xx}$", fontsize=30, pad=10)
# ax.tick_params(labelsize=35, length=15, width=3)
# for spine in ax.spines.values():
#     spine.set_linewidth(3)
# lege = fig.legend(fontsize=30, loc="upper center", ncol=1, fancybox=False, edgecolor="black", bbox_to_anchor=(0.52, 1))
# lege.get_frame().set_linewidth(3)
# fig.tight_layout()
# fig.subplots_adjust(top=0.77)
#
#
#
# # Profile1 (t=4000, 4005, 4010, 4020s)
# # Profile2 (t=4500, 4505, 4510, 4520s)
# # Profile2 (t=4900, 4905, 4910, 4920s)
# model_sgd_init = FNO_KSE(24, 64)
# model_sgd_init.load_state_dict(torch.load("/home/cc/CodeProjects/NeuralDynamicalOperator/Model/KS_model_v3.pt"))
# model_sgd_eki = FNO_KSE(24, 64)
# model_sgd_eki.load_state_dict(model_eki.state_dict())
# with torch.no_grad():
#     u_test_pred_4000_init = torchdiffeq.odeint(model_sgd_init, u_test[[0]], t_test[:11])[:, 0, :]
#     u_test_pred_4500_init = torchdiffeq.odeint(model_sgd_init, u_test[[250]], t_test[:11])[:, 0, :]
#     u_test_pred_4900_init = torchdiffeq.odeint(model_sgd_init, u_test[[450]], t_test[:11])[:, 0, :]
#     u_test_pred_4000_eki = torchdiffeq.odeint(model_sgd_eki, u_test[[0]], t_test[:11])[:, 0, :]
#     u_test_pred_4500_eki = torchdiffeq.odeint(model_sgd_eki, u_test[[250]], t_test[:11])[:, 0, :]
#     u_test_pred_4900_eki = torchdiffeq.odeint(model_sgd_eki, u_test[[450]], t_test[:11])[:, 0, :]
#
# fig = plt.figure(figsize=(30, 18))
# axs = fig.subplots(3, 4, sharex=True, sharey=True)
# axs[0,0].plot(x, u_test[0], linewidth=5)
# axs[0,0].set_ylabel(r"\unboldmath$u$", fontsize=50, rotation=0)
# axs[0,0].set_title(r"\unboldmath$t = 4000$", fontsize=50)
# axs[0,0].tick_params(labelsize=40, width=3, length=15)
# axs[0,1].plot(x, u_test[2], linewidth=5, label=r"\textbf{True}")
# axs[0,1].plot(x, u_test_pred_4000_init[2], linewidth=5, linestyle="dashed", label=r"\textbf{Model with Classical Optimization}")
# axs[0,1].plot(x, u_test_pred_4000_eki[2], linewidth=5, linestyle="dashed", label=r"\textbf{Model with Hybrid Optimization}")
# axs[0,1].set_title(r"\unboldmath$t = 4004$", fontsize=50)
# axs[0,1].tick_params(labelsize=40, width=3, length=15)
# axs[0,2].plot(x, u_test[5], linewidth=5)
# axs[0,2].plot(x, u_test_pred_4000_init[5], linewidth=5, linestyle="dashed")
# axs[0,2].plot(x, u_test_pred_4000_eki[5], linewidth=5, linestyle="dashed")
# axs[0,2].set_title(r"\unboldmath$t = 4010$", fontsize=50)
# axs[0,2].tick_params(labelsize=40, width=3, length=15)
# axs[0,3].plot(x, u_test[10], linewidth=5)
# axs[0,3].plot(x, u_test_pred_4000_init[10], linewidth=5, linestyle="dashed")
# axs[0,3].plot(x, u_test_pred_4000_eki[10], linewidth=5, linestyle="dashed")
# axs[0,3].set_title(r"\unboldmath$t = 4020$", fontsize=50)
# axs[0,3].tick_params(labelsize=40, width=3, length=15)
# axs[1,0].plot(x, u_test[250], linewidth=5)
# axs[1,0].set_ylabel(r"\unboldmath$u$", fontsize=50, rotation=0)
# axs[1,0].set_title(r"\unboldmath$t = 4500$", fontsize=50)
# axs[1,0].tick_params(labelsize=40, width=3, length=15)
# axs[1,1].plot(x, u_test[2+250], linewidth=5)
# axs[1,1].plot(x, u_test_pred_4500_init[2], linewidth=5, linestyle="dashed")
# axs[1,1].plot(x, u_test_pred_4500_eki[2], linewidth=5, linestyle="dashed")
# axs[1,1].set_title(r"\unboldmath$t = 4504$", fontsize=50)
# axs[1,1].tick_params(labelsize=40, width=3, length=15)
# axs[1,2].plot(x, u_test[5+250], linewidth=5)
# axs[1,2].plot(x, u_test_pred_4500_init[5], linewidth=5, linestyle="dashed")
# axs[1,2].plot(x, u_test_pred_4500_eki[5], linewidth=5, linestyle="dashed")
# axs[1,2].set_title(r"\unboldmath$t = 4510$", fontsize=50)
# axs[1,2].tick_params(labelsize=40, width=3, length=15)
# axs[1,3].plot(x, u_test[10+250], linewidth=5)
# axs[1,3].plot(x, u_test_pred_4500_init[10], linewidth=5, linestyle="dashed")
# axs[1,3].plot(x, u_test_pred_4500_eki[10], linewidth=5, linestyle="dashed")
# axs[1,3].set_title(r"\unboldmath$t = 4520$", fontsize=50)
# axs[1,3].tick_params(labelsize=40, width=3, length=15)
# axs[2,0].plot(x, u_test[450], linewidth=5)
# axs[2,0].set_xlabel(r"\unboldmath$x$", fontsize=50)
# axs[2,0].set_ylabel(r"\unboldmath$u$", fontsize=50, rotation=0)
# axs[2,0].set_title(r"\unboldmath$t = 4900$", fontsize=50)
# axs[2,0].tick_params(labelsize=40, width=3, length=15)
# axs[2,1].plot(x, u_test[2+450], linewidth=5)
# axs[2,1].plot(x, u_test_pred_4900_init[2], linewidth=5, linestyle="dashed")
# axs[2,1].plot(x, u_test_pred_4900_eki[2], linewidth=5, linestyle="dashed")
# axs[2,1].set_xlabel(r"\unboldmath$x$", fontsize=50)
# axs[2,1].set_title(r"\unboldmath$t = 4904$", fontsize=50)
# axs[2,1].tick_params(labelsize=40, width=3, length=15)
# axs[2,2].plot(x, u_test[5+450], linewidth=5)
# axs[2,2].plot(x, u_test_pred_4900_init[5], linewidth=5, linestyle="dashed")
# axs[2,2].plot(x, u_test_pred_4900_eki[5], linewidth=5, linestyle="dashed")
# axs[2,2].set_xlabel(r"\unboldmath$x$", fontsize=50)
# axs[2,2].set_title(r"\unboldmath$t = 4910$", fontsize=50)
# axs[2,2].tick_params(labelsize=40, width=3, length=15)
# axs[2,3].plot(x, u_test[10+450], linewidth=5)
# axs[2,3].plot(x, u_test_pred_4900_init[10], linewidth=5, linestyle="dashed")
# axs[2,3].plot(x, u_test_pred_4900_eki[10], linewidth=5, linestyle="dashed")
# axs[2,3].set_xlabel(r"\unboldmath$x$", fontsize=50)
# axs[2,3].set_title(r"\unboldmath$t = 4920$", fontsize=50)
# axs[2,3].tick_params(labelsize=40, width=3, length=15)
# for ax in axs.flatten():
#     for spine in ax.spines.values():
#         spine.set_linewidth(3)
# lege = fig.legend(fontsize=40, loc="upper center", ncol=4, fancybox=False, edgecolor="black", bbox_to_anchor=(0.52, 1))
# lege.get_frame().set_linewidth(3)
# fig.tight_layout()
# fig.subplots_adjust(top=0.87)
