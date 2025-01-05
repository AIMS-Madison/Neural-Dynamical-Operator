# 1) The code relevant to FNO are from the paper "Fourier Neural Operator for Parametric Partial Differential Equations" and tweaked by the author in this work.
# 2) The code relevant to Energy Spectrum is from the link "https://turbulence.utah.edu/codes/turbogenpython/tkespec.py" and tweaked by the author in this work.
# 3) The DKL_estimator function is from the paper "Divergence estimation for multidimensional densities via k-Nearest-Neighbor distances."


import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchdiffeq


# FNO1D
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x
class FNO_VBE(nn.Module):
    def __init__(self, modes, width):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width*2)  # output channel_dim is 1: u1(x)

    def forward(self, t, x):
        x = x.reshape(x.shape[0], x.shape[1], 1) # (N, X) --> (N, X, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 1)

        x = x.reshape(x.shape[0], x.shape[1]) # (N, X, 1) --> (N, X)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
class FNO_KSE(nn.Module):
    def __init__(self, modes, width):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width*2)  # output channel_dim is 1: u1(x)

    def forward(self, t, x):
        x = x.reshape(x.shape[0], x.shape[1], 1) # (N, X) --> (N, X, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 1)

        x = x.reshape(x.shape[0], x.shape[1]) # (N, X, 1) --> (N, X)

        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 127.8750, size_x), dtype=torch.float) ###################### modification of grid length
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

class FNO_KSE_EKI(nn.Module):
    def __init__(self, modes, width):
        super().__init__()

        self.modes1 = modes
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width*2)  # output channel_dim is 1: u1(x)

    def forward(self, t, x):
        with torch.no_grad():
            x = x.reshape(1, -1) # array(X) --> array(1, X)
            x = torch.tensor(x, dtype=torch.float32) # --> array(1, x) --> tensor(1, x)
            x = x.reshape(x.shape[0], x.shape[1], 1) # tensor(1, X) --> tensor(1, X, 1)
            grid = self.get_grid(x.shape, x.device)
            x = torch.cat((x, grid), dim=-1)
            x = self.p(x)
            x = x.permute(0, 2, 1)
            # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

            x1 = self.conv0(x)
            x1 = self.mlp0(x1)
            x2 = self.w0(x)
            x = x1 + x2
            x = F.gelu(x)

            x1 = self.conv1(x)
            x1 = self.mlp1(x1)
            x2 = self.w1(x)
            x = x1 + x2
            x = F.gelu(x)

            x1 = self.conv2(x)
            x1 = self.mlp2(x1)
            x2 = self.w2(x)
            x = x1 + x2
            x = F.gelu(x)

            x1 = self.conv3(x)
            x1 = self.mlp3(x1)
            x2 = self.w3(x)
            x = x1 + x2

            # x = x[..., :-self.padding] # pad the domain if input is non-periodic
            x = self.q(x)
            x = x.permute(0, 2, 1)

            x = x.reshape(x.shape[0], x.shape[1]) # tensor(1, X, 1) --> tensor(1, X)

        return x.reshape(-1).numpy() # tensor(1, X) --> array(X)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 127.8750, size_x), dtype=torch.float) ###################### modification of grid length
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# FNO2D
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
class FNO_NSE(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9
        self.fc0 = nn.Linear(3, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, t, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1) # (N, X, Y) --> (N, X, Y, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x.reshape(x.shape[0], x.shape[1], x.shape[2]) # (N, X, Y, 1) --> (N, X, Y)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


# Energy Spectrum
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')
def tke_spectrum_1d1d(u, Lx=1, smooth=True):
    n = len(u)
    uh = np.fft.fftn(u)/n
    tkeh = 0.5*(uh*np.conj(uh)).real

    knorm = 2.0 * np.pi / Lx
    wave_numbers = knorm*np.arange(0, n)

    tke_spectrum = np.zeros(len(wave_numbers))
    for kx in range(n):
        rkx = kx
        if (kx > n/2):
            rkx = rkx - n
        rk = np.sqrt(rkx*rkx)
        k = int(np.round(rk))
        tke_spectrum[k] = tke_spectrum[k] + tkeh[kx]

    if smooth:
        tkespecsmooth = movingaverage(tke_spectrum, 5) #smooth the spectrum
        tkespecsmooth[0:4] = tke_spectrum[0:4] # get the first 4 values from the original data
        tke_spectrum = tkespecsmooth

    knyquist = knorm*n/2

    return knyquist, wave_numbers, tke_spectrum

def tke_spectrum_2d2d(u,v,lx=1,ly=1,smooth=True):
    nx = len(u[:,0])
    ny = len(v[0,:])
    nt = nx*ny
    n = nx #int(np.round(np.power(nt,1.0/3.0)))
    uh = np.fft.fftn(u)/nt
    vh = np.fft.fftn(v)/nt

    tkeh = 0.5*(uh*np.conj(uh) + vh*np.conj(vh)).real

    k0x = 2.0*np.pi/lx
    k0y = 2.0*np.pi/ly
    knorm = (k0x + k0y)/2.0

    kxmax = nx/2
    kymax = ny/2

    wave_numbers = knorm*np.arange(0,n)

    tke_spectrum = np.zeros(len(wave_numbers))

    for kx in np.arange(nx):
        rkx = kx
        if (kx > kxmax):
            rkx = rkx - (nx)
        for ky in np.arange(ny):
            rky = ky
            if (ky>kymax):
                rky=rky - (ny)
            rk = np.sqrt(rkx*rkx + rky*rky)
            k = int(np.round(rk))
            tke_spectrum[k] = tke_spectrum[k] + tkeh[kx,ky]

    tke_spectrum = tke_spectrum/knorm
    #  tke_spectrum = tke_spectrum[1:]
    #  wave_numbers = wave_numbers[1:]
    if smooth:
        tkespecsmooth = movingaverage(tke_spectrum, 5) #smooth the spectrum
        tkespecsmooth[0:4] = tke_spectrum[0:4] # get the first 4 values from the original data
        tke_spectrum = tkespecsmooth

    knyquist = knorm*min(nx,ny)/2

    return knyquist, wave_numbers, tke_spectrum



# ACF & DKL
def acf(x, lag=200):
    i = np.arange(0, lag+1)
    v = np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1]  for i in range(1, lag+1)])
    return (i, v)

def DKL1d(data_p, data_q):
    kde_p = sp.stats.gaussian_kde(data_p)
    kde_q = sp.stats.gaussian_kde(data_q)
    grid_points = np.linspace(min(data_p.min(), data_q.min()), max(data_p.max(), data_q.max()), 1000)
    pdf_p = kde_p(grid_points)
    pdf_q = kde_q(grid_points)
    epsilon = 1e-10
    kl_values = sp.special.kl_div(pdf_p, pdf_q + epsilon)
    kl_total = np.trapz(kl_values, grid_points)
    return kl_total


def DKL_estimator(s1, s2, k=1):
    """KL-Divergence estimator using scipy's KDTree
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)
    """
    # Expects [N, D]
    assert len(s1.shape) == len(s2.shape) == 2
    # Check dimensionality of sample is identical
    assert s1.shape[1] == s2.shape[1]

    n, m = len(s1), len(s2)
    d = float(s1.shape[1])
    D = np.log(m / (n - 1))

    nu_d, nu_i = sp.spatial.KDTree(s2).query(s1, k)
    rho_d, rhio_i = sp.spatial.KDTree(s1).query(s1, k + 1)

    # KTree.query returns different shape in k==1 vs k > 1
    if k > 1:
        D += (d / n) * np.sum(np.log(nu_d[::, -1] / rho_d[::, -1]))
    else:
        D += (d / n) * np.sum(np.log( (nu_d+1e-9) /  (rho_d[::, -1]+1e-9)  ))
    return D


# Training/Testing Time Series Batches
def get_batchs(u, t, num_batchs, batch_steps):
    Nt = u.shape[0]
    heads_idx = np.random.choice(Nt-batch_steps+1, size=num_batchs, replace=False)
    batch_u0 = u[heads_idx]
    batch_t = t[:batch_steps]
    batch_u = torch.stack( [u[heads_idx+i] for i in range(batch_steps)] )
    return batch_u0, batch_t, batch_u

def integrate_batch(t, u, model, batch_time):
    device = u.device
    data_size = u.shape[0]
    num_batch = int(data_size / batch_time)
    state_error_abs = 0
    state_error_rel = 0
    stt_pred = torch.tensor([]).to(device)
    for i in range(num_batch):
        stt_batch = u[i*batch_time: (i+1)*batch_time]
        with torch.no_grad():
            stt_pred_batch = torchdiffeq.odeint(model, stt_batch[[0]], t[:batch_time])
        stt_pred_batch = stt_pred_batch.reshape(batch_time, -1)
        stt_pred = torch.cat([stt_pred, stt_pred_batch])
        state_error_abs += torch.mean( (stt_batch - stt_pred_batch)**2 ).item()
        state_error_rel += torch.mean( torch.norm(stt_batch - stt_pred_batch, 2, 1) / (torch.norm(stt_batch, 2, 1)) ).item()
    state_error_abs /= num_batch
    state_error_rel /= num_batch
    return [state_error_abs, state_error_rel, stt_pred]

def integrate_batch_eki(t, u, model_eki, batch_time):
    """
    :param t: tensor (t)
    :param u: tensor (t, x)
    :param model_eki: FNO model (solve_ivp version)
    :param batch_time: time steps (not physical time)
    :return:
    """
    u = u.cpu().numpy()
    t = t.cpu().numpy()
    data_size = u.shape[0]
    num_batch = int(data_size / batch_time)
    t_batch = t[:batch_time]
    state_error_abs = 0
    state_error_rel = 0
    # u_pred = np.array([])
    for i in range(num_batch):
        u_batch = u[i*batch_time: (i+1)*batch_time]
        u_batch_pred = sp.integrate.solve_ivp(fun=model_eki, y0=u_batch[0], t_span=[t_batch[0], t_batch[-1]], t_eval=t_batch).y.T
        # u_pred = np.concatenate([u_pred, u_batch_pred])
        state_error_abs += np.mean((u_batch - u_batch_pred)**2)
        state_error_rel += np.mean( (u_batch - u_batch_pred)**2 / (1+np.abs(u_batch)) )
    state_error_abs /= num_batch
    state_error_rel /= num_batch
    return [state_error_abs, state_error_rel]

def predict_short_relay(u, t, model, short_steps):
    u0_batch = u[::short_steps]
    with torch.no_grad():
        u_shortPred = torchdiffeq.odeint(model, u0_batch, t[:short_steps])
    u_shortPred = u_shortPred.permute(1,0,2).reshape(-1, u.shape[1])
    return u_shortPred









def solve_poisson_equation_2d_periodic(f, L, N):
    h = L / N  # Grid spacing
    # Compute wave numbers
    kx = 2 * np.pi * np.fft.fftfreq(N, h)
    ky = 2 * np.pi * np.fft.fftfreq(N, h)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    # Compute the Fourier transform of the source term
    f_hat = np.fft.fftn(f)
    # Exclude the DC component
    KX[0, 0] = 1.0
    KY[0, 0] = 1.0
    # Compute the solution in Fourier space
    u_hat = -f_hat / (KX ** 2 + KY ** 2)
    # Zero out the DC component
    u_hat[0, 0] = 0.0
    # Compute the inverse Fourier transform to obtain the solution in physical space
    u = np.real(np.fft.ifftn(u_hat))
    return u
def stream2velocity(stream, dx=1/64, dy=1/64):
    u = (stream[2:] - stream[:-2]) / (2*dy)
    v = -(stream[:, 2:] - stream[:, :-2]) / (2*dx)
    u = u[:, 1:-1]
    v = v[1:-1]
    u = u - np.mean(u)
    v = v - np.mean(v)
    return u, v


def var_ux(u, dx, smoother=False):
    if smoother == True:
        u = torch.fft.irfft( torch.fft.rfft(u)[:, :8], n=u.shape[1])
    ux = (u[:, 2:] - u[:, :-2]) / (2*dx)
    return  torch.var(ux)

def var_uxx(u, dx, smoother=False):
    if smoother == True:
        u = torch.fft.irfft( torch.fft.rfft(u)[:, :8], n=u.shape[1])
    uxx = (u[:, 2:] + u[:, :-2] - 2*u[:, 1:-1]) / (dx**2)
    return  torch.var(uxx)


def kurtosis(x):
    x_normalized = (x - torch.mean(x)) / torch.std(x)
    x_kurtosis = torch.mean(x_normalized**4) - 3.0
    return x_kurtosis

def kurtosis_ux(u, dx, smoother=False):
    if smoother == True:
        u = torch.fft.irfft( torch.fft.rfft(u)[:, :8], n=u.shape[1])
    ux = (u[:, 2:] - u[:, :-2]) / (2*dx)
    return  kurtosis(ux)

def kurtosis_uxx(u, dx, smoother=False):
    if smoother == True:
        u = torch.fft.irfft( torch.fft.rfft(u)[:, :8], n=u.shape[1])
    uxx = (u[:, 2:] + u[:, :-2] - 2*u[:, 1:-1]) / (dx**2)
    return  kurtosis(uxx)


def kurtosis_uxx_Fourier(u, dx, smoother=False):
    k = torch.fft.fftfreq(u.shape[1], d=dx)
    u_hat = torch.fft.fft(u)
    if smoother == True:
        s = 0.23
        u_hat = u_hat * torch.exp(-0.5 * (k/s)**2)
    uxx_hat = - (2 * np.pi * k) ** 2 * u_hat
    uxx = torch.fft.ifft(uxx_hat, n=u.shape[1]).real
    return uxx



def FNO_get_params(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()

def FNO_set_params(model, params):
    device = next(model.parameters()).device
    torch.nn.utils.vector_to_parameters(params, model.parameters())
    for m in model.children():
        if not isinstance(m, type(model.conv0)):
            with torch.no_grad():
                for param in m.parameters():
                    param.data = param.data.real
    model.to(device)



def cross_cov(X, Y):
    device = X.device
    J = X.shape[0]
    assert J == Y.shape[0]
    assert device == Y.device
    p, d = X.shape[1], Y.shape[1]
    X_mean = torch.zeros(p).to(device)
    Y_mean = torch.zeros(d).to(device)
    CCOV = torch.zeros(p, d).to(device)
    for j in range(J):
        X_mean = X_mean + X[j]
        Y_mean = Y_mean + Y[j]
        CCOV = CCOV + torch.outer(X[j], Y[j])
    X_mean = X_mean/J
    Y_mean = Y_mean/J
    CCOV = CCOV/J - torch.outer(X_mean, Y_mean)
    return CCOV

