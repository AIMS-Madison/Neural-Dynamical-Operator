import numpy as np
import matplotlib.pyplot as plt
plt.plot(1,1)
from scipy.fftpack import fft, ifft
import matplotlib



def ksintegrateNaive(u, Lx, dt, Nt, nplot):
    Nx = u.shape[0]                 # number of gridpoints
    kx = np.hstack([np.arange(0,Nx/2,1),\
                    np.array([0]), \
                    np.arange(-Nx/2+1,0,1)])  # integer wavenumbers: exp(2*pi*i*kx*x/L)
    alpha = 2.*np.pi*kx/Lx              # real wavenumbers:    exp(i*alpha*x)
    D = 1j*alpha                   # D = d/dx operator in Fourier space
    L = alpha**2 - alpha**4        # linear operator -D^2 - D^4 in Fourier space
    G = -0.5*D                      # -1/2 D operator in Fourier space
    Nplot = int(Nt/nplot) +1  # total number of saved time steps
    
    x = np.arange(0,Nx,1)*Lx/Nx
    t = np.arange(0,Nplot,1)*dt*nplot
    U = np.zeros((Nplot, Nx))
    
    # some convenience variables
    dt2  = dt/2.
    dt32 = 3.*dt/2.
    A =  np.ones(Nx) + dt2*L
    B = (np.ones(Nx) - dt2*L)**(-1)

    Nn  = G*fft(u*u) # -1/2 d/dx(u^2) = -u u_x, collocation calculation
    Nn1 = Nn

    U[0,:] = u # save initial value u to matrix U
    npl = 1     # counter for saved data
    
    # transform data to spectral coeffs 
    u  = fft(u)

    # timestepping loop
    for n in range(Nt):
        Nn1 = Nn                       # shift N^{n-1} <- N^n
        Nn  = G*fft(np.real(ifft(u))**2) # compute N^n = -u u_x

        u = B * (A * u + dt32*Nn - dt2*Nn1) # CNAB2 formula
        
        if n % nplot == 0:
            U[npl,:] = np.real(ifft(u))
            npl += 1            
    return U,x,t





Lx = 128.
Nx = 1024
dt = 1./16.
nplot = 8.
Nt = 6400
x = Lx*np.arange(0,Nx,1)/Nx

## Solving K-S equation with one initial condition
u = np.cos(x) + 0.1*np.cos(x/16.)*(1.+2.*np.sin(x/16.)) + 1e-3 * np.random.random(Nx)
U,x,t = ksintegrateNaive(u, Lx, dt, Nt, nplot)


## Solving K-S equation with another initial condition
u_new = np.cos(x) + 0.1*np.cos(x/16.)*(1.+2.*np.sin(x/16.)) + 1e-3 * np.random.random(Nx)
U_new, x_new, t_new = ksintegrateNaive(u_new, Lx, dt, Nt, nplot)



matplotlib.rcParams.update({'font.size':16})

c = plt.pcolor(t,x,U.T,cmap='jet',vmin=-4,vmax=4)
plt.colorbar(c)
plt.xlim(t[0], t[-1])
plt.ylim(x[0], x[-1])
plt.xlabel("t")
plt.ylabel("x")
plt.tight_layout()
plt.savefig('Nx_1024_T_400.png')
plt.close()

plt.plot(t, U[:, int(U.shape[1]/2)], 'b--', label = 'IC #1')
plt.plot(t, U_new[:, int(U_new.shape[1]/2)], 'r-.', label = 'IC #2')
plt.xlim([t[0],t[-1]])
plt.ylim([-4,4])
plt.legend(frameon = False)
plt.savefig('time_series_comparison.png')
plt.close()

