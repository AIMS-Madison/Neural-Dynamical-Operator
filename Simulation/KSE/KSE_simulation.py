import numpy as np
import scipy as sp


#Simulation Setting: Lx = 22, Lt = 5000, dx=22/1024, dt = 0.025
def ksintegrateNaive(u, Lx, dt, Nt, nplot):
    Nx = u.shape[0]                 # number of gridpoints
    kx = np.hstack([np.arange(0,Nx/2,1), \
                    np.array([0]), \
                    np.arange(-Nx/2+1,0,1)])  # integer wavenumbers: exp(2*pi*i*kx*x/L)
    alpha = 2.*np.pi*kx/Lx              # real wavenumbers:    exp(i*alpha*x)
    D = 1j*alpha                   # D = d/dx operator in Fourier space
    L = alpha**2 - alpha**4        # linear operator -D^2 - D^4 in Fourier space
    G = -0.5*D                      # -1/2 D operator in Fourier space
    Nplot = int(Nt/nplot) +1  # total number of saved time steps

    x = np.arange(0,Nx,1) * Lx/Nx
    t = np.arange(0,Nplot,1)*dt*nplot
    U = np.zeros((Nplot, Nx))

    # some convenience variables
    dt2  = dt/2.
    dt32 = 3.*dt/2.
    A =  np.ones(Nx) + dt2*L
    B = (np.ones(Nx) - dt2*L)**(-1)

    Nn  = G * sp.fftpack.fft(u*u) # -1/2 d/dx(u^2) = -u u_x, collocation calculation
    Nn1 = Nn

    U[0,:] = u # save initial value u to matrix U
    npl = 1     # counter for saved data

    # transform data to spectral coeffs
    u = sp.fftpack.fft(u)

    # timestepping loop
    for n in range(Nt):
        Nn1 = Nn                       # shift N^{n-1} <- N^n
        Nn  = G * sp.fftpack.fft(np.real(sp.fftpack.ifft(u))**2) # compute N^n = -u u_x

        u = B * (A * u + dt32*Nn - dt2*Nn1) # CNAB2 formula

        if n % nplot == 0:
            U[npl,:] = np.real(sp.fftpack.ifft(u))
            npl += 1
    return U, x, t



Lx = 22
Nx = 1024
x = Lx*np.arange(0, Nx, 1)/Nx
Lt = 5000
dt = 0.025
Nt = int(Lt/dt)
nplot = 10
u0 = 0.1*np.cos(x/16.)*(1.+2.*np.sin(x/16.))


u_data, x, t = ksintegrateNaive(u0, Lx, dt, Nt, nplot)
u_data = u_data[:-1]
t = t[:-1]


# with open("KSE_data.npy", "wb") as f:
#     np.save(f, u_data)



