import os
import numpy as np
from numpy import exp, ceil, sqrt, log, pi, zeros, eye, linspace, meshgrid, trapezoid
from numpy.linalg import solve, slogdet, inv
from scipy.special import gammaln
from scipy.signal import convolve2d
import math
import scipy.io as sio
import matplotlib.pyplot as  plt
from matplotlib.patches import Ellipse
from scipy.stats import wilcoxon, ks_2samp, multivariate_normal
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import re
from tqdm import tqdm

# utils -----------------------------------------------------------

def vec(data: torch.tensor) -> torch.tensor:
    """flatten tensor in column-major order into
    a 2 dimensional tensor (N x 1)

    Args:
        data (torch.tensor): N x M array 

    Returns:
        (torch.tensor): ((N*M), 1) vector tensor
    """
    return data.T.flatten()[:,None]    


def sum2(data):
    """Sum over columns and keep 
    data dimensionality
    """
    return (data).sum(1, keepdims=True)


def create_gaussian_mask(grid_x, grid_y, mu, sigma, n_std=3):
    """
    Create a mask for points outside n standard deviations of a 2D Gaussian.
    
    Parameters
    ----------
    grid_x, grid_y : array_like
        Meshgrid coordinates
    mu : array_like
        Mean [mu_x, mu_y]
    sigma : array_like
        Covariance matrix (2x2)
    n_std : float
        Number of standard deviations
    
    Returns
    -------
    mask : array_like
        Boolean mask (True = inside, False = outside)
    """
    # Flatten grid for computation
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    
    # Calculate Mahalanobis distance
    diff = points - mu
    inv_sigma = np.linalg.inv(sigma)
    mahal_dist_sq = np.sum(diff @ inv_sigma * diff, axis=1)
    
    # Points inside n_std satisfy: mahal_dist^2 <= n_std^2
    mask = mahal_dist_sq <= n_std**2
    return mask.reshape(grid_x.shape)


def logsumexp(a: np.array, axis=0):
    """
    Computes log(sum(exp(a), axis)) while avoiding numerical underflow.
    Equivalent to MATLAB's logsumexp by Tom Minka.

    Parameters
    ----------
    a : np.ndarray
        Input array.
    axis : int, optional
        Axis along which to sum. Default is 0.

    Returns
    -------
    s : np.ndarray
        Log-sum-exp of the input along the specified axis.
    """
    # find the maximum along the specified axis (for numerical stability)
    a_max = np.max(a, axis=axis, keepdims=True)

    # subtract and exponentiate
    a_stable = a - a_max
    s = a_max + log(np.sum(np.exp(a_stable), axis=axis, keepdims=True))

    # remove the extra dimension
    s = np.squeeze(s, axis=axis)
    a_max = np.squeeze(a_max, axis=axis)

    # handle non-finite max values (like -inf)
    return np.where(np.isfinite(a_max), s, a_max)


# Dataset ------------------------------------------

def load_cell_data(cell_data_path: str):
    """Load the data for a given cell."""
    mat = sio.loadmat(cell_data_path)
    pcs = mat['X_lowd'] # principal components of natural images
    fit = mat['f']      # average firing rate evoked by the images 

    # parameters of Gaussian image statistics prior p(images)
    mu0 = pcs[0].mean()
    mu1 = pcs[1].mean()
    sigma = np.cov(pcs[0], pcs[1])    
    return pcs, fit, mu0, mu1, sigma

# 2D Poisson  -------------------------------------------


def SUM_LOG_LIST(position):
    '''Given an integer n it recursively calculates log(n!)'''
    if position == 0:
        return np.array([0])
    if position == 1:
        return np.append(SUM_LOG_LIST(0), 0)
    new_list = SUM_LOG_LIST(position-1)
    return np.append(new_list, new_list[-1]+np.around(np.log(float(position)), 8))


def POISSON_2DCELL(tc_grid, max_firing=20):
    log_list = np.tile(SUM_LOG_LIST(max_firing)[:,None,None], tc_grid.shape)
    log_tc = np.around(np.log(tc_grid), 8)#, where=(mask==1), out = np.ones_like(tc_grid)*-100)
    log_likelihood = (np.array([(i*log_tc-tc_grid) for i in range(max_firing+1)])-log_list)
    likelihood = np.exp(log_likelihood)
    likelihood = likelihood/np.sum(likelihood, axis=0)
    return likelihood    


# Neural network model of tuning curve ---------------------------


class SimpleRegressor(nn.Module):
    def __init__(self):
        super(SimpleRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):   
        return self.model(x)
    

def nnet_fit(pcs, fit, test_size=0.2, seed=42, n_epochs=200, linspace=np.linspace(-10,10,100)):
    """Fit RGCs tuning curves (PC1, PC2) with a neural network

    Args:
        pcs (_type_): principal components of the cell's receptive field
        fit (_type_): tuning curve
        test_size (float, optional): _description_. Defaults to 0.2.
        seed (int, optional): _description_. Defaults to 42.
        n_epochs (int, optional): _description_. Defaults to 200.

    Returns:
        _type_: _description_
    """
    # setup reproducibility
    torch.manual_seed(seed)

    # get the predictors and predicted data
    X = np.copy(pcs).T              # x and y
    y = np.copy(np.array(fit[0]))   # response to fit
    
    # normalize principal components
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # create train/test split
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed)

    # convert to tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32, requires_grad=True)
    X_test  = torch.tensor(X_test_np, dtype=torch.float32, requires_grad=True)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)
    y_test  = torch.tensor(y_test_np, dtype=torch.float32).view(-1, 1)

    # initialize model, loss, and optimizer
    model = SimpleRegressor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 20 == 0 or epoch == 0:
            model.eval()
            val_loss = criterion(model(X_test), y_test).item()
            print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    # inference: evaluate on a grid for visualization
    model.eval()

    # calculate grid (will be used throughout)
    grid_x, grid_y = np.meshgrid(linspace, linspace)
    latent_space = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    latent_space_scaled = scaler.transform(latent_space)

    # predict response
    with torch.no_grad():
        preds = model(torch.tensor(latent_space_scaled, dtype=torch.float32)).numpy()
        
    grad_X_tensor = torch.tensor(latent_space_scaled, dtype=torch.float32, requires_grad=True)
    output = model(grad_X_tensor)
    output.backward(torch.ones_like(output))
    grads = grad_X_tensor.grad.detach().numpy()
    return preds, model, grads, grid_x, grid_y, latent_space_scaled, latent_space


# Neural encoder model ------------------------------------

def tuning_curve_nnet(latent_space: np.array, model):
    """predict RGCs responses evoked by natural images from the
    natural image 2D latent space    

    Args:
        latent_space (np.array): latent space formed 
        ... by natural image first 2 PCs, of shape 
        (2, N_latent_points)

    Returns:
        np.array: predictions of shape (N_latent_points,)
    """
    # predict average firing rates
    with torch.no_grad():
        preds = model(torch.tensor(latent_space, dtype=torch.float32)).numpy()[:,0]
    return preds


def tuning_curve_mog(x, model=None, mu=np.array([0.3, -1.2]), 
      Sigma_f=np.array([[1, 0.2], [0.2, 0.5]]), 
      Sigma_f2=np.array([[1, -0.4], [-0.4, 0.5]]), 
      amp=20):
    """Compute 2D tuning curve for a single neuron
    as a mixture of two Gaussians.

    The tuning curve models neural responses as a function of two stimulus dimensions
    (e.g., natural image principal components). It combines two Gaussian bumps with
    different centers and covariance structures, plus a baseline firing rate.    

    Args:
        x (np.array): Stimulus coordinates of shape (N, 2), where N is the number of points
        ...and the two columns represent PC1 and PC2 (or any 2D feature space).
        mu (np.array, optional): Center of the second Gaussian component, shape (2,). 
        ...Defaults to np.array([0.3, -1.2]).
        Sigma_f (np.array, optional): Covariance (2x2) for the first Gaussian 
        ...component centered at origin. Controls the shape and orientation of 
        ...the first tuning bump. 
        ...Defaults to np.array([[1, 0.2], [0.2, 0.5]]).
        Sigma_f2 (np.array, optional): Covariance matrix (2x2) for the second 
        ...Gaussian component centered at mu. Controls the shape and orientation 
        ...of the second tuning bump. 
        ...Defaults to np.array([[1, -0.4], [-0.4, 0.5]]). 
        amp (int, optional): Maximum amplitude (peak firing rate) 
        ...of the tuning curve in spikes/second. 
        ...Defaults to 20.

    Returns:
        np.array: Mean firing rates at each stimulus location (tuning curve), shape (N,).
        Values represent spikes/second, with a baseline of 2 spikes/s.
    """
    return amp * (exp(-0.5 * sum(x.T * solve(Sigma_f,x.T))) + exp(-0.5 * sum((x-mu).T * solve(Sigma_f2, (x-mu).T)))) / 2 + 2


def get_tuning_curve_grad(X, tuning_curve_fun, model=None, eta=1e-4):
    """compute the gradient of the tuning curve
    with finite difference approximation

    Args:
        X (_type_): latent space points
        tuning_curve_fun (_type_): tuning curve function
        eta (_type_, optional): _description_. Defaults to 1e-4.

    Returns:
        _type_: gradient of tuning curve
    """
    # tuning curve
    tc = tuning_curve_fun(X, model=model)

    # compute numerical derivative, nabla_x f
    f1 = tuning_curve_fun(X + np.array([eta, 0]), model=model) # of shape (1, nx**2) # OK
    f2 = tuning_curve_fun(X + np.array([0, eta]), model=model) # of shape (1, nx**2) # OK
    return np.vstack([f1-tc, f2-tc]) / eta # of shape (2, nx**2)


#  Plots -------------------------------------------------

def scatter_hist(x, y, ax, ax_histx, ax_histy, c = 'tab:blue', alpha = 1, htype = 'step'):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y,s=1, color=c)

    # now determine nice limits by hand:
    binwidth = 0.2
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    print(xymax)
    lim = (np.rint((xymax+1)/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axx = ax_histx.hist(x, bins=bins, color=c)
    ax_histx.set_ylim(0,(axx[0].max()//100+1)*100)
    axy = ax_histy.hist(y, bins=bins, orientation='horizontal', color=c)
    ax_histy.set_xlim(0,(axy[0].max()//100+1)*100)


def plot_gaussian_ellipse(mean, cov, ax, n_std=1.0, **kwargs):
    
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellipse)
    return ax


# Information metrics -------------------------------------------------------

def compute_ssi_and_bayes_mse_via_method1(pcs, grid_x, grid_y, tuning_curves, prior_sigma, latent_space, load=True, metrics_data_path=''):

    # compute prior
    mus = pcs[:2].mean(axis=1)
    prior = multivariate_normal(mus, prior_sigma)
    grid_prior = prior.pdf(latent_space).reshape(grid_x.shape)/prior.pdf(latent_space).reshape(grid_x.shape).sum()
    prior_entropy = -np.sum(grid_prior*np.around(np.log2(grid_prior), 8))

    # compute likelihood
    likelihood = POISSON_2DCELL(tuning_curves)

    # compute posterior
    evidence = np.sum(likelihood*np.tile(grid_prior[None,:,:], (likelihood.shape[0],1,1)), axis=(-1,-2))
    posterior = likelihood*np.tile(grid_prior[None,:,:], (likelihood.shape[0],1,1))/\
                            np.tile(evidence[:,None,None], (1,*likelihood.shape[1:])) 
    posterior[posterior==0] = 1e-50
    posterior = posterior/np.tile(posterior.sum(axis=(1,2))[:,None,None], (1, *grid_x.shape))

    # computing is slow for 300 x 300 grid (> 1 hour)
    if not load:

        # compute SSI
        post_entropy = -np.sum(posterior*np.around(np.log2(posterior),8), axis=(1,2))
        ssi = prior_entropy - np.sum(likelihood * np.tile(post_entropy[:,None,None], (1, *grid_x.shape)) ,axis=0)

        # compute Bayes error
        mse = np.empty_like(posterior[0])
        for i in range(likelihood.shape[1]):
            for j in range(likelihood.shape[2]):
                post_ij = np.sum(posterior * np.tile(likelihood[:,i,j][:,None,None], (1, *likelihood.shape[1:])), axis=0)
                delta = (grid_x - grid_x[i,j])**2+(grid_y-grid_y[i,j])**2
                mse[i,j] = np.sum(post_ij*delta)
        rmse = np.sqrt(mse)
        print("Computed SSI and Bayes error.")

    else: 

        # load precomputed metrics
        out = np.load(metrics_data_path)
        ssi = out['ssi']
        mse = out['mse']
        rmse = np.sqrt(out['mse'])
        print("mse shape: ", mse.shape)
        print("grid shape: ", out['grid_x'].shape)
        print("Loaded precomputed data:", out.keys())

    # compute SSI bound
    ssi_bound = 2**(prior_entropy - ssi-2*np.log2(50))/(np.sqrt(2*np.pi*math.e))
    return ssi, ssi_bound, mse, rmse


def compute_ssi(latent_space, firingrate, amp=20, Sigmax=np.array([[1,0.25], [0.25,1]])):
    """Compute SSI for Gaussian stimulus distribution (prior) with zero mean 
    and given covariance matrix

    Args:
        X (_type_): _description_
        firingrate (_type_): _description_
        amp (int, optional): _description_. Defaults to 20.
        Sigmax (_type_, optional): _description_. Defaults to np.array([[1,0.25], [0.25,1]]).

    Returns:
        _type_: _description_
    """
    # recover the ordered values x the principal components 
    # can take in the latent space
    pc_values = np.unique(latent_space)
    dx = pc_values[1] - pc_values[0]            # step size
    num_x = int(sqrt(latent_space.shape[0]))    # number of points along each dimension
    
    # Calculate log probability: log p(x) for multivariate Gaussian
    # log p(x) = -0.5 * (x^T * Sigma^(-1) * x) - 0.5 * log(det(2*pi*Sigma))
    # - inverse of covariance matrix
    Sigmax_inv = np.linalg.inv(Sigmax)                            

    # - Mahalanobis distance term: (latent_space^T * Sigma^(-1) * x)
    # For multiple points in latent_space (shape: 2 x N), compute for each column
    mahal_dist = np.sum(latent_space.T * (Sigmax_inv @ latent_space.T), axis=0)
    logpx = -0.5 * mahal_dist - 0.5 * np.linalg.slogdet(2 * np.pi * Sigmax)[1]

    # setup space of possible spike counts
    r = np.arange(0, amp + ceil(sqrt(amp*5))+1)[None] # of shape (n_neuron=1,31)

    # log p(r|x) of shape (num_x**2, r.shape[1])
    logpr_x =  log( vec(firingrate) )*r - vec(firingrate) - sum(gammaln(r+1))

    # log p(r,x) of shape (num_x**2, r.shape[1])
    logprx = logpr_x + vec(logpx)

    # log p(r) of shape (n_neurons=1, r.shape[1])
    logpr = logsumexp(logprx + log(dx**2), 0)[None]

    # log p(x|r) of shape (num_x**2, r.shape[1])
    logpx_r = logpr_x + vec(logpx) - logpr

    # H(X|r) of shape (n_neurons=1,31)
    HX_r = (-sum( exp(logpx_r) * logpx_r) * dx**2)[None]

    # SSI = H(X) - <H(X|r)>_p(r|x) of shape (num_x, num_x)
    # for a stimulus distribution with zero mean Gaussian with covariance matrix 
    ssi = 0.5 * slogdet(2*pi*exp(1)*Sigmax)[1] - HX_r @ exp(logpr_x).T
    ssi = ssi.reshape(num_x,num_x).T

    # I(R;X), a scalar
    Inf_ssi = vec(ssi).T @ exp(vec(logpx)) * dx**2
    return ssi, Inf_ssi, logpx, logpr_x, logpr


def compute_ilocal(latent_space: np.array,  tuning_curve_model: SimpleRegressor, 
                   sigma=np.array([[1.,-0.07466075],[-0.07466075,1.]]),
                   ny=101, ngamma=20,
                   gamma0=0.01, gamma_max=200, eta=1e-4, amp=20):
    
    # recover the ordered values the principal components can take in the latent space
    pc_values = np.unique(latent_space)
    dx = pc_values[1] - pc_values[0] # step size

    # number of points along each image latent dimension
    num_x = np.sqrt(latent_space.shape[0]).astype(int)

    # get latent space parameters
    xmax = latent_space.max()

    # Use Fisher information for low gamma values
    # Ilocal0 =  int J_gamma(y) dgamma, from gamma_0 to gamma
    Ilocal0 = zeros((num_x**2, 1))

    # predict firing rates at image latent space points (2D tuning curve)
    rate_preds = tuning_curve_nnet(latent_space, tuning_curve_model)

    # compute the gradient of the tuning curve
    # df = get_tuning_curve_grad(latent_space, tuning_curve_mog, eta=1e-4)                     # for toy MOG model
    df = get_tuning_curve_grad(latent_space, tuning_curve_nnet, tuning_curve_model, eta=1e-4)  # for fitted RGC responses

    # Use Fisher information for low gamma values
    for i in range(num_x**2):
        J = df[:,i][:,None] * df[:,i][:,None].T / rate_preds[i] # Fisher
        Ilocal0[i] = 0.5 * slogdet(eye(2) + J * gamma0 * sigma / (gamma0*eye(2)+sigma))[1]   #Ilocal0
    Ilocal0 = Ilocal0.reshape(num_x,num_x).T

    # set the number of gamma noise scales to loop through
    gamma = exp(linspace(log(gamma0), log(gamma_max), ngamma))

    # Jx = <J_gamma(y)>_p(y|x)
    meanJry = zeros((num_x**2, ngamma))

    # initialize ilocal
    Inf = zeros((ngamma, ngamma))

    # Integrate I_local over noise scales
    for igamma in tqdm(range(len(gamma)), 'Computing ilocal'):
        
        # setup noisy stimulus latent space adapted depending on gamma
        # discretisation of y = x + sqrt(gamma) * noise
        ymax = xmax + 5 * sqrt(gamma[igamma])
        y = linspace(-ymax, ymax, ny)
        dy = y[1] - y[0]
        [ycord1, ycord2] = meshgrid(y, y)
        Y = np.hstack([vec(ycord1), vec(ycord2)])
        
        # neural responses in this new space
        rate_preds = tuning_curve_nnet(Y, tuning_curve_model).reshape(ny,ny)

        # log p(x) in this new coordinate space
        logpx = -0.5 * sum(Y.T * solve(sigma, Y.T)).reshape(ny,ny).T - 0.5 * slogdet(2*pi*sigma)[1]

        # p(y|x=0), gaussian filter for convolution
        phi = exp(-0.5*(ycord1**2 + ycord2**2)/gamma[igamma]) / (2*pi*gamma[igamma])

        # p(y|x=eta) same as above, but increment x, for numerical derivative
        phi1 = exp(-0.5*((ycord1+eta)**2+ycord2**2)/gamma[igamma]) / (2*pi*gamma[igamma])
        phi2 = exp(-0.5*(ycord1**2+(ycord2+eta)**2)/gamma[igamma]) / (2*pi*gamma[igamma])

        # convolve p(x) with phi to get p(y)
        logpy = log( convolve2d(exp(logpx), phi, 'same')*(dy**2))

        # same thing, but with incremented x, for numerical derivative
        logpy1 = log( convolve2d(exp(logpx), phi1, 'same')*(dy**2))
        logpy2 = log( convolve2d(exp(logpx), phi2, 'same')*(dy**2))

        # J_R(y)
        Jry = zeros((ny, ny))
        
        # loop over mean spike counts
        for r in range(int(amp + ceil(sqrt(amp*5)) + 1)): 

            # log p(r|x) - the log likelihood
            logpr_x = log(rate_preds)*r - rate_preds - gammaln(r+1)

            # p(r,x)
            prx = exp(logpr_x + logpx)

            # convolve p(r,x) with phi(x) to get log p(r,y), then subtract
            # logp(y) to get log p(r|y)
            logpr_y = log( convolve2d(prx, phi, 'same')*(dy**2)) - logpy

            # same as above but with incremented x, for numerical derivative
            logpr_y1 = log( convolve2d(prx, phi1, 'same')*(dy**2)) - logpy1
            logpr_y2 = log( convolve2d(prx, phi2, 'same')*(dy**2)) - logpy2

            # Jry = <(d logpr_y/dy)^2>_{p(r|y)} - sum over p(r|y)
            Jry = Jry + exp(logpr_y) * ((logpr_y2 - logpr_y)**2 + (logpr_y1 - logpr_y)**2)/(eta**2)

        # p(y|x) - to go back to space of 'latent_space' of shape (num_x**2, 10201)
        py_x = exp(-0.5*(sum2(latent_space**2) - 2*latent_space @ Y.T + sum2(Y**2).T)/gamma[igamma])/(2*pi*gamma[igamma]) # Ok!

        # meanJry_gamma(x) = <J_gamma(y)>_p(y|x) of shape (num_x**2, len(gamma))
        meanJry[:, igamma] = (py_x @ vec(Jry) * (dy**2)).flatten()

        # take numerical integral over gamma, to get Ilocal
        if igamma > 0:
            Ilocal = 0.5 * trapezoid(y=meanJry[:,:igamma+1], x=gamma[:igamma+1], axis=1).reshape(num_x,num_x).T + Ilocal0 # OK!
        else:
            Ilocal = Ilocal0

        # estimates
        logpx = -0.5*sum(latent_space.T * solve(sigma, latent_space.T)) - 0.5*slogdet(2*pi*sigma)[1] # of shape (num_x**2,1) # Ok!
        Inf[igamma] = vec(Ilocal).T @ exp(vec(logpx)) * (dx**2)  # of shape (len(gamma))
    return Ilocal, Inf, gamma

# unit-testing --------------------------------------------------

def test_f(x):

    # x points to plot, from -xmax to xmax
    nx = 100
    xmax = 10
    x = np.linspace(-xmax, xmax, nx)[None] # of shape (1,100)
    dx = x[0,1] - x[0,0]

    # transform into 2D stimulus (capitalized to show 2D)
    xtemp1, xtemp2 = np.meshgrid(x, x)
    X = np.column_stack((vec(xtemp1), vec(xtemp2))) # of shape (nx**2, 2)    

    # Parameters
    amp = 20                         # maximum firing rate
    Sigma_f = np.array([[1, 0.2], 
                        [0.2, 0.5]])
    Sigma_f2 = np.array([[1, -0.4], 
                        [-0.4, 0.5]])
    mu = np.array([0.3, -1.2])

    # test
    assert np.allclose(np.exp(-0.5 * sum(X.T * solve(Sigma_f, X.T))).mean(), 0.0104, atol=1e-4), "wrong"
    assert np.allclose(tuning_curve_mog(X, mu, Sigma_f, Sigma_f2, amp).mean(), 2.1942)


def test_ssi(ssi, inf_ssi, logpx, logpr_x, logpr):

    assert np.allclose(logpx[:10], np.array([-81.80560781, -80.21121268, -78.66035052, -77.15302134,
    -75.68922512, -74.26896189, -72.89223162, -71.55903433,
    -70.26937001, -69.02323866])), "wrong logpx"

    assert np.allclose(logpr_x[:3,:3], np.array([[-2.        , -1.30685282, -1.30685282],
    [-2.        , -1.30685282, -1.30685282],
    [-2.        , -1.30685282, -1.30685282]])), "wrong logpr_x"

    assert np.allclose(logpr, np.array([ -4.004177  ,  -3.0262503 ,  -2.6462648 ,  -2.540976  ,
            -2.55821745,  -2.61336901,  -2.66757795,  -2.71162076,
            -2.74981398,  -2.79006207,  -2.84012527,  -2.90677374,
            -2.99569149,  -3.11148839,  -3.25774536,  -3.43710368,
            -3.65139045,  -3.90175929,  -4.18882754,  -4.51279853,
            -4.87356435,  -5.27078859,  -5.7039714 ,  -6.17249935,
            -6.67568333,  -7.21278685,  -7.78304701,  -8.38568994,
            -9.01994198,  -9.68503768, -10.38022546])), "Wrong logpr"

    # SSI
    assert np.allclose(ssi.mean(), -0.1210, atol=1e-4), "wrong SSI"
    assert np.allclose(inf_ssi, 0.5991, atol=1e-4), "wrong Inf_ssi"


