"""Plot a figure 3 panel for each RGC cell contained in data/contrast_cells/

author: laquitainesteeve@gmail.com based on code from Carlo Paris & Matthew Chalk

Usage:
    
    # setup conda environment, activate and run pipeline
    conda activate envs/fisher_info_limits2
    nohup python src/pipes/fig3.py

Execution time: 3 min per cell on CPU

Tested on an Ubuntu 24.04.1 LTS (32 cores, 188 GB RAM, Intel(R) Core(TM) i9-14900K @3.2 GHz/5.8 GHz)

Returns:
    .svg: figures of RGC cells fig3 panels in ./figures/all/
"""
# import packages
import os 
import sys
from matplotlib import pyplot as plt;
import scipy.io as sio
import numpy as np
from numpy import log
import math
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
import argparse
import torch.nn as nn

# set project path
proj_path = '/home/steeve/steeve/idv/code/fisher-info-limits'
os.chdir(proj_path)

# add custom package to path
sys.path.append('.')

# import custom package
from src import nodes

# setup ilocal parmaeters
SEED = 10
ngamma = 20     # default=100;
gamma0 = 0.01
gamma_max = 200
eta = 1e-4
ny = 101         # discretisation of y = x + sqrt(gamma)*noise. Adapt depending on gamma and tuning curve discretization.
AMP = 20         # max possible spike count

# setup cell data path
CELL_DATA_PATH = 'data/contrast_cells/carlo_data_cellno3.mat'       # cell responses and image principal components
METRICS_DATA_PATH = 'data/computed_contrast_cells/BDEvSSI_no3.npz'  # precomputed information metrics

# setup figure paraleters
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["axes.spines.top"] = False
plt.rcParams["xtick.major.width"] = 0.5 
plt.rcParams["xtick.minor.width"] = 0.5 
plt.rcParams["ytick.major.width"] = 0.5 
plt.rcParams["ytick.minor.width"] = 0.5
plt.rcParams["xtick.major.size"] = 3.5 * 1.1
plt.rcParams["xtick.minor.size"] = 2 * 1.1
plt.rcParams["ytick.major.size"] = 3.5 * 1.1
plt.rcParams["ytick.minor.size"] = 2 * 1.1

# setup paths
CELLS_PATH = os.path.join(proj_path, 'data/contrast_cells/') # path containing cells .mat files (e.g., carlo_data_cellno2.mat? (205 MB))
DATA_PATH = os.path.join(proj_path, 'data/computed_contrast_cells/') # path containing cells .mat files (e.g., carlo_data_cellno2.mat? (205 MB))



# utils ---------------------------------------------

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

    
# Neural network model of cell average response ---------------------------


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
    

def nnet_fit(pcs, fit, test_size=0.2, seed=42, n_epochs=200, linspace=np.linspace(-3,3,301)):
    """Fit tuning curves with a neural network

    Args:
        pcs (_type_): principal components of the cell's receptive field
        fit (_type_): tuning curve
        test_size (float, optional): _description_. Defaults to 0.2.
        seed (int, optional): _description_. Defaults to 42.
        n_epochs (int, optional): _description_. Defaults to 200.

    Returns:
        _type_: _description_
    """
    # get the predictors and predicted data
    X = np.copy(pcs[:2]).T          # x and y
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

    # Training loop
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
    grid_input = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    grid_input_scaled = scaler.transform(grid_input)

    # predict response
    with torch.no_grad():
        preds = model(torch.tensor(grid_input_scaled, dtype=torch.float32)).numpy()
        
    grad_X_tensor = torch.tensor(grid_input_scaled, dtype=torch.float32, requires_grad=True)
    output = model(grad_X_tensor)
    output.backward(torch.ones_like(output))
    grads = grad_X_tensor.grad.detach().numpy()
    return preds, model, grads, grid_x, grid_y, grid_input_scaled, grid_input


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


if __name__ == "__main__":
    """Entry point
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute psds")
    parser.add_argument("--recording-path", default= './dataset/00_raw/recording_dense_probe1', help="recording path.")
    
    args = parser.parse_args()
    
    # loop over the cells contained in the data path
    for i, input_filename in enumerate(os.listdir(CELLS_PATH)):

        # check that the file is a matlab .mat file
        if '.mat' not in input_filename:
            continue

        # print the name of the cell data file
        print(input_filename)                

        # ensure it contains the cell number "no" tag
        match = re.search(r"no\d+", input_filename)
        cell_no = match.group()

        # setup the precomputed data file to load 
        precomputed_filename = f'BDEvSSI_{cell_no}.npz'        
        print(precomputed_filename)
        print('\n')

        # load the data for that cell
        mat = sio.loadmat(os.path.join(CELLS_PATH, input_filename))
        np.random.seed(SEED)

        # get tuning curve data
        pcs = mat['X_lowd'] # principal components
        fit = mat['f']      # average firing rate (tuning)

        # prior --------------------------------------------------

        # - zero mean and data-driven covariance
        mu0 = pcs[0].mean()
        mu1 = pcs[1].mean()
        sigma = np.cov(pcs[0], pcs[1])

        # tuning curve model --------------------------------------------------

        baseline = 1e-2

        # get neuron's 2-D tuning curve
        # - train a neural net to predict RGCs responses in natural image 2-D latent space (space of (PC1,PC2) values)
        _, tuning_curve_model, _, grid_x, grid_y, latent_space_scaled, latent_space = nodes.nnet_fit(pcs[:2], fit, linspace=np.linspace(-10,10,100))

        # - predict neural responses in the image 2-D latent space (2-D tuning curve)
        rate_preds = nodes.tuning_curve_nnet(latent_space, tuning_curve_model)
        tuning_curve = rate_preds.reshape(grid_x.shape) + baseline

        # 2D latent space -------------------------------------------------------------

        # x points to plot, from -xmax to xmax
        nx = grid_x.shape[0]
        xmax = latent_space.max()              # 10 - OK! = 10 for grid linspace on -10 to 10 with 100 steps
        x = np.linspace(-xmax, xmax, nx)[None] # of shape (1,100)
        dx = x[0,1] - x[0,0]                   # note: dx is 10 X smaller than in replication - OK! = 0.202

        # create neuron 2D tuning curve f(PC1,PC2) 
        firingrate = rate_preds.reshape(nx,nx)

        # SSI   -------------------------------------------------------------

        # compute/load neural information metrics 
        # ssi, ssi_bound, mse, rmse = compute_ssi_and_bayes_mse_via_method1(pcs, grid_x, grid_y, tuning_c, sigma, latent_space, load=True, metrics_data_path=METRICS_DATA_PATH)
        ssi, Inf_ssi, logpx, logpr_x, logpr = nodes.compute_ssi(latent_space, firingrate, dx, amp=AMP, Sigmax=sigma)

        # iLocal   -------------------------------------------------------------
        
        Ilocal, Inf, gamma = nodes.compute_ilocal(latent_space, tuning_curve_model, nx, dx, ny, ngamma, gamma0, gamma_max, eta, AMP, sigma)

        # Plot -------------------------------------------------------------

        # setup parameters
        xylim = (-3,3)
        xyticks = (-3,0,3)

        # create mask to display data only within 3 standard deviations
        # beyond that we don't have enough data to estimate the tuning curve reliably
        mask_3std = nodes.create_gaussian_mask(grid_x, grid_y, np.array([mu0, mu1]), sigma, n_std=3)

        # apply mask to data
        preds_masked = np.where(mask_3std, rate_preds.reshape(grid_x.shape), np.nan)
        ssi_masked = np.where(mask_3std, ssi, np.nan)
        Ilocal_masked = np.where(mask_3std, Ilocal, np.nan)

        # setup plot
        fig = plt.figure(figsize=(1.7,8))

        # create main GridSpec: 1 col, 3 rows
        gs_main = gridspec.GridSpec(4,1, figure=fig, wspace=0, height_ratios=[1.5,1,1,1])

        # First subplot: Stimulus pcs & histogram --------------------------------------------------------------------------------
        
        gs_top = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                subplot_spec=gs_main[0],
                                                width_ratios=[4, 1],
                                                height_ratios=[1, 4],
                                                hspace=0.5, wspace=0.2)

        # create axes for the mosaic
        axs = {}
        axs['histx'] = fig.add_subplot(gs_top[0, 0])
        axs['scatter'] = fig.add_subplot(gs_top[1, 0])
        axs['histy'] = fig.add_subplot(gs_top[1, 1])

        # plot scatter points and histograms
        nodes.scatter_hist(pcs[0], pcs[1], axs['scatter'], axs['histx'], axs['histy'], c=(.7,0.7,0.7))
        axs['scatter'].set_aspect('equal')
        axs['scatter'].set_xlabel('Natural image PC1')
        axs['scatter'].set_ylabel('Natural image PC2')
        axs['scatter'].set_xlim(xylim)
        axs['scatter'].set_ylim(xylim)
        axs['scatter'].set_xticks(xyticks,xyticks)
        axs['scatter'].set_yticks(xyticks,xyticks)

        # plot prior (contours)
        for n_std in np.arange(0, 5, 1):
            nodes.plot_gaussian_ellipse(np.array([mu0, mu1]), sigma,
                                axs['scatter'], n_std=n_std,
                                edgecolor='red', facecolor='None')

        # aesthetics
        axs['scatter'].spines[['top','right']].set_visible(False)
        axs['histx'].spines[['top','right']].set_visible(False)
        axs['histx'].set_ylabel('Count')
        axs['histy'].spines[['top','right']].set_visible(False)
        axs['histy'].set_xlabel('Count')

        # Second subplot: Tuning curve and prior --------------------------------------------------------------------------------

        ax_bottom = fig.add_subplot(gs_main[1])

        # plot tuning curve as heatmap (MASKED - white outside 3 std)
        im = ax_bottom.contourf(grid_x, grid_y, preds_masked, levels=50, cmap='viridis', extend='neither')
        ax_bottom.set_facecolor('white')  # Set background to white for masked regions
        # colorbar
        divider = make_axes_locatable(ax_bottom)
        cax = divider.append_axes("right", size="10%", pad=0.3)
        cbar = plt.colorbar(im, cax=cax, label="mean spike count")

        # plot prior as contours
        for n_std in np.arange(0, 6, 1):
            nodes.plot_gaussian_ellipse(np.array([mu0, mu1]), sigma,
                                ax_bottom, n_std=n_std,
                                edgecolor='red', facecolor='None')

        # formatting
        ax_bottom.set_aspect('equal')
        ax_bottom.spines[['right']].set_visible(False)
        ax_bottom.set_xlabel("Natural image PC1")
        ax_bottom.set_ylabel("Natural image PC2")
        ax_bottom.set_xlim(xylim)
        ax_bottom.set_ylim(xylim)
        ax_bottom.set_xticks(xyticks,xyticks)
        ax_bottom.set_yticks(xyticks,xyticks)
        ax_bottom.spines[['top','right']].set_visible(False)

        # Third subplot: plot SSI (MASKED - white outside 3 std) ----------------------------------------------------
        ax_bottom3 = fig.add_subplot(gs_main[2])
        im = ax_bottom3.contourf(grid_x, grid_y, ssi_masked/log(2), levels=50, cmap='viridis', extend='neither')

        # formatting
        ax_bottom3.set_facecolor('white')  # Set background to white for masked regions
        ax_bottom3.set_xlim(xylim)
        ax_bottom3.set_ylim(xylim)
        ax_bottom3.set_xticks(xyticks,xyticks)
        ax_bottom3.set_yticks(xyticks,xyticks)
        # colorbar
        divider = make_axes_locatable(ax_bottom3)
        cax = divider.append_axes("right", size="10%", pad=0.3)
        cbar = plt.colorbar(im, cax=cax, label='SSI (bits)')

        # plot prior as contours
        for n_std in np.arange(0, 6, 1):
            nodes.plot_gaussian_ellipse(np.array([mu0, mu1]), sigma,
                                ax_bottom3, n_std=n_std,
                                edgecolor='red', facecolor='None')
        # formatting
        ax_bottom3.set_xlabel("Natural image PC1")
        ax_bottom3.set_ylabel("Natural image PC2")
        ax_bottom3.set_aspect('equal')
        ax_bottom3.spines[['top','right']].set_visible(False)



        # Fourth subplot: plot Ilocal (MASKED - white outside 3 std) ----------------------------------------------------
        ax_bottom3 = fig.add_subplot(gs_main[3])
        im = ax_bottom3.contourf(grid_x, grid_y, Ilocal_masked / log(2), levels=50, cmap='viridis', extend='neither')

        # formatting
        ax_bottom3.set_facecolor('white')  # Set background to white for masked regions
        ax_bottom3.set_xlim(xylim)
        ax_bottom3.set_ylim(xylim)
        ax_bottom3.set_xticks(xyticks,xyticks)
        ax_bottom3.set_yticks(xyticks,xyticks)
        # colorbar
        divider = make_axes_locatable(ax_bottom3)
        cax = divider.append_axes("right", size="10%", pad=0.3)
        cbar = plt.colorbar(im, cax=cax, label='$I_{local}$ (bits)')

        # plot prior as contours
        for n_std in np.arange(0, 6, 1):
            nodes.plot_gaussian_ellipse(np.array([mu0, mu1]), sigma,
                                ax_bottom3, n_std=n_std,
                                edgecolor='red', facecolor='None')
        # formatting
        ax_bottom3.set_xlabel("Natural image PC1")
        ax_bottom3.set_ylabel("Natural image PC2")
        ax_bottom3.set_aspect('equal')
        ax_bottom3.spines[['top','right']].set_visible(False)

        # format figure
        fig.subplots_adjust(wspace=0.9, hspace=0.5)

        # save figure
        plt.savefig(f'figures/all/fig3_{cell_no}.jpeg', bbox_inches = 'tight', transparent=True, dpi=400)    
    

