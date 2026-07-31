import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cedalion import units
import xarray as xr
from scipy.stats import pearsonr
import pyvista as pv
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, to_hex
import os
import cedalion.vis.blocks as vbx


def zscore(x): return (x - np.mean(x)) / np.std(x)


def plot_block_average_mspoc(Sx_train, Sy_train,stim, mSPoC_X, mSPoC_Y, color_x="green", color_y="red"):
    Sy_train = Sy_train.assign_coords(samples=("time", np.arange(len(Sy_train.time))))
    # Assign the unit string to the time coordinate attributes
    Sy_train.time.attrs["units"] = "s" 
    Sy_train.attrs["units"] = "micromolar" 

    epochs = Sy_train.cd.to_epochs(
    stim,  # stimulus dataframe
        ['task'],  #
        before=1* units.s, 
        after=40* units.s, 
    )

    # calculate baseline
    baseline = epochs.sel(reltime=(epochs.reltime < 0)).mean("reltime")
    # subtract baseline
    epochs_blcorrected = epochs - baseline
    # group trials by trial_type. 
    blockaverage = epochs_blcorrected.groupby("trial_type").mean("epoch")
    hbo_to_plot = blockaverage.sel(trial_type='task', mSPoC_Y=mSPoC_Y).squeeze()

    Sx_train = Sx_train.assign_coords(samples=("time", np.arange(len(Sy_train.time))))
    Sx_train.time.attrs["units"] = "s" 

    # If you want to add it to the DataArray itself as well:
    Sx_train.attrs["units"] = "a.u" # or relevant fNIRS unit

    epochs = Sx_train.cd.to_epochs(
    stim,  # stimulus dataframe
        ['task'],  # I have only 1 types
        before=1* units.s,  # seconds before stimulus
        after=40* units.s,  # seconds after stimulus
    )

    # calculate baseline
    baseline = epochs.sel(reltime=(epochs.reltime < 0)).mean("reltime")
    # subtract baseline
    epochs_blcorrected = epochs - baseline
    # group trials by trial_type. For each group individually average the epoch dimension
    blockaverage = epochs_blcorrected.groupby("trial_type").mean("epoch")

    # Plot block averages.
    f, ax = plt.subplots(figsize=(8, 5))
    meg_to_plot = blockaverage.sel(trial_type='task', mSPoC_X=mSPoC_X).squeeze()

    ax.plot(blockaverage.reltime, meg_to_plot, color=color_x, lw=2, label='Task')
    ax.plot(blockaverage.reltime, hbo_to_plot, color=color_y, lw=2, label='Task')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axhline(0, color="black", linewidth=1) 
    ax.axvline(0, color="black", linewidth=1) 
    ax.set_xlabel("Relative Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Block Average: Sx (MEG) vs Sy (fNIRS HbO)")

    plt.show()



def plot_reconstructed_sources(Sx_hbo, Sy_hbo, Sx_hbr, Sy_hbr, component = 1):

    Sx_hbo = Sx_hbo.sel(mSPoC_X=str('Sx' + str(component))) # Select the component you want to plot
    Sy_hbo = Sy_hbo.sel(mSPoC_Y=str('Sy' + str(component)))
    Sx_hbr = Sx_hbr.sel(mSPoC_X=str('Sx' + str(component)))
    Sy_hbr = Sy_hbr.sel(mSPoC_Y=str('Sy' + str(component)))


    # Create a continuous time axis for the concatenated good trials
    # prevents gaps in the plot where trials were removed
    num_samples = len(Sy_hbo)
    continuous_time = np.arange(num_samples) * 0.5  # Assuming dt=0.5s between samples in the concatenated data

    # Normalize sources (z-score)
    sx_hbo_norm = zscore(Sx_hbo.values.flatten())
    sy_hbo_norm = zscore(Sy_hbo.values.flatten())
    
    sx_hbr_norm = zscore(Sx_hbr.values.flatten())
    sy_hbr_norm = zscore(Sy_hbr.values.flatten())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True)

    # Subplot 1: HbO (Oxy-hemoglobin)
    ax1.plot(continuous_time, sx_hbo_norm, color='green', linewidth=2, label=r"MEG Power ($\hat{S}_{x1}$)")
    ax1.plot(continuous_time, sy_hbo_norm, color='red', linewidth=2, alpha=0.7, label=r"HbO ($\hat{S}_{y1}$)")
    ax1.set_title("Optimized mSPoC Sources", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Normalized Units", fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_ylim(-3, 3) 
    ax1.grid(True, linestyle='--', alpha=0.4)

    num_samples_hbr = len(Sy_hbr)
    continuous_time_hbr = np.arange(num_samples_hbr) * 0.5  # Assuming dt=0.5s between samples in the concatenated data
    # Subplot 2: HbR (Deoxy-hemoglobin)
    ax2.plot(continuous_time_hbr, sx_hbr_norm, color='green', linewidth=2, label=r"MEG Power ($\hat{S}_{x1}$)")
    ax2.plot(continuous_time_hbr, sy_hbr_norm, color='blue', linewidth=2, alpha=0.7, label=r"HbR ($\hat{S}_{y1}$)")
    ax2.set_xlabel("Time (s)", fontsize=14)
    ax2.set_ylabel("Normalized Units", fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylim(-3, 3)
    ax2.grid(True, linestyle='--', alpha=0.4)

    max_time = 300
    ax2.set_xlim(0, max_time)

    # Add vertical lines at every 40s interval
    epoch_ticks = np.arange(0, max_time + 1, 40)

    v_lines = np.arange(40, max_time, 40)
    for ax in [ax1, ax2]:
        # Set the actual tick marks on the axis
        ax.set_xticks(epoch_ticks)
        for xc in v_lines:
            ax.axvline(x=xc, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)

    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=12, ncol=2, frameon=False)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=12, ncol=2, frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True, sharey=True)

    # --- Plotting HbO ---
    r_hbo, _ = pearsonr(sx_hbo_norm, sy_hbo_norm)
    ax1.scatter(sx_hbo_norm, sy_hbo_norm, s=4, color='gray', alpha=0.6) 

    # Calculate and plot the linear fit for HbO
    slope_hbo, intercept_hbo = np.polyfit(sx_hbo_norm, sy_hbo_norm, 1)
    x_vals = np.array([-3, 3])
    y_vals_hbo = slope_hbo * x_vals + intercept_hbo
    ax1.plot(x_vals, y_vals_hbo, color='crimson', linewidth=2, label='Linear Fit')

    ax1.text(0.05, 0.95, f'$r = {r_hbo:.2f}$', transform=ax1.transAxes, 
            fontsize=14, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    ax1.set_xlabel("MEG Power", fontsize=14)
    ax1.set_ylabel("HbO", fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # --- Plotting HbR ---
    r_hbr, _ = pearsonr(sx_hbr_norm, sy_hbr_norm)
    ax2.scatter(sx_hbr_norm, sy_hbr_norm, s=4, color='gray', alpha=0.6)

    # Calculate and plot the linear fit for HbR
    slope_hbr, intercept_hbr = np.polyfit(sx_hbr_norm, sy_hbr_norm, 1)
    y_vals_hbr = slope_hbr * x_vals + intercept_hbr
    ax2.plot(x_vals, y_vals_hbr, color='dodgerblue', linewidth=2, label='Linear Fit')

    ax2.text(0.05, 0.95, f'$r = {r_hbr:.2f}$', transform=ax2.transAxes, 
            fontsize=14, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    ax2.set_xlabel("MEG Power", fontsize=14)
    ax2.set_ylabel("HbR", fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    for ax in [ax1, ax2]:
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()




def plot_temporal_weight(wt_hbo, wt_hbr, time_shifts, optimal_lag_hbo, optimal_lag_hbr, meas_id):
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(7, 5))
    dt =0.5
    
    plt.plot(time_shifts, wt_hbo, marker='o', markersize=6, color='darkorange', linewidth=2.5, label='HbO')
    plt.plot(time_shifts, wt_hbr, marker='o', markersize=6, color='darkblue', linewidth=2.5, label='HbR')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3) # Zero line
    # Zero line for reference
    plt.axhline(0, color='black', linestyle='--', alpha=0.3) 
    plt.axvline(optimal_lag_hbo, color='orange', linestyle=':', linewidth=3, alpha=0.8)
    plt.axvline(optimal_lag_hbr, color='darkblue', linestyle=':', linewidth=3, alpha=0.8)
    plt.title((f"{meas_id}"), fontsize=22, fontweight='bold') 
    plt.xlabel("Time Shift (s)", fontsize=20)
    plt.ylabel("Weight (a.u.)", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.grid(True, alpha=0.3)


def compute_spatial_pattern(X_array, W_np):
    """
    X_array: The original data [time x parcel]
    W_np: The filter matrix [parcel x N_components]
    """
    X = X_array.values
    N = X.shape[0]  # Number of samples
    
    # Ensure W is a 2D matrix (parcel x components)
    if W_np.ndim == 1:
        W = W_np[:, np.newaxis]
    else:
        W = W_np    

    # Compute Covariance of the data (C)
    # The source defines this as (X.T @ X) / (N - 1) 
    C = (X.T @ X) / (N - 1)

    # Haufe Transformation for multiple components
    # Numerator: C @ W (shape: parcel x components)
    numerator = C @ W
    
    # Denominator: The variance of each extracted component
    # This is the diagonal of W.T @ C @ W 
    # use np.diag to get a vector of variances (one for each component)
    component_variances = np.diag(W.T @ C @ W)
    
    # Calculate A for all components
    A = numerator / component_variances

    # Create DataArray with 2 dimensions: parcel and mSPoC_Component
    # define labels "Sx1", "Sx2" 
    n_components = W.shape[1]
    comp_labels = [f"{i+1}" for i in range(n_components)]
    
    A_xr = xr.DataArray(
        A,
        coords={
            "parcel": X_array.parcel,
            "mSPoC_Component": comp_labels
        },
        dims=["parcel", "mSPoC_Component"],
        name="mSPoC_Spatial_Pattern"
    )
    # Normalization per component
    return A_xr / np.abs(A_xr).max(dim="parcel")


def plot_spatial_patterns(Ax_hbo,Ay_hbo, Ax_hbr, Ay_hbr, delay_hbo,delay_hbr,save_suffix, output_dir):
       parcel_list = Ax_hbo.parcel.values
   
       fig, ax = plt.subplots(figsize=(17, 9)) 

       x = np.arange(len(parcel_list))
       width = 0.18  
       gap = 0.1   

       # Group 1: MEG + HbO
       ax.bar(x - width - gap, Ax_hbo, width, 
              label='MEG (paired w/ HbO)', color='green', alpha=1)
       ax.bar(x - gap, Ay_hbo, width,
              label='DOT HbO', color='red', alpha=1)

       # Group 2: MEG + HbR
       ax.bar(x + gap, Ax_hbr, width, 
              label='MEG (paired w/ HbR)', color='lightgreen', alpha=1)
       ax.bar(x + width + gap, Ay_hbr, width, 
              label='DOT HbR', color='blue', alpha=1)

       ax.set_xticks(x)
       ax.set_xticklabels(parcel_list, rotation=90, fontsize=22) # Increase fontsize here
       ax.tick_params(axis='y', labelsize=22)
       ax.set_xticks(x)
       ax.set_xticklabels(parcel_list, rotation=90)
       ax.set_title(f"\n$\\tau_{{\mathrm{{HbO}}}}$ = {delay_hbo:.1f}, $\\tau_{{\mathrm{{HbR}}}}$ = {delay_hbr:.1f}", fontsize=24)

       # Legend with 2 columns to visually separate the pairs
       ax.legend(ncol=2, loc='lower right', frameon=True, fontsize=16)
       plt.tight_layout()


def plot_brain_map(A_1d, head_ras, parcel_list, title, percentile, cmap):
       
    # Clean and identify active vertices
    clean_mesh_parcels = head_ras.brain.vertices.parcel.str.replace("LHH", "LH")
    is_active_vertex = clean_mesh_parcels.isin(parcel_list)
    
    # Map the values from A_1d to the vertex array
    vertex_parcel_names = clean_mesh_parcels.values[is_active_vertex.values]
    vertex_values = np.full(len(clean_mesh_parcels), np.nan)
    vertex_values[is_active_vertex.values] = A_1d.sel(parcel=vertex_parcel_names).values

    # Define the Threshold (ex. Top 30% / 70th Percentile)
    active_mags = vertex_values[is_active_vertex.values]
    threshold_val = np.percentile(active_mags, percentile)
    
    # Setup Colormap and Normalization
    max_val = np.nanmax(np.abs(vertex_values))

    norm = colors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
    cmap = cm.get_cmap(cmap) # Red for HbO, Blue for HbR 

    # Create the RGB color array (N_vertices x 3)
    # Start with neutral light gray [0.9, 0.9, 0.9] for inactive/dropped areas 
    vertex_colors = np.full((len(vertex_values), 3), 0.9) 
    print(f"The {percentile}th percentile cutoff is {threshold_val:.3f} units")

    # Apply colors ONLY to vertices in the parcel_list AND above the threshold
    # This greys out the bottom 70% of magnitudes
    to_color_mask = is_active_vertex.values & (vertex_values >= threshold_val)
    
    if np.any(to_color_mask):
        vertex_colors[to_color_mask] = cmap(norm(vertex_values[to_color_mask]))[:, :3]

    plotter.background_color = "white"
    plotter.add_text(title, 
                     position=(0.3, 0.85), 
                     viewport=True,
                     font_size=30, 
                     color='black', 
                     shadow=False)
    # ----------------------
    # Plot the brain surface
    vbx.plot_surface(plotter, head_ras.brain, color=vertex_colors, silhouette=True)

    # We use a mesh for the colorbar, setting clim to the actual data max
    dummy_mesh = pv.Line((0,0,0), (0,0,0)) 
    dummy_mesh.cell_data["Spatial Pattern (a.u.)"] = 1

    sargs = dict(
        title_font_size=38,   # Size of "Spatial Pattern (a.u.)"
        label_font_size=34, 
        shadow=False, 
        n_labels=3,          
        italic=False,
        fmt="%.0f",           
        font_family="arial"   
    )

    plotter.add_mesh(
        dummy_mesh, 
        scalars="Spatial Pattern (a.u.)", 
        clim=[-1, 1], 
        cmap=cmap,
        smooth_shading=True,
        show_scalar_bar=True,
        scalar_bar_args=sargs  
    )
    return plotter


def plot_brain_maps_grid(Ax_hbo, Ay_hbo, Ax_hbr, Ay_hbr,head_ras, parcel_list ):

    # Extract the deep purple from the very end of PuOr
    puor_purple = to_hex(cm.PuOr_r(0)) 
    blue = to_hex(cm.bwr(0)) 

    #  Create the maps
    hbo_custom = LinearSegmentedColormap.from_list("HbO_Custom", [puor_purple, "#FFFFFF", "#FF0000"])
    hbr_custom = LinearSegmentedColormap.from_list("HbR_Custom", [puor_purple, "#FFFFFF", blue])

    # MEG-HbO Spatial Pattern
    pv.set_jupyter_backend('static')
    shape = (1, 1)
    plotter = pv.Plotter(shape=shape, window_size=[1000, 900],border=False)
    plotter.background_color = "white"

    plot_brain_map(Ax_hbo, head_ras, parcel_list, 
                                'MEG', percentile =0, cmap = 'PRGn', plotter=plotter)
    plotter.view_yz(negative=True)
    plotter.screenshot(output_dir/"spatial_patterns_comparison_meg-hbo.png")
    plotter.show()
    pv.set_jupyter_backend('static') 
    shape = (1, 1)
    plotter = pv.Plotter(shape=shape, window_size=[1000, 900],border=False)
    plotter.background_color = "white"

    plot_brain_map(Ay_hbo, head_ras, parcel_list, 
                                'DOT HbO', percentile=0,  cmap = hbo_custom, plotter=plotter)
    plotter.view_yz(negative=True)
    plotter.screenshot(output_dir/"spatial_patterns_comparison_hbo.png")
    plotter.show()

    #MEG-HbR Spatial Pattern
    pv.set_jupyter_backend('static') 
    shape = (1, 1)
    plotter = pv.Plotter(shape=shape, window_size=[1000, 900],border=False)
    plotter.background_color = "white"

    plot_brain_map(Ax_hbr, head_ras, parcel_list, 
                                'MEG', percentile =0, cmap = 'PRGn', plotter=plotter)
    plotter.view_yz(negative=True)
    plotter.screenshot(output_dir/"spatial_patterns_comparison_meg-hbr.png")
    plotter.show()
    pv.set_jupyter_backend('static') 
    shape = (1, 1)
    plotter = pv.Plotter(shape=shape, window_size=[1000, 900],border=False)
    plotter.background_color = "white"

    plot_brain_map(Ay_hbr, head_ras, parcel_list, 
                                'DOT HbR', percentile=0,  cmap = hbr_custom, plotter=plotter)
    plotter.view_yz(negative=True)
    plotter.show()



def prepare_training_data(Y, X, good_trials):
    # Include zscore here
    dt = 0.5
    target_fs = 1/dt

    # Samples in MEG for one fNIRS sample
    e_len = int(100 / target_fs) # e.g., 100 / 2 = 50 samples

    new_times = np.arange(Y.time[0], Y.time[-1], dt)
    Y = Y.interp(time=new_times, method="linear")
    new_parcel_names = [name.replace('_LHH', '_LH') for name in Y.parcel.values]

    Y_zscore = xr.DataArray(
        Y,
        coords={
            "time": Y.time.values,
            "parcel": new_parcel_names 
        },
        dims=("time", "parcel")
    )
    n_epochs = X.shape[0] // e_len 
    print(f"Number of epochs: {n_epochs}")

    # Synchronize: Crop Y from start, Crop X to match exact epoch count
    Y_sync = Y_zscore.isel(time=slice(0, n_epochs))
    X_final = X.isel(time=slice(0, n_epochs * e_len))

    # Define Trial Parameters ( 40s trials)
    trial_duration = 40  # seconds
    samples_per_trial_y = int(trial_duration / dt) 
    samples_per_trial_x = samples_per_trial_y * e_len # Samples in MEG for one fNIRS sample

    # Concatenate only the good trials
    X_good_list = []
    Y_good_list = []

    for t in good_trials:
        # Calculate indices for this specific 40s block
        y_start = t * samples_per_trial_y
        y_end = y_start + samples_per_trial_y
        x_start = t * samples_per_trial_x
        x_end = x_start + samples_per_trial_x
        # Extract slices
        Y_good_list.append(Y_sync.isel(time=slice(y_start, y_end))) # selects data based on its positional index (the "sample number")
        X_good_list.append(X_final.isel(time=slice(x_start, x_end)))

    # Combine all good blocks into the final training data
    X_concat = xr.concat(X_good_list, dim="time")
    Y_concat = xr.concat(Y_good_list, dim="time")
    # z-score the X_train and Y_train after concatenating 
    X_train = zscore(X_concat)
    Y_train = zscore(Y_concat)

    return X_train, Y_train


def plot_brain_maps_grid(Ax_hbo, Ay_hbo, Ax_hbr, Ay_hbr,head_ras, parcel_list ):

    puor_purple = to_hex(cm.PuOr_r(0)) 
    blue = to_hex(cm.bwr(0)) 

    # Create the maps
    hbo_custom = LinearSegmentedColormap.from_list("HbO_Custom", [puor_purple, "#FFFFFF", "#FF0000"])
    hbr_custom = LinearSegmentedColormap.from_list("HbR_Custom", [puor_purple, "#FFFFFF", blue])
    
    # MEG-HbO Spatial Pattern
    pv.set_jupyter_backend('static') # or 'server' for interactive
    shape = (1, 1)
    plotter = pv.Plotter(shape=shape, window_size=[1000, 900],border=False)
    plotter.background_color = "white"
    plot_brain_map(Ax_hbo, head_ras, parcel_list, 
                                'MEG', percentile =0, cmap = 'PRGn', plotter=plotter)
    plotter.view_yz(negative=True)
    plotter.screenshot(output_dir/"spatial_patterns_comparison_meg-hbo.png")
    plotter.show()

    pv.set_jupyter_backend('static') # or 'server' for interactive
    shape = (1, 1)
    plotter = pv.Plotter(shape=shape, window_size=[1000, 900],border=False)
    plotter.background_color = "white"
    plot_brain_map(Ay_hbo, head_ras, parcel_list, 
                                'DOT HbO', percentile=0,  cmap = hbo_custom, plotter=plotter)
    plotter.view_yz(negative=True)
    plotter.screenshot(output_dir/"spatial_patterns_comparison_hbo.png")
    plotter.show()

    #MEG-HbR Spatial Pattern
    pv.set_jupyter_backend('static') # or 'server' for interactive
    shape = (1, 1)
    plotter = pv.Plotter(shape=shape, window_size=[1000, 900],border=False)
    plotter.background_color = "white"
    plot_brain_map(Ax_hbr, head_ras, parcel_list, 
                                'MEG', percentile =0, cmap = 'PRGn', plotter=plotter)
    plotter.view_yz(negative=True)
    plotter.screenshot(output_dir/"spatial_patterns_comparison_meg-hbr.png")
    plotter.show()


    pv.set_jupyter_backend('static') # or 'server' for interactive
    shape = (1, 1)
    plotter = pv.Plotter(shape=shape, window_size=[1000, 900],border=False)
    plotter.background_color = "white"
    plot_brain_map(Ay_hbr, head_ras, parcel_list, 
                                'DOT HbR', percentile=0,  cmap = hbr_custom, plotter=plotter)
    plotter.view_yz(negative=True)
    plotter.screenshot(output_dir/"spatial_patterns_comparison_hbr.png")
    plotter.show()


def calculate_single_trial_snr(data, stim, bad_trial_indices=None,is_power=True, 
                               activation_start=3, activation_end=15, 
                               baseline_start=-1, baseline_end=0):
    """
    Calculates SNR using (un-averaged) trial data.
    is_power: Set to True if input is already power/variance (e.g., Sx)
              
    """
    # Assign the unit string to the time coordinate attributes
    data.time.attrs["units"] = "s" 

    # Segment data into epochs (1s before to 40s after stimulus)
    epochs = data.cd.to_epochs(
        stim,
        ['task'], 
        before=1 * units.s,
        after=40 * units.s,
    )
    epochs = epochs.rename({"reltime": "time"})

    # Get the actual number of epochs successfully created 
    actual_epoch_count = len(epochs.epoch) 
    
    current_good_trials = [t for t in range(actual_epoch_count) 
                           if t not in bad_trial_indices]
    epochs = epochs.isel(epoch=current_good_trials) 

    # Baseline correction (using pre-stimulus interval)
    trial_baseline_means  = epochs.sel(time=(epochs.time < 0)).mean("time")
    epochs_blc = epochs - trial_baseline_means
    
    # Numerator (Signal): Use the baseline-corrected task window
    act_win = epochs_blc.sel(time=slice(activation_start, activation_end))
    # Denominator (Noise Floor): Use the uncorrected epochs
    base_win = epochs_blc.sel(time=slice(baseline_start, baseline_end))

    # Drop empty or all-NaN epochs 
    act_win = act_win.dropna(dim="epoch", how="all")
    base_win  = base_win.dropna(dim="epoch", how="all")

    # Calculate Signal Power
    if is_power:
        # Numerator: Magnitude of the power change
        # Denominator: Absolute power level during rest
        activation_power = act_win.mean(dim=["time", "epoch"])
        baseline_power = base_win.mean(dim=["time", "epoch"])
    else:
        # Numerator: Variance of the task-locked response
        # Denominator: Variance of the noise floor
        activation_power = (act_win ** 2).mean(dim=["time", "epoch"])
        baseline_power = (base_win ** 2).mean(dim=["time", "epoch"])

    # Log Transform to dB
    snr = 10 * np.log10(np.abs(activation_power) / (np.abs(baseline_power) + 1e-18))

    return snr