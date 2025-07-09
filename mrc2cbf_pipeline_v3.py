#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging
import warnings  
import argparse
import math
from functools import partial
from multiprocessing import Pool, cpu_count 
import multiprocessing as mp 
import sys 
from typing import Tuple, Optional, Dict, Any, List, Union

import numpy as np
import mrcfile
import fabio
from fabio.cbfimage import CbfImage 
from scipy.ndimage import gaussian_filter, shift as ndimage_shift
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from numpy.typing import NDArray

# --- Physical Constants ---
H_PLANCK = 6.62607015e-34  # J·s
M_ELECTRON = 9.1093837015e-31 # kg
E_CHARGE = 1.602176634e-19  # C
C_LIGHT = 2.99792458e8      # m/s

# --- Configuration ---
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING) 

# --- Logging Setup ---
def setup_logging(log_file_path: str, console_level=logging.INFO, file_level=logging.DEBUG):
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(file_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(console_level)
    logger = logging.getLogger()
    logger.setLevel(min(console_level, file_level)) 
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# --- MDoc Parsing ---
def parse_mdoc(mdoc_path: str) -> Dict[str, Any]:
    metadata = {}
    current_frameset_data = {}
    in_first_frameset = False 
    try:
        with open(mdoc_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith('[FrameSet'):
                    if line.strip() == '[FrameSet = 0]': 
                        in_first_frameset = True
                    else: 
                        in_first_frameset = False 
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip().lower(); value = value.strip()
                    target_dict = current_frameset_data if in_first_frameset else metadata
                    try:
                        if '.' in value or 'e' in value.lower(): 
                            target_dict[key] = float(value)
                        else: 
                            target_dict[key] = int(value)
                    except ValueError: 
                        target_dict[key] = value 
        combined_metadata = {k.lower(): v for k, v in metadata.items()}
        combined_metadata.update({k.lower(): v for k, v in current_frameset_data.items()})
        return combined_metadata
    except FileNotFoundError: 
        logging.error(f"Mdoc file not found: {mdoc_path}")
        return {}
    except Exception as e: 
        logging.error(f"Error parsing mdoc file {mdoc_path}: {e}", exc_info=True)
        return {}

# --- Wavelength Calculation ---
def calculate_wavelength_A(voltage_kv: float) -> float:
    if voltage_kv <= 0: raise ValueError("Voltage must be positive.")
    voltage_v = voltage_kv * 1000.0 
    term1 = 2 * M_ELECTRON * E_CHARGE * voltage_v
    term2 = 1 + (E_CHARGE * voltage_v) / (2 * M_ELECTRON * C_LIGHT**2)
    wavelength_m = H_PLANCK / math.sqrt(term1 * term2)
    wavelength_A = wavelength_m * 1e10 
    logging.debug(f"Calculated wavelength for {voltage_kv} kV: {wavelength_A:.6f} Å")
    return wavelength_A

# --- Gaussian Fitting ---
def gaussian_2d(coords: Tuple[NDArray[np.float64], NDArray[np.float64]],
                amplitude: float, center_x: float, center_y: float,
                sigma_x: float, sigma_y: float, offset: float) -> NDArray[np.float64]:
    x, y = coords; xo, yo = center_x, center_y
    exponent = -(((x - xo)**2) / (2 * sigma_x**2) + ((y - yo)**2) / (2 * sigma_y**2))
    return (offset + amplitude * np.exp(exponent)).ravel()

# --- Beam Center Finding ---
def find_beam_center_gaussian_fit(
    image: NDArray[np.float64], roi_size: int = 100, sigma_blur: float = 3.0,
    max_initial_deviation: float = 50.0, fit_bounds: bool = True
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    if image.ndim != 2: raise ValueError("Input image must be 2D.")
    if roi_size <= 0 or sigma_blur < 0: raise ValueError("roi_size must be positive, sigma_blur must be non-negative.")
    img_h, img_w = image.shape; center_y, center_x = img_h // 2, img_w // 2
    half_roi = roi_size // 2
    roi_y_start = max(0, center_y - half_roi); roi_y_end = min(img_h, center_y + half_roi + (roi_size % 2))
    roi_x_start = max(0, center_x - half_roi); roi_x_end = min(img_w, center_x + half_roi + (roi_size % 2))
    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]; roi_h, roi_w = roi.shape
    if roi_h == 0 or roi_w == 0:
        logging.warning(f"Beam centering: ROI size ({roi_size}) resulted in an empty array. Skipping fit.")
        return None, None, {'status': 'Failed (Empty ROI)'}
    roi_center_y, roi_center_x = roi_h / 2.0, roi_w / 2.0
    blurred_roi = gaussian_filter(roi, sigma=sigma_blur)
    try: 
        peak_idx_flat = np.argmax(blurred_roi)
        peak_y_roi, peak_x_roi = np.unravel_index(peak_idx_flat, blurred_roi.shape)
    except ValueError: 
        logging.warning("Beam centering: Could not find peak in blurred ROI. Skipping fit.")
        return None, None, {'status': 'Failed (argmax error)'}
    deviation = np.sqrt((peak_y_roi - roi_center_y)**2 + (peak_x_roi - roi_center_x)**2)
    if deviation > max_initial_deviation:
        logging.warning(f"Beam centering: Initial peak ({peak_x_roi:.1f}, {peak_y_roi:.1f}) in ROI is {deviation:.1f} px from ROI center. Skipping fit.")
        return None, None, {'status': 'Failed (Initial Peak Too Far)'}
    y_roi_coords, x_roi_coords = np.indices(roi.shape); roi_data_flat = roi.ravel()
    initial_amplitude = blurred_roi[peak_y_roi, peak_x_roi] - np.percentile(blurred_roi, 10)
    if initial_amplitude <= 0: initial_amplitude = 1.0
    initial_guess = (initial_amplitude, peak_x_roi, peak_y_roi, max(1.0, roi_w / 10.0), max(1.0, roi_h / 10.0), np.percentile(roi, 10))
    bounds = (-np.inf, np.inf)
    if fit_bounds: bounds = ([0, 0, 0, 0.5, 0.5, -np.inf], [np.inf, roi_w, roi_h, roi_w, roi_h, np.inf])
    fit_details = {'initial_guess': initial_guess, 'bounds': bounds if fit_bounds else 'None', 'roi_coords': (roi_x_start, roi_y_start),
                   'roi_shape': roi.shape, 'popt': None, 'pcov': None, 'status': '', 'method': ''}
    try:
        popt, pcov = curve_fit(gaussian_2d, (x_roi_coords, y_roi_coords), roi_data_flat, p0=initial_guess, bounds=bounds if fit_bounds else (-np.inf, np.inf), maxfev=5000)
        if fit_bounds:
             if not (bounds[0][1] < popt[1] < bounds[1][1] and bounds[0][2] < popt[2] < bounds[1][2]): 
                 logging.warning(f"Beam centering: Fit center ({popt[1]:.2f}, {popt[2]:.2f}) hit ROI boundary.")
             if popt[3] > roi_w or popt[4] > roi_h: 
                 logging.warning(f"Beam centering: Fit sigmas ({popt[3]:.2f}, {popt[4]:.2f}) seem large for ROI.")
        fit_center_x_roi, fit_center_y_roi = popt[1], popt[2]
        fit_details.update({'popt': popt, 'pcov': pcov, 'status': 'Fit Success', 'method': 'Gaussian Fit'})
    except RuntimeError as e: 
        logging.warning(f"Beam centering: Gaussian fit failed (RuntimeError: {e}). Falling back.")
        fit_center_x_roi, fit_center_y_roi = peak_x_roi, peak_y_roi
        fit_details.update({'status': 'Fit Failed (RuntimeError)', 'method': 'Blurred Peak Fallback'})
    except ValueError as e: 
        logging.warning(f"Beam centering: Gaussian fit failed (ValueError: {e}). Falling back.")
        fit_center_x_roi, fit_center_y_roi = peak_x_roi, peak_y_roi
        fit_details.update({'status': 'Fit Failed (Bounds Error)', 'method': 'Blurred Peak Fallback'})
    except Exception as e: 
        logging.error(f"Beam centering: Unexpected error: {e}", exc_info=True)
        fit_center_x_roi, fit_center_y_roi = peak_x_roi, peak_y_roi
        fit_details.update({'status': 'Fit Failed (Unexpected)', 'method': 'Blurred Peak Fallback'})
    beam_center_x = roi_x_start + fit_center_x_roi
    beam_center_y = roi_y_start + fit_center_y_roi
    if not (0 <= beam_center_x < img_w and 0 <= beam_center_y < img_h): 
        logging.error(f"Beam centering: Calculated center outside image bounds.")
        return None, None, {**fit_details, 'status': 'Failed (Outside Bounds)'}
    return beam_center_y, beam_center_x, fit_details

# --- Image Shifting ---
def shift_image(image: NDArray, current_center_yx: Tuple[float, float], target_center_yx: Tuple[float, float],
                order: int = 1, mode: str = 'constant', cval: float = 0.0) -> Tuple[NDArray, Tuple[float, float]]:
    current_y, current_x = current_center_yx
    target_y, target_x = target_center_yx
    shift_vector_yx = (target_y - current_y, target_x - current_x)
    shifted_image = ndimage_shift(image, shift=shift_vector_yx, order=order, mode=mode, cval=cval)
    return shifted_image, shift_vector_yx

# --- Array Binning ---
def bin_array(data: NDArray, bin_z: int, bin_y: int, bin_x: int) -> Tuple[NDArray, Tuple[int, int, int]]:
    if not all(isinstance(b, int) and b >= 1 for b in [bin_z, bin_y, bin_x]): raise ValueError("Bin factors must be positive integers.")
    if data.ndim != 3: raise ValueError(f"Input data must be 3D, got {data.ndim}D.")
    nz, ny, nx = data.shape
    nz_new, ny_new, nx_new = nz // bin_z, ny // bin_y, nx // bin_x
    if nz % bin_z != 0 or ny % bin_y != 0 or nx % bin_x != 0: 
        logging.warning(f"Binning: Data shape not perfectly divisible. Trailing slices discarded.")
    trimmed_data = data[:nz_new * bin_z, :ny_new * bin_y, :nx_new * bin_x]
    shape = (nz_new, bin_z, ny_new, bin_y, nx_new, bin_x)
    binned_data = trimmed_data.reshape(shape).sum(axis=(5, 3, 1))
    return binned_data, (nz_new, ny_new, nx_new)

# --- CBF Header Generation ---
def create_cbf_header(template_params: Dict[str, Any],
                       frame_specific_params: Dict[str, Any],
                       applied_pedestal: int = 0,
                       overload_value: int = 1000000) -> str: 
    header = f"""# Detector: {template_params.get('detector_model', 'PILATUS')}, S/N {template_params.get('serial_number', '00-0000')}
# Pixel_size {template_params['pixel_size_m']:.6e} m x {template_params['pixel_size_m']:.6e} m
# Detector_distance {template_params['detector_distance_m']:.5f} m
# Wavelength {template_params['wavelength_A']:.5f} A
# Beam_xy ({frame_specific_params['beam_center_x_px']:.2f}, {frame_specific_params['beam_center_y_px']:.2f}) pixels
# Start_angle {frame_specific_params['start_angle_deg']:.4f} deg.
# Angle_increment {template_params['angle_increment_deg']:.4f} deg.
# Applied_Pedestal {applied_pedestal} ! Added pedestal value
# Count_cutoff {overload_value} counts         ! Saturation value
# Detector_2theta 0.0000 deg."""
    return header

# --- Second Pass Beam Center Validation/Smoothing ---
def smooth_beam_centers(beam_centers_yx: List[Optional[Tuple[float, float]]],
                        max_jump_pixels: float = 2.0, window_length: int = 11,
                        polyorder: int = 2, fallback_strategy: str = 'previous'
                       ) -> List[Optional[Tuple[float, float]]]:
    num_frames = len(beam_centers_yx)
    if num_frames == 0: 
        return []
    valid_indices = [i for i, bc in enumerate(beam_centers_yx) if bc is not None]
    if len(valid_indices) < 2:
        logging.warning("Smoothing: Less than 2 valid centers, cannot perform jump detection/smoothing.")
        if fallback_strategy == 'global_median' and len(valid_indices) == 1:
             median_y, median_x = beam_centers_yx[valid_indices[0]]
             return [(median_y, median_x) if bc is None else bc for bc in beam_centers_yx]
        return beam_centers_yx

    outlier_indices = set()
    last_valid_idx = valid_indices[0]
    for i in range(1, len(valid_indices)):
        current_idx = valid_indices[i]
        if last_valid_idx < len(beam_centers_yx) and current_idx < len(beam_centers_yx):
            prev_bc, curr_bc = beam_centers_yx[last_valid_idx], beam_centers_yx[current_idx]
            if prev_bc is not None and curr_bc is not None:
                prev_y, prev_x = prev_bc
                curr_y, curr_x = curr_bc
                jump_dist = np.sqrt((curr_y - prev_y)**2 + (curr_x - prev_x)**2)
                if jump_dist > max_jump_pixels:
                    logging.warning(f"Smoothing: Frame {current_idx}: Large jump ({jump_dist:.2f} px) from frame {last_valid_idx}. Marking outlier.")
                    outlier_indices.add(current_idx)
                else:
                    last_valid_idx = current_idx
            else: 
                logging.warning(f"Smoothing: Unexpected None at index {current_idx} or {last_valid_idx}.")
        else: 
            logging.error(f"Smoothing: Invalid index: current={current_idx}, last_valid={last_valid_idx}, len={num_frames}")

    smoothed_centers_yx = list(beam_centers_yx)
    frames_to_fix = sorted(list(set(i for i, bc in enumerate(smoothed_centers_yx) if bc is None or i in outlier_indices)))

    if not frames_to_fix:
         logging.info("Smoothing: No None values or large jumps detected.")
    else:
        logging.info(f"Smoothing: Attempting to fix {len(frames_to_fix)} frames using strategy: {fallback_strategy}")

    non_outlier_indices = [i for i in valid_indices if i not in outlier_indices]
    global_median_y, global_median_x = None, None
    if non_outlier_indices:
        global_median_y = np.median([smoothed_centers_yx[i][0] for i in non_outlier_indices])
        global_median_x = np.median([smoothed_centers_yx[i][1] for i in non_outlier_indices])
    elif valid_indices:
        logging.warning("Smoothing: All valid points were outliers? Using median of all original valid points.")
        global_median_y = np.median([beam_centers_yx[i][0] for i in valid_indices])
        global_median_x = np.median([beam_centers_yx[i][1] for i in valid_indices])

    for i in frames_to_fix:
        fixed = False
        if fallback_strategy == 'previous':
            for j in range(i - 1, -1, -1):
                if 0 <= j < len(smoothed_centers_yx) and smoothed_centers_yx[j] is not None and j not in outlier_indices:
                    smoothed_centers_yx[i] = smoothed_centers_yx[j]
                    logging.debug(f"Smoothing Frame {i}: Used previous valid center from frame {j}.")
                    fixed = True; break
            if not fixed and global_median_y is not None:
                smoothed_centers_yx[i] = (global_median_y, global_median_x)
                logging.debug(f"Smoothing Frame {i}: No preceding valid, used global median.")
                fixed = True
        elif fallback_strategy == 'interpolate':
            prev_valid, next_valid = -1, -1
            for j in range(i - 1, -1, -1):
                 if 0 <= j < len(smoothed_centers_yx) and smoothed_centers_yx[j] is not None and j not in outlier_indices: 
                     prev_valid = j; break
            for j in range(i + 1, num_frames):
                 if 0 <= j < len(smoothed_centers_yx) and smoothed_centers_yx[j] is not None and j not in outlier_indices: 
                     next_valid = j; break
            if prev_valid != -1 and next_valid != -1:
                 y1_t, y2_t = smoothed_centers_yx[prev_valid], smoothed_centers_yx[next_valid]
                 if y1_t is not None and y2_t is not None:
                     y1, x1 = y1_t; y2, x2 = y2_t
                     fraction = (i - prev_valid) / (next_valid - prev_valid)
                     interp_y = y1 + fraction * (y2 - y1); interp_x = x1 + fraction * (x2 - x1)
                     smoothed_centers_yx[i] = (interp_y, interp_x)
                     logging.debug(f"Smoothing Frame {i}: Interpolated between {prev_valid} and {next_valid}.")
                     fixed = True
            if not fixed: 
                 if prev_valid != -1 and smoothed_centers_yx[prev_valid] is not None:
                     smoothed_centers_yx[i] = smoothed_centers_yx[prev_valid]; fixed = True
                     logging.debug(f"Smoothing Frame {i}: Used previous (no next valid).")
                 elif next_valid != -1 and smoothed_centers_yx[next_valid] is not None:
                     smoothed_centers_yx[i] = smoothed_centers_yx[next_valid]; fixed = True
                     logging.debug(f"Smoothing Frame {i}: Used next (no previous valid).")
                 elif global_median_y is not None:
                     smoothed_centers_yx[i] = (global_median_y, global_median_x); fixed = True
                     logging.debug(f"Smoothing Frame {i}: No neighbors, used global median.")
        elif fallback_strategy == 'global_median':
             if global_median_y is not None:
                 smoothed_centers_yx[i] = (global_median_y, global_median_x); fixed = True
                 logging.debug(f"Smoothing Frame {i}: Used global median.")

        if not fixed:
            smoothed_centers_yx[i] = None
            logging.warning(f"Smoothing Frame {i}: Beam center remains None after fallback '{fallback_strategy}'.")

    final_valid_indices = [i for i, bc in enumerate(smoothed_centers_yx) if bc is not None]
    if len(final_valid_indices) >= window_length and polyorder < window_length:
        try:
            if window_length % 2 == 0: 
                logging.warning(f"Smoothing: Sav-Gol window_length {window_length} is even, adjusting.")
                window_length += 1
            if window_length > len(final_valid_indices): 
                raise ValueError("Adjusted window too long.")

            final_valid_y = np.array([smoothed_centers_yx[i][0] for i in final_valid_indices])
            final_valid_x = np.array([smoothed_centers_yx[i][1] for i in final_valid_indices])
            smoothed_y = savgol_filter(final_valid_y, window_length, polyorder)
            smoothed_x = savgol_filter(final_valid_x, window_length, polyorder)

            for k, idx in enumerate(final_valid_indices):
                 original_y, original_x = smoothed_centers_yx[idx]
                 smooth_dist = np.sqrt((smoothed_y[k]-original_y)**2 + (smoothed_x[k]-original_x)**2)
                 smoothing_dev_thresh = max_jump_pixels * 1.5
                 if smooth_dist > smoothing_dev_thresh:
                      logging.warning(f"Smoothing Frame {idx}: Sav-Gol large deviation ({smooth_dist:.2f} px). Retaining pre-smoothed value.")
                 else: 
                      smoothed_centers_yx[idx] = (smoothed_y[k], smoothed_x[k])
            logging.info(f"Smoothing: Applied Savitzky-Golay (win={window_length}, order={polyorder}).")
        except ValueError as e:
             logging.warning(f"Smoothing: Skipping Sav-Gol: {e}")
        except Exception as e:
            logging.warning(f"Smoothing: Sav-Gol failed unexpectedly: {e}.")
    elif len(final_valid_indices) > 0: 
        logging.warning(f"Smoothing: Skipping Sav-Gol: Not enough valid points ({len(final_valid_indices)}) for window {window_length}.")
    else: 
        logging.warning("Smoothing: Skipping Sav-Gol: No valid points.")

    return smoothed_centers_yx

# --- Global variables for worker processes ---
g_worker_binned_data_beamfind = None 
g_worker_beam_find_func = None      

g_worker_binned_data_cbfwrite = None    
g_worker_common_params_cbfwrite = None  
g_worker_frame_params_list_cbfwrite = None
g_worker_final_beam_centers_cbfwrite = None
g_worker_output_dir_cbfwrite = None     


def init_worker_beam_find(bds_main, beam_find_func_main):
    """Initializer for beam finding worker processes."""
    global g_worker_binned_data_beamfind, g_worker_beam_find_func
    init_start_time = time.time()
    g_worker_binned_data_beamfind = bds_main
    g_worker_beam_find_func = beam_find_func_main
    logging.debug(f"BeamFind Worker {os.getpid()} initialized in {time.time() - init_start_time:.4f}s")

def find_beam_center_for_frame_mp(idx: int) -> Tuple[int, Optional[Tuple[float, float]], Optional[Dict]]:
    """Worker function to find beam center for a single frame using global data."""
    global g_worker_binned_data_beamfind, g_worker_beam_find_func
    try:
        if g_worker_binned_data_beamfind is None or g_worker_beam_find_func is None:
            logging.error(f"Frame {idx}: BeamFind worker not properly initialized!")
            return idx, None, {'status': 'Failed (Worker Not Initialized)'}

        frame_data_float = g_worker_binned_data_beamfind[idx].astype(np.float64)
        beam_y, beam_x, fit_details = g_worker_beam_find_func(frame_data_float)
        
        if beam_y is not None and beam_x is not None:
            return idx, (beam_y, beam_x), fit_details
        else:
            return idx, None, fit_details
    except Exception as e:
        logging.error(f"Frame {idx}: Error during parallel beam center finding: {e}", exc_info=True)
        return idx, None, {'status': f'Failed in MP (Exception: {type(e).__name__})'}

def init_worker_cbf_write(bds_main, cp_main, fpl_main, fbc_main, od_main):
    """Initializer for CBF writing worker processes."""
    global g_worker_binned_data_cbfwrite, g_worker_common_params_cbfwrite, g_worker_frame_params_list_cbfwrite, g_worker_final_beam_centers_cbfwrite, g_worker_output_dir_cbfwrite
    init_start_time = time.time()
    g_worker_binned_data_cbfwrite = bds_main
    g_worker_common_params_cbfwrite = cp_main
    g_worker_frame_params_list_cbfwrite = fpl_main
    g_worker_final_beam_centers_cbfwrite = fbc_main
    g_worker_output_dir_cbfwrite = od_main
    logging.debug(f"CBFWrite Worker {os.getpid()} initialized in {time.time() - init_start_time:.4f}s")


def process_frame_cbf_mp(task_args: Tuple[int, bool, bool] 
                        ) -> Tuple[int, Optional[Tuple[float, float]], Optional[Tuple[float, float]], str, Optional[str], Dict[str, float]]:
    global g_worker_binned_data_cbfwrite, g_worker_common_params_cbfwrite, g_worker_frame_params_list_cbfwrite, g_worker_final_beam_centers_cbfwrite, g_worker_output_dir_cbfwrite
    
    frame_idx, diagnostic_plot_frame, apply_shift = task_args

    timings = {
        'slicing_astype': 0.0, 'shift_image': 0.0, 'post_shift_proc': 0.0,
        'cbf_header_prep': 0.0, 'cbf_write': 0.0, 'plotting': 0.0, 'total_frame_proc': 0.0
    }
    overall_frame_start_time = time.time()

    try:
        binned_data_stack_local = g_worker_binned_data_cbfwrite
        common_params_local = g_worker_common_params_cbfwrite
        frame_params_list_local = g_worker_frame_params_list_cbfwrite
        final_beam_centers_local = g_worker_final_beam_centers_cbfwrite
        output_dir_local = g_worker_output_dir_cbfwrite

        if any(v is None for v in [binned_data_stack_local, common_params_local, frame_params_list_local, final_beam_centers_local, output_dir_local]):
            logging.error(f"Frame {frame_idx}: CBFWrite worker not properly initialized with all shared data.")
            timings['total_frame_proc'] = time.time() - overall_frame_start_time
            return frame_idx, None, None, 'Failed (Worker Init Error)', None, timings

        t_start_slicing = time.time()
        frame_data = binned_data_stack_local[frame_idx].astype(np.float32)
        timings['slicing_astype'] = time.time() - t_start_slicing
        
        frame_params = frame_params_list_local[frame_idx]
        num_rows, num_cols = frame_data.shape
        target_center_yx = ((num_rows - 1) / 2.0, (num_cols - 1) / 2.0)

        beam_center_final_local = final_beam_centers_local[frame_idx]

        if beam_center_final_local is None:
            logging.warning(f"Frame {frame_idx}: No valid beam center provided. Skipping CBF generation.")
            timings['total_frame_proc'] = time.time() - overall_frame_start_time
            return frame_idx, None, None, 'Skipped (No Beam Center)', None, timings

        current_center_yx = beam_center_final_local
        beam_y, beam_x = current_center_yx

        final_img_data: NDArray[np.int32]
        shift_vector_yx: Optional[Tuple[float, float]] = None
        should_shift_data = apply_shift
        
        if should_shift_data:
            pedestal_value = common_params_local.get('pedestal', 0.0)
            t_start_shift = time.time()
            if np.allclose(current_center_yx, target_center_yx, atol=1e-3):
                 shifted_img = frame_data 
                 shift_vector_yx = (0.0, 0.0)
            else:
                 shifted_img, shift_vector_yx = shift_image(
                     frame_data,
                     current_center_yx=current_center_yx,
                     target_center_yx=target_center_yx,
                     order=common_params_local.get('shift_interpolation_order', 1),
                     mode='constant',
                     cval=pedestal_value
                 )
            timings['shift_image'] = time.time() - t_start_shift
            
            t_start_post_shift = time.time()
            final_img_data = np.clip(np.round(shifted_img), 0, np.iinfo(np.int32).max).astype(np.int32)
            timings['post_shift_proc'] = time.time() - t_start_post_shift
        else:
            t_start_post_shift = time.time()
            final_img_data = np.clip(np.round(frame_data), 0, np.iinfo(np.int32).max).astype(np.int32)
            timings['post_shift_proc'] = time.time() - t_start_post_shift
            shift_vector_yx = (0.0, 0.0) 

        t_start_header_prep = time.time()
        header_beam_x = target_center_yx[1] if should_shift_data else beam_x
        header_beam_y = target_center_yx[0] if should_shift_data else beam_y
        header_beam_x_1based = header_beam_x + 1.0
        header_beam_y_1based = header_beam_y + 1.0
        actual_pedestal_value = int(common_params_local.get('pedestal', 0)) 
        overload_val = int(common_params_local.get('overload', 1000000)) 
        cbf_header_str = create_cbf_header(
            template_params=common_params_local,
            frame_specific_params={
                'beam_center_x_px': header_beam_x_1based,
                'beam_center_y_px': header_beam_y_1based,
                'start_angle_deg': frame_params['current_angle']
            },
            applied_pedestal=actual_pedestal_value, 
            overload_value=overload_val 
        )
        cbf_image = CbfImage(data=final_img_data)
        cbf_image.header.update({
            "_array_data.header_convention": "PILATUS_1.2",
            "_array_data.header_contents": cbf_header_str,
        })
        timings['cbf_header_prep'] = time.time() - t_start_header_prep

        cbf_file_path = os.path.join(output_dir_local, common_params_local['filename_template'].format(frame_idx + 1))
        status = 'Write Failed'
        try:
            t_start_write = time.time()
            cbf_image.write(cbf_file_path)
            timings['cbf_write'] = time.time() - t_start_write
            status = 'Processed'
        except Exception as write_e:
            logging.error(f"Frame {frame_idx}: Failed to write CBF file {cbf_file_path}: {write_e}", exc_info=True)
            cbf_file_path = None
        
        if diagnostic_plot_frame:
            t_start_plot = time.time()
            try: 
                plt.figure(figsize=(10, 8)); gs = plt.GridSpec(2, 2)
                ax1 = plt.subplot(gs[0, 0]); ax1.imshow(frame_data, cmap='inferno', origin='lower'); ax1.set_title("Original Binned")
                ax1.plot(beam_x, beam_y, 'bo', ms=5, mfc='none'); ax1.plot(target_center_yx[1], target_center_yx[0], 'rx', ms=5)
                ax2 = plt.subplot(gs[0, 1]); ax2.imshow(final_img_data, cmap='inferno', origin='lower'); ax2.set_title("Processed for CBF")
                ax2.plot(target_center_yx[1], target_center_yx[0], 'rx', ms=5)
                plt.suptitle(f'Diagnostic Plot - Frame {frame_idx}', fontsize=14); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_filename = os.path.join(output_dir_local, f"diagnostic_frame_{frame_idx+1:05d}.png")
                plt.savefig(plot_filename); plt.close()
            except Exception as plot_e: 
                logging.error(f"Frame {frame_idx}: Failed diagnostic plot: {plot_e}")
            timings['plotting'] = time.time() - t_start_plot
        
        timings['total_frame_proc'] = time.time() - overall_frame_start_time
        logging.debug(f"Frame {frame_idx}: Timings (s): Slice+astype: {timings['slicing_astype']:.4f}, Shift: {timings['shift_image']:.4f}, PostShift: {timings['post_shift_proc']:.4f}, CBFHeader: {timings['cbf_header_prep']:.4f}, CBFWrite: {timings['cbf_write']:.4f}, Plot: {timings['plotting']:.4f}, Total: {timings['total_frame_proc']:.4f}")
        
        return frame_idx, (beam_y, beam_x) if beam_center_final_local else None, shift_vector_yx, status, cbf_file_path, timings

    except Exception as e:
        timings['total_frame_proc'] = time.time() - overall_frame_start_time
        logging.error(f"Error processing frame {frame_idx} in worker: {e}", exc_info=True) 
        return frame_idx, None, None, f'Failed ({type(e).__name__})', None, timings


# --- Helper function for multiprocessing ---
def _worker_wrapper_beam_find(idx: int): 
    return find_beam_center_for_frame_mp(idx)

def _worker_wrapper_cbf_write(args_tuple: Tuple[int, bool, bool]): # CORRECTED
    """ Helper function to pass arguments to process_frame_cbf_mp. """
    return process_frame_cbf_mp(args_tuple) # Pass the tuple directly


# --- Main Pipeline Function ---
def mrc_to_cbf_pipeline_mp_init( 
    mrc_path: str, output_dir: str, pixel_size_mm: float, detector_distance_mm: float,
    wavelength_A: float, start_angle_deg: float, angle_increment_deg: float,
    overload_cli: int, 
    bin_x: int = 1, bin_y: int = 1, bin_z: int = 1, pedestal: Optional[int] = None,
    auto_pedestal: bool = True, skip_beam_centering: bool = False, 
    beam_center_roi_size: int = 100, beam_center_sigma_blur: float = 3.0, 
    beam_center_max_initial_deviation: float = 50.0, beam_center_fit_bounds: bool = True,
    perform_second_pass: bool = True, max_beam_jump_pixels: float = 2.0, 
    smoothing_window_length: int = 11, smoothing_polyorder: int = 2, 
    smoothing_fallback: str = 'previous', apply_image_shift: bool = True, 
    shift_interpolation_order: int = 1, num_workers: Optional[int] = None,
    filename_template: str = "image_{:05d}.cbf", 
    first_pass_diagnostic_plots: bool = False,
    final_diagnostic_plots: bool = True, diagnostic_plot_specific_frames: Optional[List[int]] = None,
    save_beam_centers_file: bool = True, limit_frames: Optional[int] = None
) -> None:
    start_time = time.time()
    logging.info(f"Pipeline started for: {mrc_path} -> CBF output (MP Initializer, Parallel Beamfind)")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Beam centering: {'SKIPPED (using geometric center)' if skip_beam_centering else 'ENABLED'}")
    if not skip_beam_centering: logging.info(f"Second pass smoothing/validation: {'ENABLED' if perform_second_pass else 'DISABLED'}")
    logging.info(f"Image shifting: {'ENABLED' if apply_image_shift else 'DISABLED'}")

    try:
        if not os.path.exists(mrc_path): raise FileNotFoundError(f"MRC file not found: {mrc_path}")
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            logging.info("Loading MRC data..."); data = mrc.data.copy()
            logging.info(f"MRC data loaded. Shape: {data.shape}, Type: {data.dtype}")
    except Exception as e: logging.critical(f"Failed to read MRC file: {e}", exc_info=True); return

    try:
        logging.info(f"Binning data (z={bin_z}, y={bin_y}, x={bin_x})..."); binned_data, new_shape = bin_array(data, bin_z, bin_y, bin_x)
        logging.info(f"Binned data shape: {new_shape}, Type: {binned_data.dtype}"); del data
    except Exception as e: logging.critical(f"Binning failed: {e}", exc_info=True); return

    if limit_frames is not None and limit_frames > 0:
        if limit_frames < binned_data.shape[0]: logging.warning(f"Limiting to first {limit_frames} frames."); binned_data = binned_data[:limit_frames]; new_shape = binned_data.shape
        else: logging.info(f"Frame limit ({limit_frames}) >= total frames. Processing all.")
    num_frames, nrows, ncols = new_shape
    if num_frames == 0: logging.critical("No frames remaining. Exiting."); return

    actual_pedestal = 0; common_params_for_cbf_workers_main = {} 
    try:
        if pedestal is not None: actual_pedestal = int(pedestal); logging.info(f"Using user pedestal: {actual_pedestal}")
        elif auto_pedestal:
            min_intensity = np.min(binned_data)
            if min_intensity < 0: actual_pedestal = abs(int(np.floor(min_intensity))); logging.info(f"Calculated pedestal: {actual_pedestal} (from min value {min_intensity})")
            else: logging.info("Auto-pedestal: No pedestal needed (min value >= 0).")
        else: logging.info("Pedestal: Disabled.")
        if actual_pedestal > 0:
            logging.info(f"Applying pedestal {actual_pedestal}...");
            if np.issubdtype(binned_data.dtype, np.integer):
                 if np.max(binned_data) > np.iinfo(np.int32).max - actual_pedestal: logging.warning(f"Pedestal+max_val may exceed int32 max.")
            binned_data = binned_data.astype(np.int64) + actual_pedestal
        common_params_for_cbf_workers_main['pedestal'] = float(actual_pedestal)
    except Exception as e: logging.critical(f"Error during pedestal handling: {e}", exc_info=True); return

    new_pixel_size_m = (pixel_size_mm / 1000.0) * bin_x
    new_angle_increment_deg_binned = angle_increment_deg * bin_z 
    detector_distance_m = detector_distance_mm / 1000.0
    
    if num_workers is None:
        try: workers = cpu_count(); logging.info(f"Using {workers} worker processes (cpu_count).")
        except NotImplementedError: workers = 1; logging.warning("cpu_count() not implemented; using 1 worker.")
    elif num_workers >= 1: workers = int(num_workers); logging.info(f"Using {workers} worker process(es).")
    else: workers = 1; logging.warning(f"Invalid num_workers ({num_workers}); using 1.")
    
    effective_workers = workers
    if num_frames > 0 and num_frames < workers: 
        effective_workers = num_frames
        logging.info(f"Adjusted effective workers to {effective_workers} (fewer frames than workers) for some operations.")

    final_beam_centers_yx: List[Optional[Tuple[float, float]]] = [None] * num_frames
    fit_details_list: List[Optional[Dict]] = [None] * num_frames 

    if skip_beam_centering:
        logging.info("Beam centering skipped by user request.")
        geometric_center_y = (nrows - 1) / 2.0; geometric_center_x = (ncols - 1) / 2.0
        logging.info(f"Using fixed geometric center: ({geometric_center_y:.2f}, {geometric_center_x:.2f}) pixels (0-based).")
        final_beam_centers_yx = [(geometric_center_y, geometric_center_x)] * num_frames
        for i in range(num_frames): fit_details_list[i] = {'status': 'Skipped (Geometric Center Used)'}
        perform_second_pass = False; first_pass_diagnostic_plots = False
    else:
        logging.info("Starting first pass beam center finding (parallel)...")
        first_pass_start_time = time.time()
        beam_find_partial_func = partial(find_beam_center_gaussian_fit,
                                   roi_size=beam_center_roi_size, sigma_blur=beam_center_sigma_blur,
                                   max_initial_deviation=beam_center_max_initial_deviation, fit_bounds=beam_center_fit_bounds)
        beam_find_tasks = list(range(num_frames))
        beam_find_workers = min(effective_workers, num_frames) if num_frames > 0 else 1

        if beam_find_workers > 1:
            with Pool(processes=beam_find_workers, 
                      initializer=init_worker_beam_find, 
                      initargs=(binned_data, beam_find_partial_func)) as pool:
                chunk_bf = max(1, num_frames // (beam_find_workers * 4)) if beam_find_workers > 0 else 1
                logging.info(f"BeamFind Pool: Using chunksize: {chunk_bf} for imap_unordered with {beam_find_workers} workers.")
                
                temp_results_beam_find = [None] * num_frames
                for i, result_bf in enumerate(pool.imap_unordered(_worker_wrapper_beam_find, beam_find_tasks, chunksize=chunk_bf)):
                    idx_res_bf, center_res_bf, details_res_bf = result_bf
                    temp_results_beam_find[idx_res_bf] = (center_res_bf, details_res_bf)
                    if (i + 1) % (max(1, num_frames // 10)) == 0 or (i + 1) == num_frames: 
                        logging.info(f"  Beam finding progress (parallel): {i+1}/{num_frames} tasks processed...")
                for idx_bf_final in range(num_frames):
                    if temp_results_beam_find[idx_bf_final] is not None:
                        center_val, details_val = temp_results_beam_find[idx_bf_final]
                        final_beam_centers_yx[idx_bf_final] = center_val
                        fit_details_list[idx_bf_final] = details_val
                        if center_val is None and details_val:
                             logging.warning(f"Frame {idx_bf_final}: Beam center finding failed in MP - {details_val.get('status', 'Unknown')}")
                    else: 
                        logging.error(f"Frame {idx_bf_final}: No result from beam finding worker.")
                        fit_details_list[idx_bf_final] = {'status': 'Failed (No result from worker)'}
        else: 
            logging.info("Running beam center finding sequentially...")
            init_worker_beam_find(binned_data, beam_find_partial_func) 
            for idx in range(num_frames):
                idx_res_bf, center_res_bf, details_res_bf = find_beam_center_for_frame_mp(idx)
                final_beam_centers_yx[idx_res_bf] = center_res_bf
                fit_details_list[idx_res_bf] = details_res_bf
                if center_res_bf is None and details_res_bf:
                     logging.warning(f"Frame {idx_res_bf}: Beam center finding failed - {details_res_bf.get('status', 'Unknown')}")
                if (idx + 1) % 50 == 0 or idx == num_frames - 1: 
                    logging.info(f"  Beam finding progress (sequential): {idx+1}/{num_frames}")

        first_pass_duration = time.time() - first_pass_start_time
        logging.info(f"First pass beam finding finished in {first_pass_duration:.2f}s.")
        valid_centers_count = sum(1 for bc in final_beam_centers_yx if bc is not None)
        logging.info(f"Found {valid_centers_count}/{num_frames} valid centers in first pass.")
        if valid_centers_count == 0: logging.critical("No valid centers found. Cannot proceed."); return
        
        first_pass_centers_for_plot = list(final_beam_centers_yx) 
        if first_pass_diagnostic_plots: 
             logging.info("Generating first pass diagnostic plot...") 
             try:
                 plt.figure(figsize=(10, 6)); valid_idx_plot = [i for i, bc in enumerate(first_pass_centers_for_plot) if bc is not None]
                 if valid_idx_plot:
                     plt.plot(valid_idx_plot, [first_pass_centers_for_plot[i][1] for i in valid_idx_plot], '.-', label='X')
                     plt.plot(valid_idx_plot, [first_pass_centers_for_plot[i][0] for i in valid_idx_plot], '.-', label='Y')
                 plt.xlabel('Frame Index (0-based)'); plt.ylabel('Center (pixels, 0-based)'); plt.title('Beam Centers - First Pass')
                 plt.legend(); plt.grid(True); plt.xlim(0, max(0, num_frames - 1))
                 plot_path = os.path.join(output_dir, "beam_centers_first_pass.png"); plt.savefig(plot_path)
                 logging.info(f"Saved first pass plot to {plot_path}"); plt.close()
             except Exception as e: logging.error(f"Failed generating first pass plot: {e}")

        if perform_second_pass: 
            logging.info("Starting second pass validation/smoothing...")
            try:
                if smoothing_window_length % 2 == 0: logging.warning(f"Adjusting smoothing window to {smoothing_window_length + 1}."); smoothing_window_length += 1
                final_beam_centers_yx = smooth_beam_centers(final_beam_centers_yx, max_jump_pixels=max_beam_jump_pixels, window_length=smoothing_window_length, polyorder=smoothing_polyorder, fallback_strategy=smoothing_fallback)
                logging.info("Second pass finished.")
            except Exception as e: logging.error(f"Error during second pass smoothing: {e}", exc_info=True); final_beam_centers_yx = first_pass_centers_for_plot
        else: logging.info("Skipping second pass validation/smoothing.")

    final_valid_count = sum(1 for bc in final_beam_centers_yx if bc is not None)
    logging.info(f"Final beam centers available for {final_valid_count}/{num_frames} frames.")
    if final_valid_count == 0 and not skip_beam_centering: logging.critical("No valid beam centers. Cannot proceed."); return
    elif final_valid_count < num_frames: logging.warning(f"{num_frames - final_valid_count} frames lack valid center and will be skipped.")

    if final_diagnostic_plots: 
        logging.info("Generating final beam center diagnostic plot...")
        try:
            plt.figure(figsize=(12, 7)); gs = plt.GridSpec(2, 1, height_ratios=[3, 1]); ax1 = plt.subplot(gs[0])
            valid_final_idx_plot = [i for i, bc in enumerate(final_beam_centers_yx) if bc is not None]
            if valid_final_idx_plot:
                label_suffix = "(Geometric)" if skip_beam_centering else f"(n={len(valid_final_idx_plot)})"
                ax1.plot(valid_final_idx_plot, [final_beam_centers_yx[i][1] for i in valid_final_idx_plot], '.-', label=f'Final Beam X {label_suffix}', ms=4, lw=1)
                ax1.plot(valid_final_idx_plot, [final_beam_centers_yx[i][0] for i in valid_final_idx_plot], '.-', label=f'Final Beam Y {label_suffix}', ms=4, lw=1)
            if not skip_beam_centering and (perform_second_pass or first_pass_diagnostic_plots): 
                 valid_first_idx_plot = [i for i, bc in enumerate(first_pass_centers_for_plot) if bc is not None]
                 if valid_first_idx_plot:
                      ax1.plot(valid_first_idx_plot, [first_pass_centers_for_plot[i][1] for i in valid_first_idx_plot], 'x', color='gray', alpha=0.5, label='First Pass X', ms=3)
                      ax1.plot(valid_first_idx_plot, [first_pass_centers_for_plot[i][0] for i in valid_first_idx_plot], '+', color='darkgray', alpha=0.5, label='First Pass Y', ms=3)
            ax1.set_ylabel('Center (pixels, 0-based)'); ax1.set_title('Beam Center: Geometric' if skip_beam_centering else 'Beam Centers: Final (Smoothed)'); ax1.legend(fontsize='small'); ax1.grid(True); plt.setp(ax1.get_xticklabels(), visible=False); ax1.set_xlim(0, max(0, num_frames - 1))
            ax2 = plt.subplot(gs[1], sharex=ax1)
            if not skip_beam_centering and perform_second_pass and len(valid_final_idx_plot) > 1:
                 jumps_dist = np.sqrt(np.diff([final_beam_centers_yx[i][1] for i in valid_final_idx_plot])**2 + np.diff([final_beam_centers_yx[i][0] for i in valid_final_idx_plot])**2)
                 ax2.plot(np.array(valid_final_idx_plot)[1:], jumps_dist, '.-', label='Jump Dist (px)', color='purple', ms=3, lw=0.8)
                 ax2.axhline(max_beam_jump_pixels, color='red', ls='--', lw=1, label=f'Outlier Thr ({max_beam_jump_pixels} px)'); ax2.set_ylabel('Frame-to-Frame\nJump (pixels)'); ax2.legend(fontsize='small')
            else: 
                 ax2.text(0.5, 0.5, 'Jump plot N/A', ha='center', va='center', transform=ax2.transAxes, color='gray'); ax2.set_yticks([])
            ax2.grid(True); plt.xlabel('Frame Index (0-based)'); plt.tight_layout(); 
            final_plot_path = os.path.join(output_dir, "beam_centers_final.png")
            plt.savefig(final_plot_path); plt.close()
            logging.info(f"Saved final beam center plot to {final_plot_path}")
        except Exception as e: logging.error(f"Failed generating final plot: {e}")

    if save_beam_centers_file: 
        bc_file_path = os.path.join(output_dir, "beam_centers_final.txt")
        try:
            with open(bc_file_path, 'w') as f:
                f.write(f"# Beam centers for {os.path.basename(mrc_path)}\n")
                f.write("# Columns: Frame_Index (0-based), Beam_Y (pixels, 0-based), Beam_X (pixels, 0-based), Status\n")
                for i, bc in enumerate(final_beam_centers_yx):
                    status_detail = fit_details_list[i].get('status', 'Unknown') if i < len(fit_details_list) and fit_details_list[i] else "Unknown"
                    if bc is not None: 
                        f.write(f"{i:<18d}  {bc[0]:<22.4f}  {bc[1]:<22.4f}  Valid ({status_detail})\n")
                    else: 
                         if not skip_beam_centering and perform_second_pass and i < len(first_pass_centers_for_plot) and first_pass_centers_for_plot[i] is not None: 
                             status_detail += " (Rejected Outlier)"
                         f.write(f"{i:<18d}  {'None':<22s}  {'None':<22s}  Failed ({status_detail})\n") 
            logging.info(f"Saved final beam center data to {bc_file_path}")
        except Exception as e: logging.error(f"Failed to save beam center file: {e}")

    common_params_for_cbf_workers_main.update({
        'pixel_size_m': new_pixel_size_m, 
        'detector_distance_m': detector_distance_m,
        'wavelength_A': wavelength_A, 
        'angle_increment_deg': new_angle_increment_deg_binned, 
        'filename_template': filename_template, 
        'shift_interpolation_order': shift_interpolation_order,
        'detector_model': 'PILATUS', 
        'serial_number': '00-0000',
        'overload': overload_cli, 
        'output_dir_mp': output_dir 
    })
    
    frame_params_list_main = [{'current_angle': start_angle_deg + idx * new_angle_increment_deg_binned} for idx in range(num_frames)]
    
    logging.info("Starting CBF writing (MP Initializer version)...")
    process_start_time = time.time()
    
    tasks_cbf_write = []
    for idx in range(num_frames):
        plot_this_frame = diagnostic_plot_specific_frames is not None and idx in diagnostic_plot_specific_frames
        tasks_cbf_write.append((idx, plot_this_frame, apply_image_shift)) 

    all_frame_timings = [] 
    processed_count = 0; skipped_count = 0; failed_count = 0

    cbf_write_workers = min(effective_workers, num_frames) if num_frames > 0 else 1

    if cbf_write_workers > 1 and num_frames > 0 :
        try:
            init_args_cbf_write = (binned_data, common_params_for_cbf_workers_main, frame_params_list_main, final_beam_centers_yx, output_dir)
            chunk_cbf = max(1, num_frames // (cbf_write_workers * 4)) if cbf_write_workers > 0 else 1
            logging.info(f"CBF Write Pool: Using chunksize: {chunk_cbf} for imap_unordered with {cbf_write_workers} workers.")

            with Pool(processes=cbf_write_workers, 
                      initializer=init_worker_cbf_write, 
                      initargs=init_args_cbf_write) as pool:
                result_iterator = pool.imap_unordered(_worker_wrapper_cbf_write, tasks_cbf_write, chunksize=chunk_cbf) # CORRECTED WRAPPER NAME
                logging.info(f"Submitted {len(tasks_cbf_write)} CBF tasks to pool ({cbf_write_workers} workers).")
                for i, result in enumerate(result_iterator):
                    if result is None: 
                        logging.error(f"CBF Task {i} (iterator) returned None")
                        failed_count +=1; continue
                    if len(result) < 6: # Check for the expected number of items including timings dict
                        logging.error(f"CBF Task {i} (iterator) returned malformed result: {result}")
                        failed_count +=1; continue
                        
                    frame_idx_res, _, _, status, _, frame_timings_dict = result 
                    
                    if frame_timings_dict: all_frame_timings.append(frame_timings_dict)
                    else: all_frame_timings.append({'total_frame_proc': -999}) # Should not happen if worker catches errors

                    if status == 'Processed': processed_count += 1
                    elif status.startswith('Skipped'): skipped_count += 1; logging.warning(f"Frame {frame_idx_res}: Skipped ({status})")
                    else: failed_count += 1; logging.error(f"Frame {frame_idx_res}: Failed ({status})")
                    if (i + 1) % (max(1, num_frames // 10)) == 0 or (i + 1) == num_frames: 
                        logging.info(f"  CBF writing progress: {i+1}/{num_frames}...")
        except Exception as e: 
            logging.critical(f"CBF writing multiprocessing pool error: {e}", exc_info=True)
            failed_count = num_frames - processed_count - skipped_count 
    elif num_frames > 0: 
        logging.info("Running CBF writing sequentially (simulating worker init)...")
        init_worker_cbf_write(binned_data, common_params_for_cbf_workers_main, frame_params_list_main, final_beam_centers_yx, output_dir)
        for i, task_args_tuple_seq in enumerate(tasks_cbf_write):
            result = _worker_wrapper_cbf_write(task_args_tuple_seq) # CORRECTED WRAPPER NAME
            if result is None: continue
            if len(result) < 6: continue
            frame_idx_res, _, _, status, _, frame_timings_dict = result
            if frame_timings_dict: all_frame_timings.append(frame_timings_dict)
            else: all_frame_timings.append({'total_frame_proc': -999})

            log_msg = f"Frame {frame_idx_res}: Status={status}"
            if status == 'Processed': 
                processed_count += 1
                if (i + 1) % (max(1, num_frames // 20)) == 0 or (i + 1) == num_frames: 
                    logging.info(f"  CBF sequential progress: {i+1}/{num_frames} processed.")
                else: 
                    logging.debug(log_msg) 
            elif status.startswith('Skipped'): 
                skipped_count += 1; logging.warning(log_msg)
            else: 
                failed_count += 1; logging.error(log_msg)

    logging.info(f"CBF generation finished in {time.time() - process_start_time:.2f} seconds.")
    
    if all_frame_timings:
        logging.info("--- Average Per-Frame CBF Processing Timings (seconds) ---")
        timing_keys_to_log = ['slicing_astype', 'shift_image', 'post_shift_proc', 
                              'cbf_header_prep', 'cbf_write', 'plotting', 'total_frame_proc']
        for key in timing_keys_to_log:
            valid_times_for_key = [t[key] for t in all_frame_timings if isinstance(t, dict) and key in t and t[key] != -999]
            if valid_times_for_key:
                avg_time = np.mean(valid_times_for_key)
                std_time = np.std(valid_times_for_key)
                logging.info(f"  Avg {key:<18}: {avg_time:.4f} +/- {std_time:.4f}")
            else:
                logging.info(f"  Avg {key:<18}: No data or all failed")
    
    logging.info(f"Summary: {processed_count} processed, {skipped_count} skipped, {failed_count} failed.")
    logging.info(f"Total pipeline execution time: {time.time() - start_time:.2f} seconds.")
    if failed_count > 0: logging.warning(f"{failed_count} frames failed.")
    if skipped_count > 0: logging.warning(f"{skipped_count} frames skipped.")
    if processed_count == 0 and num_frames > 0: logging.error("No frames were successfully processed.")
    elif processed_count > 0: logging.info("Pipeline completed.")

# ==============================================================================
# --- Command Line Argument Parsing and Main Execution ---
# ==============================================================================
def main():
    if sys.platform == "darwin":
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None or current_method != 'fork': 
            try:
                mp.set_start_method('fork', force=True)
            except RuntimeError as e:
                print(f"WARNING: Could not force 'fork' start method on macOS (current/default: {current_method}). Error: {e}. Using current/default.", file=sys.stderr)
            
    parser = argparse.ArgumentParser(
        description="Convert MRC image stacks to CBF format with optional beam centering, "
                    "smoothing, and metadata extraction from .mdoc files. Uses fabio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("mrc_path", help="Path to the input MRC file.")
    parser.add_argument("output_dir", help="Path to the directory for output CBF files and logs.")
    parser.add_argument("--mdoc", dest="mdoc_path", default=None, help="Optional .mdoc path.")
    parser.add_argument("--pixel-size-mm", type=float, default=None, help="Pixel size (mm, unbinned).")
    parser.add_argument("--detector-distance-mm", type=float, default=None, help="Detector distance (mm).")
    parser.add_argument("--wavelength-A", type=float, default=None, help="Wavelength (Å).")
    parser.add_argument("--start-angle-deg", type=float, default=None, help="Start tilt angle (deg).")
    parser.add_argument("--angle-increment-deg", type=float, default=None, help="Angle increment per original frame (deg).")
    parser.add_argument("--bin-x", type=int, default=None, help="Binning X.")
    parser.add_argument("--bin-y", type=int, default=None, help="Binning Y.")
    parser.add_argument("--bin-z", type=int, default=1, help="Binning Z (frames).")
    parser.add_argument("--pedestal", type=int, default=None, help="Fixed pedestal. Else auto.")
    parser.add_argument("--no-auto-pedestal", action="store_false", dest="auto_pedestal", help="Disable auto pedestal.")
    parser.add_argument("--skip-beam-centering", action="store_true", help="Skip beam centering, use geometric.")
    parser.add_argument("--no-image-shift", action="store_false", dest="apply_image_shift", help="Disable image shifting.")
    parser.add_argument("--shift-order", type=int, default=1, dest="shift_interpolation_order", choices=[0,1,2,3,4,5], help="Shift interpolation order.")
    parser.add_argument("--overload", type=int, default=1000000, dest="overload_value", 
                        help="Detector saturation value (Count_cutoff for CBF header).")
    bcenter_group = parser.add_argument_group('Beam Centering Tuning'); smooth_group = parser.add_argument_group('Beam Center Smoothing Tuning')
    bcenter_group.add_argument("--roi-size", type=int, default=100, dest="beam_center_roi_size")
    bcenter_group.add_argument("--blur-sigma", type=float, default=3.0, dest="beam_center_sigma_blur")
    bcenter_group.add_argument("--max-dev", type=float, default=50.0, dest="beam_center_max_initial_deviation")
    bcenter_group.add_argument("--no-fit-bounds", action="store_false", dest="beam_center_fit_bounds")
    smooth_group.add_argument("--no-smoothing", action="store_false", dest="perform_second_pass")
    smooth_group.add_argument("--max-jump", type=float, default=2.0, dest="max_beam_jump_pixels")
    smooth_group.add_argument("--smooth-window", type=int, default=11, dest="smoothing_window_length")
    smooth_group.add_argument("--smooth-order", type=int, default=2, dest="smoothing_polyorder")
    smooth_group.add_argument("--smooth-fallback", default='previous', choices=['previous', 'interpolate', 'global_median'], dest="smoothing_fallback")
    parser.add_argument("-n", "--num-workers", type=int, default=None, help="Number of parallel workers.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Process only first N frames.")
    parser.add_argument("--filename-template", default="image_{:05d}.cbf", help="Output CBF filename template.") 
    diag_group = parser.add_argument_group('Diagnostic Options')
    diag_group.add_argument("--plot-first-pass", action="store_true", dest="first_pass_diagnostic_plots")
    diag_group.add_argument("--no-final-plot", action="store_false", dest="final_diagnostic_plots")
    diag_group.add_argument("--plot-frames", type=str, default=None, dest="diagnostic_plot_specific_frames_str")
    diag_group.add_argument("--no-save-centers", action="store_false", dest="save_beam_centers_file")
    diag_group.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console log level.")
    diag_group.add_argument("--log-file-level", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="File log level.")
    args = parser.parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, f"mrc2cbf_mp_init_parBeam_{time.strftime('%Y%m%d_%H%M%S')}.log") 
        setup_logging(log_file, console_level=getattr(logging, args.log_level.upper()), file_level=getattr(logging, args.log_file_level.upper()))
    except Exception as e: 
        print(f"CRITICAL ERROR: Failed to set up logging: {e}", file=sys.stderr) 
        sys.exit(1) 

    logging.info(f"Using multiprocessing start method: {mp.get_start_method(allow_none=True)}")
    logging.info("Command line arguments parsed (CBF with MP Initializer & Parallel Beamfind).") 
    logging.debug(f"Arguments: {vars(args)}")

    mdoc_data = {}
    required_params = ['pixel_size_mm', 'detector_distance_mm', 'wavelength_A',
                       'start_angle_deg', 'angle_increment_deg']
    
    mdoc_derived_pixel_size = None; mdoc_derived_det_dist = None; mdoc_derived_wavelength = None
    mdoc_derived_start_angle = None; mdoc_derived_angle_increment = None

    if args.mdoc_path:
        logging.info(f"Parsing mdoc file: {args.mdoc_path}"); mdoc_data = parse_mdoc(args.mdoc_path)
        if not mdoc_data: logging.warning("Mdoc parsing failed or file was empty.")
        else:
            try:
                mdoc_binning = mdoc_data.get('binning', 1) 
                if args.bin_x is None: args.bin_x = mdoc_binning
                if args.bin_y is None: args.bin_y = mdoc_binning
                logging.info(f"Using binning factors: bin_x={args.bin_x}, bin_y={args.bin_y}")
                if 'camerapixelsize' in mdoc_data:
                     ps_um = mdoc_data['camerapixelsize']; mdoc_derived_pixel_size = ps_um / 1000.0 
                     logging.info(f"Mdoc CameraPixelSize {ps_um:.1f} um/pix -> UNBINNED pixel size {mdoc_derived_pixel_size:.6f} mm/pix.")
                elif 'pixelspacing' in mdoc_data: logging.warning(f"Found mdoc PixelSpacing = {mdoc_data['pixelspacing']}, but 'CameraPixelSize' was missing. Ignoring.")
                if 'voltage' in mdoc_data: mdoc_derived_wavelength = calculate_wavelength_A(mdoc_data['voltage']); logging.info(f"Mdoc Voltage {mdoc_data['voltage']} kV -> Wavelength {mdoc_derived_wavelength:.5f} Å.")
                if 'cameralength' in mdoc_data: mdoc_derived_det_dist = mdoc_data['cameralength']; logging.info(f"Mdoc CameraLength: {mdoc_derived_det_dist:.2f} mm.")
                if 'tiltangle' in mdoc_data: mdoc_derived_start_angle = mdoc_data['tiltangle']; logging.info(f"Mdoc TiltAngle (FrameSet 0): {mdoc_derived_start_angle:.4f} deg.")
                angle_incr_calculated = False 
                if 'degreespersecond' in mdoc_data and 'exposuretime' in mdoc_data and 'numsubframes' in mdoc_data and mdoc_data['numsubframes'] > 0:
                    deg_per_sec = mdoc_data['degreespersecond']; exp_time_sec = mdoc_data['exposuretime']; num_frames_orig = mdoc_data['numsubframes']
                    angle_incr = deg_per_sec * (exp_time_sec / num_frames_orig); mdoc_derived_angle_increment = angle_incr
                    logging.info(f"Mdoc DegreesPerSecond -> Angle Increment {angle_incr:.5f} deg/frame (orig)."); angle_incr_calculated = True
                elif not angle_incr_calculated and 'rotationrate' in mdoc_data and 'exposuretime' in mdoc_data and 'numsubframes' in mdoc_data and mdoc_data['numsubframes'] > 0:
                    rad_per_sec = mdoc_data['rotationrate']; exp_time_sec = mdoc_data['exposuretime']; num_frames_orig = mdoc_data['numsubframes']
                    angle_incr = math.degrees(rad_per_sec) * (exp_time_sec / num_frames_orig); mdoc_derived_angle_increment = angle_incr
                    logging.info(f"Mdoc RotationRate -> Angle Increment {angle_incr:.5f} deg/frame (orig).")
            except Exception as e: logging.error(f"Unexpected error processing mdoc data: {e}", exc_info=True)

    final_pipeline_args = {
        'mrc_path': args.mrc_path, 'output_dir': args.output_dir,
        'pixel_size_mm': args.pixel_size_mm if args.pixel_size_mm is not None else mdoc_derived_pixel_size,
        'detector_distance_mm': args.detector_distance_mm if args.detector_distance_mm is not None else mdoc_derived_det_dist,
        'wavelength_A': args.wavelength_A if args.wavelength_A is not None else mdoc_derived_wavelength,
        'start_angle_deg': args.start_angle_deg if args.start_angle_deg is not None else mdoc_derived_start_angle,
        'angle_increment_deg': args.angle_increment_deg if args.angle_increment_deg is not None else mdoc_derived_angle_increment,
        'overload_cli': args.overload_value, 
        'bin_x': args.bin_x, 'bin_y': args.bin_y, 'bin_z': args.bin_z,
        'pedestal': args.pedestal, 'auto_pedestal': args.auto_pedestal,
        'skip_beam_centering': args.skip_beam_centering,
        'beam_center_roi_size': args.beam_center_roi_size,
        'beam_center_sigma_blur': args.beam_center_sigma_blur,
        'beam_center_max_initial_deviation': args.beam_center_max_initial_deviation,
        'beam_center_fit_bounds': args.beam_center_fit_bounds,
        'perform_second_pass': args.perform_second_pass,
        'max_beam_jump_pixels': args.max_beam_jump_pixels,
        'smoothing_window_length': args.smoothing_window_length,
        'smoothing_polyorder': args.smoothing_polyorder,
        'smoothing_fallback': args.smoothing_fallback,
        'apply_image_shift': args.apply_image_shift,
        'shift_interpolation_order': args.shift_interpolation_order,
        'num_workers': args.num_workers,
        'filename_template': args.filename_template,
        'first_pass_diagnostic_plots': args.first_pass_diagnostic_plots,
        'final_diagnostic_plots': args.final_diagnostic_plots,
        'save_beam_centers_file': args.save_beam_centers_file,
        'limit_frames': args.limit_frames
    }
    
    if args.diagnostic_plot_specific_frames_str:
        try: 
            frame_indices = [int(f.strip()) for f in args.diagnostic_plot_specific_frames_str.split(',') if f.strip()]
            final_pipeline_args['diagnostic_plot_specific_frames'] = frame_indices
        except ValueError: 
            logging.warning("Could not parse --plot-frames. Ignoring.")
            final_pipeline_args['diagnostic_plot_specific_frames'] = None
    else: 
        final_pipeline_args['diagnostic_plot_specific_frames'] = None

    missing_final_params = [p for p in required_params if final_pipeline_args.get(p) is None]
    if missing_final_params:
        logging.critical(f"CRITICAL ERROR: Missing required parameters: {', '.join(missing_final_params)}")
        sys.exit(1)
    
    logging.info("Parameter preparation complete. Starting CBF pipeline with MP Initializer and Parallel Beamfind.")
    logging.debug(f"Final pipeline arguments: {final_pipeline_args}")
    try: 
        mrc_to_cbf_pipeline_mp_init(**final_pipeline_args) 
    except Exception as e: 
        logging.critical(f"Pipeline failed with an unhandled exception: {e}", exc_info=True)
        sys.exit(1) 

if __name__ == '__main__':
    if sys.platform == "darwin":
        current_method = mp.get_start_method(allow_none=True)
        # Only attempt to set if not already 'fork' or if not set (None implies it will use default 'spawn' for Pool)
        if current_method is None: 
            try:
                mp.set_start_method('fork')
            except RuntimeError: 
                try:
                    mp.set_start_method('fork', force=True)
                except RuntimeError as e_force:
                    print(f"WARNING: Could not force 'fork' start method on macOS (current/default: {current_method}). Error: {e_force}. Using current/default.", file=sys.stderr)
        elif current_method != 'fork':
             try:
                mp.set_start_method('fork', force=True)
             except RuntimeError as e_force:
                print(f"WARNING: Could not change start method to 'fork' on macOS (current: {current_method}). Error: {e_force}. Using current.", file=sys.stderr)
            
    main()  