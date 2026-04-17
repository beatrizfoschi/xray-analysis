import os
import concurrent.futures
import glob
import h5py

import numpy as np
import pandas as pd
import skimage as sk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


import scipy.ndimage as ndi
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist

from LaueTools.IOLaueTools import writefile_Peaklist


def load_images(dirpath: str) -> np.ndarray:
    """
    Load all .tif images in a directory into a 3D array: (Nimages, ny, nx).
    Uses ProcessPoolExecutor to parallelize skimage.io.imread.
    """
    im_files = glob.glob(f"{dirpath}/*.tif")
    im_files.sort()

    images = [None] * len(im_files)

    with concurrent.futures.ProcessPoolExecutor() as pool:
        for ii, image in enumerate(pool.map(sk.io.imread, im_files)):
            images[ii] = image

    return np.array(images)


def filter_and_rescale_images(
    im_array: np.ndarray,
    filter_type: str = "max",
    cutoff_freq:float = 0.005
):
    """
    Reduce a stack of images (N, ny, nx) to a 2D image by applying a reduction
    along axis 0 (min/max/mean/std/median), then apply a Butterworth filter and
    rescale to float32 in a [0,1]-like range (min-subtracted).
    """
    if im_array.ndim == 3:
        print(f"3D image will be reduced with the {filter_type} filter along axis 0.")

        if filter_type not in ("min", "max", "mean", "std", "median"):
            print("Filter not implemented. Quiting...")
            return 1

        if filter_type == "max":
            filt_im = im_array.max(axis=0)
        elif filter_type == "min":
            filt_im = im_array.min(axis=0)
        elif filter_type == "mean":
            filt_im = im_array.mean(axis=0)
        elif filter_type == "std":
            filt_im = im_array.std(axis=0)
        elif filter_type == "median":
            filt_im = np.median(im_array, axis=0)
    else:
        filt_im = im_array

    # sk.filters.butterworth exists in skimage >=0.19-ish; keep as-is
    filt_im = sk.filters.butterworth(filt_im, cutoff_frequency_ratio = cutoff_freq)

    # subtract minimum and cast to float32
    filt_im = sk.util.img_as_float32(filt_im - filt_im.min())
    return filt_im


def segment_image(
    im_array: np.ndarray,
    kernel_size: int = 3,
    sigma: float = 0.1,
    iterations: int = 2,
    threshold: float = 0,
):
    """
    Segment image into a boolean mask.

    If input is 3D, reduce by max(axis=0).
    If threshold == 0: use sk.filters.threshold_triangle.
    Then: mask = im >= threshold, binary opening, gaussian filter on mask.
    """
    if im_array.ndim != 2:
        im_array = im_array.max(axis=0)

    if threshold == 0:
        thrs = sk.filters.threshold_triangle(im_array)
    else:
        thrs = threshold

    mask = im_array >= thrs

    if iterations > 0:
        mask = ndi.binary_opening(
            mask,
            structure=np.ones((kernel_size, kernel_size)),
            iterations=iterations,
        )
    if sigma > 0:
        mask = ndi.gaussian_filter(mask, sigma=sigma)
    return mask


def label_segmented_image(im_array: np.ndarray, intensity_image=None):
    """
    Label segmented boolean image, clear borders, return:
    (label_image, n_labels, label_img_rgb)
    """
    if im_array.dtype != bool:
        print("I need a boolean array. Quiting...")
        return 1

    im_array = sk.segmentation.clear_border(im_array)

    label_image, n_labels = sk.measure.label(
        im_array, return_num=True, connectivity=2
    )

    label_img_rgb = sk.color.label2rgb(
        label_image, image=intensity_image, bg_label=0
    )

    return label_image, n_labels, label_img_rgb


def plot_labeled_image(label_img_rgb, regionprops, cmap = 'turbo'):
    """
    Show labeled image and plot bounding boxes of regionprops.
    """
    f, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(label_img_rgb, cmap = cmap)

    for region in regionprops:
        minr, minc, maxr, maxc = region.bbox

        rect = Rectangle(
            (minc-1, minr-1),
            (maxc - minc),
            (maxr - minr),
            fill=False,
            edgecolor="red",
            linewidth=0.5,
        )
        ax.add_patch(rect)

    f.tight_layout()
    return


def measure_peaks(labeled_image, intensity_image):
    """
    Wrapper for skimage.measure.regionprops.
    """
    return sk.measure.regionprops(labeled_image, intensity_image=intensity_image)


def gaussian_2d_rotated(coords, A, x0, y0, sigma_x, sigma_y, theta, C):
    """
    Rotated 2D Gaussian + offset, returns raveled model.
    coords = (x, y) as 2 arrays (meshgrid flattened later).
    """
    x, y = coords

    x0 = float(x0)
    y0 = float(y0)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    a = (cos_t**2) / (2 * sigma_x**2) + (sin_t**2) / (2 * sigma_y**2)
    b = (-np.sin(2 * theta)) / (4 * sigma_x**2) + (np.sin(2 * theta)) / (4 * sigma_y**2)
    c = (sin_t**2) / (2 * sigma_x**2) + (cos_t**2) / (2 * sigma_y**2)

    g = A * np.exp(
        -(
            a * (x - x0) ** 2
            + 2 * b * (x - x0) * (y - y0)
            + c * (y - y0) ** 2
        )
    ) + C

    return g.ravel()


def fit_gaussian_2d_rotated(image):
    """
    Fit a single rotated 2D gaussian to an image using curve_fit.
    Returns (popt, pcov, fit_img, x, y)
    """
    image = np.asarray(image)
    ny, nx = image.shape

    x = np.arange(nx)
    y = np.arange(ny)
    x, y = np.meshgrid(x, y)

    A0 = image.max() - np.abs(image.min())
    C0 = np.abs(image.min())

    total = image.sum()
    x0 = (x * image).sum() / total
    y0 = (y * image).sum() / total

    sigma_x0 = nx / 4
    sigma_y0 = ny / 4
    theta0 = 0.0

    p0 = (A0, x0, y0, sigma_x0, sigma_y0, theta0, C0)

    coords = np.vstack((x.ravel(), y.ravel()))
    data = image.ravel()

    lower = [0, 0, 0, 1e-6, 1e-6, -np.pi / 2, -np.inf]
    upper = [np.inf, nx, ny, nx, ny, np.pi / 2, np.inf]
    bounds = (lower, upper)

    popt, pcov = curve_fit(
        gaussian_2d_rotated, coords, data, p0=p0, bounds=bounds, maxfev=10000
    )

    fit_img = gaussian_2d_rotated(coords, *popt).reshape(ny, nx)

    return popt, pcov, fit_img, x, y


def gaussian_mixture_2d(coords, *params):
    """
    Sum of n rotated Gaussians + constant background C (last param).
    params layout:
      [A1, x01, y01, sx1, sy1, th1,  A2, x02, y02, sx2, sy2, th2, ..., C]
    """
    x, y = coords
    n = (len(params) - 1) // 6
    C = params[-1]

    model = np.zeros_like(x, dtype=float)

    for i in range(n):
        A, x0, y0, sx, sy, th = params[6 * i : 6 * i + 6]

        cos_t = np.cos(th)
        sin_t = np.sin(th)

        a = (cos_t**2) / (2 * sx**2) + (sin_t**2) / (2 * sy**2)
        b = (-np.sin(2 * th)) / (4 * sx**2) + (np.sin(2 * th)) / (4 * sy**2)
        c = (sin_t**2) / (2 * sx**2) + (cos_t**2) / (2 * sy**2)

        model += A * np.exp(
            -(
                a * (x - x0) ** 2
                + 2 * b * (x - x0) * (y - y0)
                + c * (y - y0) ** 2
            )
        )

    return (model + C).ravel()


def fit_gaussian_mixture_2d(image, n_components: int, init_params):
    """
    Fit mixture of n_components rotated Gaussians + constant background.
    init_params should be a list of length 6*n_components + 1 (including C).
    Returns (popt, pcov, fitted, X, Y)
    """
    image = np.asarray(image)
    ny, nx = image.shape

    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    coords = np.vstack((X.ravel(), Y.ravel()))
    data = image.ravel()

    lower = []
    upper = []
    for _ in range(n_components):
        lower += [0, 0, 0, 1e-6, 1e-6, -np.pi / 2]
        upper += [np.inf, nx, ny, nx, ny, np.pi / 2]

    lower += [-np.inf]
    upper += [np.inf]
    bounds = (lower, upper)

    popt, pcov = curve_fit(
        gaussian_mixture_2d, coords, data, p0=init_params, bounds=bounds, maxfev=20000
    )

    fitted = gaussian_mixture_2d(coords, *popt).reshape(ny, nx)
    return popt, pcov, fitted, X, Y


def auto_init_gaussian_mixture_global(
    image,
    n_components=None,
    smooth_sigma=2,
    threshold_rel=0.2,
    min_distance=6,
):
    """
    Heuristic init for gaussian mixture:
    - smooth image
    - find local maxima via maximum_filter
    - keep peaks above threshold_rel * max
    - compute centers (y,x) sorted by peak value, optionally keep top n_components
    - build weights based on distance to centers with Gaussian kernel (smooth_sigma)
    - for each component: weighted mean + covariance -> (A, xm, ym, sx, sy, theta)
    - background C0 = median(img)
    - clamp params within bounds
    """
    img = np.asarray(image, float)
    ny, nx = img.shape

    smoothed = ndi.gaussian_filter(img, smooth_sigma)
    neighborhood = ndi.maximum_filter(smoothed, size=min_distance)
    peaks = smoothed == neighborhood
    peaks &= smoothed > (threshold_rel * smoothed.max())

    labeled, num = ndi.label(peaks)
    slices = ndi.find_objects(labeled)

    peak_info = []
    for slc in slices:
        # argmax within slice then convert to global coords
        y, x = np.unravel_index(np.argmax(smoothed[slc]), smoothed[slc].shape)
        y += slc[0].start
        x += slc[1].start
        peak_info.append((smoothed[y, x], x, y))

    peak_info.sort(reverse=True)

    if n_components:
        peak_info = peak_info[:n_components]

    centers = np.array([(x, y) for _, x, y in peak_info])
    n = len(centers)

    Y, X = np.mgrid[0:ny, 0:nx]
    coords = np.column_stack([X.ravel(), Y.ravel()])
    pixels = img.ravel()

    dists = cdist(coords, centers)
    weights = np.exp(-(dists**2) / (2 * (smooth_sigma**2)))
    weights *= pixels[:, None]
    weights /= (weights.sum(axis=1, keepdims=True) + 1e-12)

    init_params = []

    for i in range(n):
        w = weights[:, i]
        total = w.sum() + 1e-12

        xm = (coords[:, 0] * w).sum() / total
        ym = (coords[:, 1] * w).sum() / total

        dx = coords[:, 0] - xm
        dy = coords[:, 1] - ym

        Ixx = (w * dx * dx).sum() / total
        Iyy = (w * dy * dy).sum() / total
        Ixy = (w * dx * dy).sum() / total

        cov = np.array([[Ixx, Ixy], [Ixy, Iyy]])
        eigvals, eigvecs = np.linalg.eigh(cov)

        sigma_x = np.sqrt(eigvals[1])
        sigma_y = np.sqrt(eigvals[0])

        theta = np.arctan2(eigvecs[1, 1], eigvecs[0, 1])

        # amplitude guess: pixel at center - median(img)
        A = img[int(centers[i, 1]), int(centers[i, 0])] - np.median(img)

        init_params += [A, xm, ym, sigma_x, sigma_y, theta]

    C0 = np.median(img)
    init_params.append(C0)

    # clamp to bounds used later
    lower = [0, 0, 0, 1e-6, 1e-6, -np.pi / 2] * n + [-np.inf]
    upper = [np.inf, nx, ny, nx, ny, np.pi / 2] * n + [np.inf]

    for ii in range(len(lower)):
        if init_params[ii] < lower[ii]:
            init_params[ii] = lower[ii]
        if init_params[ii] > upper[ii]:
            init_params[ii] = upper[ii]

    return init_params


def r_squared_image(data, model, mask=None):
    """
    R^2 = 1 - SS_res / SS_tot, computed on flattened arrays,
    optionally using boolean mask.
    """
    data = np.asarray(data, float)
    model = np.asarray(model, float)

    if mask is not None:
        data = data[mask]
        model = model[mask]
    else:
        data = data.ravel()
        model = model.ravel()

    ss_res = np.sum((data - model) ** 2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    return 1 - ss_res / ss_tot


def reduced_chi_squared(data, model, n_params: int = 2, noise_std=None, mask=None):
    """
    Reduced chi^2 with optional per-pixel noise_std.
    If noise_std is None, estimate using MAD of residuals.
    """
    data = np.asarray(data, float)
    model = np.asarray(model, float)

    if mask is not None:
        data = data[mask]
        model = model[mask]
        N = mask.sum()
    else:
        data = data.ravel()
        model = model.ravel()
        N = data.size

    residuals = data - model

    if noise_std is None:
        mad = np.median(np.abs(residuals - np.median(residuals)))
        noise_std = 1.4826 * mad + 1e-12

    # if noise_std is an array, flatten or mask it
    if np.ndim(noise_std) > 0:
        if mask is not None:
            noise_std = noise_std[mask]
        else:
            noise_std = np.ravel(noise_std)

    chi2 = np.sum((residuals / noise_std) ** 2)
    dof = max(N - n_params, 1)
    return chi2 / dof


def fwhm_from_sigma(s):
    return 2 * np.sqrt(2 * np.log(2)) * s


def reduced_chi_squared_poisson(
    data,
    model,
    n_params: int = 2,
    gain: float = 1.0,
    read_noise: float = 0.0,
    eps: float = 1e-12,
    mask=None,
):
    """
    Reduced chi^2 assuming Poisson + read noise.
    Converts ADU->e- using gain.
    variance = model_e + read_noise^2, clipped by eps.
    """
    data = np.asarray(data, float)
    model = np.asarray(model, float)

    if mask is not None:
        data = data[mask]
        model = model[mask]
        N = mask.sum()
    else:
        data = data.ravel()
        model = model.ravel()
        N = data.size

    model_e = model * gain
    data_e = data * gain

    variance = model_e + (read_noise**2)
    variance = np.maximum(variance, eps)

    chi2 = np.sum((data_e - model_e) ** 2 / variance)
    dof = max(N - n_params, 1)
    return chi2 / dof


def write_h5_spotsfile(
    image_array: np.ndarray,
    regionprops,
    outpath: str = "segmented_spotsfile.h5",
    d: int = 10,
    overwrite: bool = False,
    max_components: int = 2
):
    """
    Writes per-spot images and fitted parameters into an HDF5 file.

    It tries to fit a 1-component Gaussian mixture on a cropped ROI of size 2d around the
    region weighted centroid, and keeps results with r2 >= 0.9.
    """
    n_labels = len(regionprops)
    n_success = 0

    if overwrite and os.path.exists(outpath):
        os.remove(outpath)

    with h5py.File(outpath, "a") as hout:
        for ii, region in enumerate(regionprops):
            ycen, xcen = region.centroid_weighted

            ymin, ymax, xmin, xmax = get_spot_limits(image_array, ycen, xcen, d)

            image = image_array[
                ymin : ymax,
                xmin : xmax,
            ]

            try:
                for n_comp in range(1, max_components+1):
                    try:
                        init_params = auto_init_gaussian_mixture_global(image, n_components=n_comp, smooth_sigma=0.2)
                        popt, pcov, fitted, X, Y = fit_gaussian_mixture_2d(
                            image, n_components=n_comp, init_params=init_params
                        )
                        r2 = r_squared_image(image, fitted)
                    except ValueError:
                        n_comp -= 1
                        init_params = auto_init_gaussian_mixture_global(image, n_components=n_comp, smooth_sigma=0.2)
                        popt, pcov, fitted, X, Y = fit_gaussian_mixture_2d(
                            image, n_components=n_comp, init_params=init_params
                        )
                        r2 = r_squared_image(image, fitted)
                        break
                    # print(ii, n_comp, r2)
                    if r2 > 0.9:
                        break

                C = popt[-1]

                if r2 < 0.9:
                    hout[f"spot_{ii:04d}_0/r_squared"] = r2
                    hout[f"spot_{ii:04d}_0/image"] = image
                    hout[f"spot_{ii:04d}_0/yxcen"] = region.centroid_weighted
                    hout[f"spot_{ii:04d}_0/bbox"] = region.bbox
                    continue
                n_success += 1

                for jj in range(1, n_comp+1):
                    A, xm, ym, sigma_x, sigma_y, theta = popt[6 * (jj-1) : 6 * (jj-1) + 6]
                    hout[f"spot_{ii:04d}_{jj}/r_squared"] = r2
                    hout[f"spot_{ii:04d}_{jj}/image"] = image
                    hout[f"spot_{ii:04d}_{jj}/yxcen"] = region.centroid_weighted
                    hout[f"spot_{ii:04d}_{jj}/bbox"] = region.bbox

                    hout[f"spot_{ii:04d}_{jj}/peak_X"] = round(xm + (int(xcen) - d), 2)
                    hout[f"spot_{ii:04d}_{jj}/peak_Y"] = round(ym + (int(ycen) - d), 2)

                    hout[f"spot_{ii:04d}_{jj}/peak_Itot"] = round(A, 2)
                    hout[f"spot_{ii:04d}_{jj}/peak_Isub"] = round(A - C, 2)

                    hout[f"spot_{ii:04d}_{jj}/peak_fwaxmaj"] = round(fwhm_from_sigma(max(sigma_x, sigma_y)), 2)
                    hout[f"spot_{ii:04d}_{jj}/peak_fwaxmin"] = round(fwhm_from_sigma(min(sigma_x, sigma_y)), 2)

                    hout[f"spot_{ii:04d}_{jj}/peak_inclination"] = round(np.rad2deg(theta), 2)

                    hout[f"spot_{ii:04d}_{jj}/Xdev"] = round(((xcen - (xcen - d)) - xm), 2)
                    hout[f"spot_{ii:04d}_{jj}/Ydev"] = round(((ycen - (ycen - d)) - ym), 2)

                    hout[f"spot_{ii:04d}_{jj}/peak_bkg"] = round(C, 2)
                    hout[f"spot_{ii:04d}_{jj}/Ipixmax"] = int(region.image_intensity.max())

            except RuntimeError:
                r2 = 0
                hout[f"spot_{ii:04d}_0/r_squared"] = r2
                continue

    print(
        f"I successfully segmented {n_success} out of {n_labels} spots. "
        f"Rate of success: {n_success / n_labels:.3f}"
    )

def get_spot_limits(image_array: np.ndarray, ycen: float, xcen: float, d: int):
    """Return (ymin, ymax, xmin, xmax) for a d-pixel box around (xcen, ycen), clamped to image bounds."""
    ymin, ymax = int(ycen-d), int(ycen+d)
    xmin, xmax = int(xcen-d), int(xcen+d)

    if ymin < 0:
        ymin = 0

    if ymax > image_array.shape[0]:
        ymax = image_array.shape[0]
    if xmin < 0:
        xmin = 0

    if xmax > image_array.shape[0]:
        xmax = image_array.shape[0]

    return ymin, ymax, xmin, xmax


def convert_spotsfile2peaklist(h5path: str):
    """
    Read the spots HDF5 file and return lists of peak quantities
    for spots with r_squared >= 0.9.
    """
    peak_X = []
    peak_Y = []
    peak_I = []
    peak_fwaxmaj = []
    peak_fwaxmin = []
    peak_inclination = []
    Xdev = []
    Ydev = []
    peak_bkg = []
    Ipixmax = []

    with h5py.File(h5path, "r") as hin:
        for key in hin.keys():
            r2 = hin[f"{key}/r_squared"][()]

            if r2 < 0.9:
                continue

            peak_X.append(hin[f"{key}/peak_X"][()])
            peak_Y.append(hin[f"{key}/peak_Y"][()])
            peak_I.append(hin[f"{key}/peak_Isub"][()])
            peak_fwaxmaj.append(hin[f"{key}/peak_fwaxmaj"][()])
            peak_fwaxmin.append(hin[f"{key}/peak_fwaxmin"][()])
            peak_inclination.append(hin[f"{key}/peak_inclination"][()])
            Xdev.append(hin[f"{key}/Xdev"][()])
            Ydev.append(hin[f"{key}/Ydev"][()])
            peak_bkg.append(hin[f"{key}/peak_bkg"][()])
            Ipixmax.append(hin[f"{key}/Ipixmax"][()])

    peaklist = np.stack([
        peak_X,
        peak_Y,
        peak_I,
        peak_fwaxmaj,
        peak_fwaxmin,
        peak_inclination,
        Xdev,
        Ydev,
        peak_bkg,
    ], axis = 1)

    order = np.argsort(Ipixmax)

    peaklist = peaklist[order]

    return peaklist[::-1]


def write_peaklist_dat(peaklist: np.ndarray, outname: str) -> None:
    """Write a peaklist array to a space-delimited .dat file compatible with LaueTools."""
    if not outname.endswith('.dat'):
        outname = f'{outname}.dat'
    df = pd.DataFrame(data=peaklist, columns=['peak_X','peak_Y','peak_I','peak_fwaxmaj','peak_fwaxmin','peak_inclination','Xdev','Ydev','peak_bkg'])

    df.to_csv(outname, sep = ' ', index = False)

def LoG_segmentation(image: np.ndarray, sigma: float = 0.01, mult: float = 0.25) -> np.ndarray:
    """
    Segment diffraction spots using a Laplacian of Gaussian filter.

    Applies LoG (inverted sign) then thresholds at mult * triangle_threshold.
    Returns a boolean mask of segmented regions.
    """

    filt_im = ndi.gaussian_laplace(image, 0.01)*-1

    thrs = sk.filters.threshold_triangle(filt_im) * mult

    return filt_im >= thrs


def process_one_image(image_index, scan_h5_path, segment_folder, datfile_folder, corfile_folder, image_format = 'h5', entry = '1.1/measurement/eiger4m'):

    if image_format == 'h5':
        with h5py.File(scan_h5_path, 'r') as hin:
            frame = hin[entry][image_index].astype(np.float32)

    mask = LoG_segmentation(frame)
    label_img, n_labels, lab_rgb = label_segmented_image(mask, filt_im)
    regionprops = measure_peaks(label_img, filt_im)
    filt_im = filter_and_rescale_images(frame, cutoff_freq=0.001)

    h5segmentpath = os.path.join(segment_folder, f'frame_{image_index:05}.h5')

    write_h5_spotsfile(filt_im, regionprops, outpath = h5segmentpath)
    peaklist = convert_spotsfile2peaklist(h5segmentpath)

    datfilepath = os.path.join(datfile_folder, f'frame_{image_index:05}.dat')
    write_peaklist_dat(peaklist, datfilepath)

















































    
