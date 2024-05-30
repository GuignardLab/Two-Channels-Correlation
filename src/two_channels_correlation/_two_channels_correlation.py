"""
You should write your module/set of function/class(es) here
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from tifffile import imread, imwrite
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import pearsonr, spearmanr
import argparse


def intensity_normalisation(
    image: np.ndarray,
    percentile: float | int = 1,
    output_type: str | np.dtype = "float32",
) -> np.ndarray:
    """Normalise the image to 0-1 range using the given percentile.

    Args:
        image (np.ndarray): Image to be normalised.
        percentile (float | int, optional): Percentile to be used for normalisation.
                                            Defaults to 1.
        output_type (str | np.dtype, optional): Output data type. Defaults to "float32".
    Returns:
        np.ndarray: Normalised image.
    """
    if percentile < 0 or percentile > 100:
        raise ValueError("Percentile should be between 0 and 100.")
    min_val = np.percentile(image, percentile)
    max_val = np.percentile(image, 100 - percentile)
    out = np.where(image < min_val, min_val, image)
    out[out > max_val] = max_val
    return ((out - min_val) / (max_val - min_val)).astype(output_type)


def z_normalisation(image: np.ndarray, normalizer: np.ndarray, sigma=2) -> np.ndarray:
    """Normalise the image along the z-axis.

    Args:
        image (np.ndarray): Image to be normalised.
        normalizer (np.ndarray): Ubiquitous nuclei image to
                                compute the factor for each z-slice.
        sigma (int, optional): Sigma for the gaussian filter of z factor. Defaults to 2.

    Returns:
        np.ndarray: Normalised image.
    """
    embryo_th = np.array(list(map(threshold_otsu, normalizer)))
    embro_mask = (normalizer.T > embryo_th[None, :]).T
    em_intensity = np.where(embro_mask, normalizer, np.nan)
    z_median = gaussian_filter1d(np.nanmedian(em_intensity, axis=(1, 2)), sigma=2)
    z_median = np.min(z_median) / z_median

    return image * z_median[:, None, None]


def compute_correlation(
    C0: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    folder_path: Path,
    write_intermediate: bool = True,
    plot: bool = True,
    perc_int_norm: float = 0.5,
    sigma_z_smooth: int = 2,
    perc_membrane: int = 3,
    sigma_int_smooth: int = 1,
    output: Path = None,
):
    """Compute the correlation between two channels.

    Args:
        C0 (np.ndarray): Image of the ubiquitous nuclei channel.
        C1 (np.ndarray): Image of the first channel.
        C2  (np.ndarray): Image of the second channel.
        write_intermediate (bool, optional): Write intermediate images. Defaults to True.
        plot (bool, optional): Plot the scatter plot. Defaults to True.
        perc_int_norm (float, optional): Percentile for intensity normalisation. Defaults to 0.5.
        sigma_z_smooth (int, optional): Sigma for z normalisation. Defaults to 2.
        perc_membrane (int, optional): Percentile for membrane detection. Defaults to 3.
        sigma_int_smooth (int, optional): Sigma for intensity smoothing. Defaults to 1.
    """

    C1_z_norm = z_normalisation(C1, C0, sigma=sigma_z_smooth)
    C2_z_norm = z_normalisation(C2, C0, sigma=sigma_z_smooth)

    if write_intermediate:
        imwrite(folder_path / "C1_z_normed.tif", (C1_z_norm).astype("uint16"))
        imwrite(folder_path / "C2_z_normed.tif", (C2_z_norm).astype("uint16"))

    C1_norm = intensity_normalisation(C1_z_norm, percentile=perc_int_norm)
    C2_norm = intensity_normalisation(C2_z_norm, percentile=perc_int_norm)

    if write_intermediate:
        imwrite(folder_path / "C1_normed.tif", (C1_norm).astype("float32"))
        imwrite(folder_path / "C2_normed.tif", (C2_norm).astype("float32"))

    max_im = np.fmax(C1_norm, C2_norm)

    membranes = max_im >= np.percentile(max_im, 100 - perc_membrane)

    if write_intermediate:
        imwrite(folder_path / "membranes.tif", membranes.astype("uint8") * 255)

    if sigma_int_smooth > 0:
        C1_filtered = gaussian_filter(C1_z_norm, sigma=sigma_int_smooth)
        C2_filtered = gaussian_filter(C2_z_norm, sigma=sigma_int_smooth)
    else:
        C1_filtered = C1_z_norm
        C1_filtered = C2_z_norm

    S, e_s = spearmanr(C1_filtered[membranes], C2_filtered[membranes])
    R, e_r = pearsonr(C1_filtered[membranes], C2_filtered[membranes])
    print(f"Spearman correlation: {S:.2e}, p-val:{e_s:.2e}")
    print(f"Pearson correlation: {R:.2e}, p-val:{e_r:.2e}")

    fig, ax = plt.subplots(figsize=(8, 8))
    min_r = np.percentile(
        np.hstack([C1_filtered[membranes], C2_filtered[membranes]]), 0.05
    )
    max_r = np.percentile(
        np.hstack([C1_filtered[membranes], C2_filtered[membranes]]), 99.9
    )
    range_hist = (min_r, max_r)
    ax.hist2d(
        C1_filtered[membranes],
        C2_filtered[membranes],
        bins=(200, 200),
        cmap="magma",
        range=(range_hist, range_hist),
    )
    ax.set_xlabel("C2 intensity")
    ax.set_ylabel("C3 intensity")
    fig.tight_layout()
    ax.axis("equal")

    if output is not None:
        fig.savefig(output / "hist2d_plot.svg")
    if plot:
        plt.show()


def prep_data(
    folder_path: Path,
    *,
    p_im_c0: str = None,
    p_im_c1: str = None,
    p_im_c2: str = None,
    im_all: str = None,
    czyx: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the data for the pipeline.
    Either provide the individual images or the image with all channels and the channel order.

    Args:
        folder_path (Path): Path to the folder with images.
        p_im_c0 (str): Name of the ubiquitous nuclei image.
        p_im_c1 (str): Name of the first channel image.
        p_im_c2 (str): Name of the second channel image.
        im_all (str): Name of the image with all channels.
        czyx (tuple[int, int, int, int] | None, optional): Order of the axes. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Images of the ubiquitous nuclei,
            first channel, and second channel.
    """
    if czyx is None or im_all is None:
        C0_path = folder_path / p_im_c0
        C1_path = folder_path / p_im_c1
        C2_path = folder_path / p_im_c2

        C0 = imread(C0_path)
        C1 = imread(C1_path)
        C2 = imread(C2_path)

    else:
        im_all = imread(folder_path.expanduser() / im_all)
        reorder = im_all.transpose(czyx)
        C0 = reorder[0]
        C1 = reorder[1]
        C2 = reorder[2]

    return C0, C1, C2

def run_all():
    parser = argparse.ArgumentParser(
        description="A simple correlation checker for two channels in 3D images."
    )
    parser.add_argument(
        "-fp",
        "--folder_path",
        type=str,
        help="Path to the folder with images.",
        default=".",
    )
    parser.add_argument(
        "-p",
        "--p_im",
        type=str,
        help=("Name of the image with all channels."
              " If this or the czyx argument is None or not provided"
              " the script will look for the individual channels."),
        default=None,
    )
    parser.add_argument(
        "-czyx",
        type=int,
        nargs=4,
        help=("Axis order for the image (channel, z, y, x). Default is None."
              " If this or the p_im argument is None or not provided"
              " the script will look for the individual channels."),
        default=None,
    )
    parser.add_argument(
        "-p_c0",
        "--p_im_c0",
        type=str,
        help="Name of the ubiquitous nuclei image. (ignored if p_im and czyx are provided)",
        default="C1-4002.tif",
    )
    parser.add_argument(
        "-p_c1",
        "--p_im_c1",
        type=str,
        help="Name of the first channel image. (ignored if p_im and czyx are provided)",
        default="C2-4002.tif",
    )
    parser.add_argument(
        "-p_c2",
        "--p_im_c2",
        type=str,
        help="Name of the second channel image. (ignored if p_im and czyx are provided)",
        default="C3-4002.tif",
    )
    parser.add_argument(
        "-wi",
        "--write_intermediate",
        type=bool,
        help="Write intermediate images. Default is False.",
        default=False,
    )
    parser.add_argument(
        "-pl",
        "--plot",
        type=bool,
        help="Plot the scatter plot. Default is True.",
        default=True,
    )
    parser.add_argument(
        "-pin",
        "--perc_int_norm",
        type=float,
        help="Percentile for intensity normalisation. Default is 0.5.",
        default=0.5,
    )
    parser.add_argument(
        "-szs",
        "--sigma_z_smooth",
        type=int,
        help="Sigma for z normalisation. Default is 2.",
        default=2,
    )
    parser.add_argument(
        "-pm",
        "--perc_membrane",
        type=int,
        help="Percentile for membrane detection. Default is 3.",
        default=3,
    )
    parser.add_argument(
        "-sis",
        "--sigma_int_smooth",
        type=int,
        help="Sigma for intensity smoothing. Default is 1.",
        default=1,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to the output folder for the plot.",
        default=".",
    )

    args = parser.parse_args()

    folder_path = Path(args.folder_path)
    p_im_c0 = args.p_im_c0
    p_im_c1 = args.p_im_c1
    p_im_c2 = args.p_im_c2
    p_im = args.p_im
    czyx = tuple(args.czyx)
    print(czyx)
    write_intermediate = args.write_intermediate
    plot = args.plot
    perc_int_norm = args.perc_int_norm
    sigma_z_smooth = args.sigma_z_smooth
    perc_membrane = args.perc_membrane
    sigma_int_smooth = args.sigma_int_smooth
    output = Path(args.output)

    C0, C1, C2 = prep_data(folder_path, p_im_c0=p_im_c0, p_im_c1=p_im_c1, p_im_c2=p_im_c2, im_all=p_im, czyx=czyx)

    compute_correlation(
        C0,
        C1,
        C2,
        folder_path,
        write_intermediate=write_intermediate,
        plot=plot,
        perc_int_norm=perc_int_norm,
        sigma_z_smooth=sigma_z_smooth,
        perc_membrane=perc_membrane,
        sigma_int_smooth=sigma_int_smooth,
        output=output,
    )