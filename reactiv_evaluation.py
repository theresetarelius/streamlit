import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image


def evaluate_reactiv(reactiv_output: dict, gt_path: str, thresholds=None, save_path: str = None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 1.0, 0.1), 2)

    rgb     = reactiv_output["rgb"]
    extent  = reactiv_output["extent"]
    west, south, east, north = extent
    out_h, out_w = rgb.shape[:2]

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    saturation = np.where(cmax > 0, (cmax - cmin) / (cmax + 1e-8), 0.0)

    wgs84 = CRS.from_epsg(4326)

    with rasterio.open(gt_path) as gt_src:
        dst_transform = rasterio.transform.from_bounds(west, south, east, north, out_w, out_h)
        src_data = gt_src.read(1).astype(np.float32)
        gt_reprojected = np.zeros((out_h, out_w), dtype=np.float32)
        reproject(
            source=src_data,
            destination=gt_reprojected,
            src_transform=gt_src.transform,
            src_crs=gt_src.crs,
            dst_transform=dst_transform,
            dst_crs=wgs84,
            resampling=Resampling.nearest
        )

    gt_binary = (gt_reprojected > 0.5).astype(np.uint8)

    results = []
    for thr in thresholds:
        pred = (saturation >= thr).astype(np.uint8)
        TP = int(np.sum((pred == 1) & (gt_binary == 1)))
        FP = int(np.sum((pred == 1) & (gt_binary == 0)))
        TN = int(np.sum((pred == 0) & (gt_binary == 0)))
        FN = int(np.sum((pred == 0) & (gt_binary == 1)))
        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        results.append({"threshold": thr, "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                        "precision": precision, "recall": recall, "f1": f1})

    mid_thr  = thresholds[len(thresholds) // 2]
    pred_mid = (saturation >= mid_thr).astype(np.uint8)
    overlay  = rgb.copy()
    tp_mask  = (pred_mid == 1) & (gt_binary == 1)
    fp_mask  = (pred_mid == 1) & (gt_binary == 0)
    fn_mask  = (pred_mid == 0) & (gt_binary == 1)

    alpha = 0.5
    for c, val in zip(range(3), [0.0, 1.0, 0.0]):
        overlay[:, :, c] = np.where(tp_mask, alpha * val + (1 - alpha) * rgb[:, :, c], overlay[:, :, c])
    for c, val in zip(range(3), [1.0, 0.0, 0.0]):
        overlay[:, :, c] = np.where(fp_mask, alpha * val + (1 - alpha) * rgb[:, :, c], overlay[:, :, c])
    for c, val in zip(range(3), [0.0, 0.0, 1.0]):
        overlay[:, :, c] = np.where(fn_mask, alpha * val + (1 - alpha) * rgb[:, :, c], overlay[:, :, c])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(rgb);      axes[0].set_title("REACTIV RGB output"); axes[0].axis("off")
    axes[1].imshow(gt_binary, cmap="gray"); axes[1].set_title("Ground truth mask"); axes[1].axis("off")
    axes[2].imshow(overlay);  axes[2].set_title(f"Overlay (tröskel={mid_thr:.2f})"); axes[2].axis("off")
    axes[2].legend(handles=[
        mpatches.Patch(color="green", label="TP"),
        mpatches.Patch(color="red",   label="FP"),
        mpatches.Patch(color="blue",  label="FN"),
    ], loc="lower right", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return results
