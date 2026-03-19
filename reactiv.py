import os
import re
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from datetime import datetime
from PIL import Image
import rasterio.windows


def hsv_to_rgb(h, s, v):
    """
    Vectorized HSV -> RGB conversion, matching GEE's hsvToRgb().
    All inputs are numpy arrays in [0, 1].
    Returns rgb array of shape (3, H, W).
    """
    h6 = h * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    rgb = np.zeros((3, *h.shape), dtype=np.float32)

    for idx, (r, g, b) in enumerate([
        (v, t, p),   # i == 0
        (q, v, p),   # i == 1
        (p, v, t),   # i == 2
        (p, q, v),   # i == 3
        (t, p, v),   # i == 4
        (v, p, q),   # i == 5
    ]):
        mask = (i == idx)
        rgb[0][mask] = r[mask]
        rgb[1][mask] = g[mask]
        rgb[2][mask] = b[mask]

    return rgb


def run_reactiv(input_data: dict, data_folder: str = None) -> dict:
    """
    Run the REACTIV algorithm on Capella GEO GeoTIFF files.
    Matches GEE implementation by Elise Colin (ONERA).

    Args:
        input_data: dict with keys "startDate", "endDate", "bbox"
                    bbox = [west, south, east, north]
        data_folder: path to folder containing Capella subfolders.
                     Defaults to /opt/saab/mex/streamlit/data

    Returns:
        dict with keys "rgb" (numpy array H x W x 3, float32 [0,1])
        and "extent" [west, south, east, north],
        or dict with key "error" on failure.
    """

    if data_folder is None:
        data_folder = "/opt/saab/mex/streamlit/data"

    start_date = input_data["startDate"]
    end_date   = input_data["endDate"]
    bbox       = input_data["bbox"]

    west, south, east, north = bbox
    start = datetime.fromisoformat(str(start_date))
    end   = datetime.fromisoformat(str(end_date))

    print(f"Date range: {start} → {end}")
    print(f"BBox: {west:.3f},{south:.3f},{east:.3f},{north:.3f}")

    # -------------------------------------------------
    # FIND CAPELLA GEO FILES
    # Folder pattern: CAPELLA_C##_SP_GEO_HH_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS
    # -------------------------------------------------

    tif_files = []  # list of (path, date)

    for root_dir, dirs, files in os.walk(data_folder):
        folder_name = os.path.basename(root_dir)

        if not folder_name.startswith("CAPELLA_") or "_GEO_" not in folder_name:
            continue

        parts = folder_name.split("_")
        date_str = None
        for part in parts:
            if re.match(r"^\d{14}$", part):
                date_str = part[:8]
                break

        if not date_str:
            print(f"  Skipping (no date in folder): {folder_name}")
            continue

        date = datetime.strptime(date_str, "%Y%m%d")

        if not (start <= date <= end):
            continue

        for f in files:
            if f.endswith(".tif") and "preview" not in f.lower():
                tif_files.append((os.path.join(root_dir, f), date))

    print(f"Found {len(tif_files)} Capella .tif files in date range")

    if not tif_files:
        return {"error": "No Capella GEO files found in data folder for the given date range"}

    # -------------------------------------------------
    # LOAD IMAGES
    # -------------------------------------------------

    images  = []
    dates   = []
    extents = []
    wgs84   = CRS.from_epsg(4326)

    for path, date in sorted(tif_files, key=lambda x: x[1]):
        filename = os.path.basename(path)

        with rasterio.open(path) as src:
            src_crs = src.crs

            if src_crs and src_crs != wgs84:
                file_west, file_south, file_east, file_north = transform_bounds(
                    src_crs, wgs84, *src.bounds
                )
            else:
                file_west, file_south, file_east, file_north = src.bounds

            clip_west  = max(west,  file_west)
            clip_east  = min(east,  file_east)
            clip_south = max(south, file_south)
            clip_north = min(north, file_north)

            if clip_west >= clip_east or clip_south >= clip_north:
                print(f"  No overlap: {filename}")
                continue

            if src_crs and src_crs != wgs84:
                native_bounds = transform_bounds(
                    wgs84, src_crs, clip_west, clip_south, clip_east, clip_north
                )
            else:
                native_bounds = (clip_west, clip_south, clip_east, clip_north)

            window = from_bounds(*native_bounds, transform=src.transform)

            # Klipp window till filens faktiska gränser
            file_window = rasterio.windows.Window(0, 0, src.width, src.height)
            window = window.intersection(file_window)

            if window.width <= 0 or window.height <= 0:
                print(f"  Window outside file bounds: {filename}")
                continue

            win_h = max(1, int(round(window.height)))
            win_w = max(1, int(round(window.width)))

            GRID_SIZE = 512
            scale = min(GRID_SIZE / win_h, GRID_SIZE / win_w, 1.0)
            out_h = max(1, int(win_h * scale))
            out_w = max(1, int(win_w * scale))

            try:
                arr = src.read(
                    1,
                    window=window,
                    out_shape=(out_h, out_w),
                    resampling=rasterio.enums.Resampling.average
                ).astype(np.float32)
            except Exception as e:
                print(f" Skipping corrupt file: {filename} ({e})")
                continue

            # Capella uint16 DN -> linjär intensitet
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            arr[arr < -1e6] = np.nan
            arr[arr == 0] = np.nan  # Capella använder 0 som nodata

            valid = arr[~np.isnan(arr)]
            if valid.size > 0:
                p99_dn = np.percentile(valid, 99)
                if p99_dn > 0:
                    arr = (arr / p99_dn) ** 2  # DN amplitud -> normaliserad linjär intensitet

            if arr.size == 0 or np.all(np.isnan(arr)):
                print(f"  Empty after clip: {filename}")
                continue

            print(f"  Loaded {filename}: shape={arr.shape}, range=[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")

            images.append(arr)
            dates.append(date)
            extents.append((clip_west, clip_south, clip_east, clip_north))

    if len(images) == 0:
        return {"error": "No overlapping Capella images found in date range"}

    print(f"Using {len(images)} images for REACTIV")

    # -------------------------------------------------
    # RESAMPLE TO COMMON GRID
    # -------------------------------------------------

    max_h = max(i.shape[0] for i in images)
    max_w = max(i.shape[1] for i in images)

    resampled = []
    for im in images:
        if im.shape == (max_h, max_w):
            resampled.append(im)
            continue
        nan_mask = np.isnan(im)
        im_fill  = np.where(nan_mask, 0.0, im)
        pil      = Image.fromarray(im_fill)
        pil      = pil.resize((max_w, max_h), Image.BILINEAR)
        result   = np.array(pil, dtype=np.float32)
        mask_pil = Image.fromarray(nan_mask.astype(np.uint8) * 255)
        mask_pil = mask_pil.resize((max_w, max_h), Image.NEAREST)
        mask_arr = np.array(mask_pil) > 127
        result[mask_arr] = np.nan
        resampled.append(result)

    stack = np.stack(resampled)  # (N, H, W)

    # -------------------------------------------------
    # REACTIV ALGORITHM — matchar GEE exakt
    # -------------------------------------------------

    # Step 1: amplitude = sqrt(linear power), som GEE's amplitude()
    amplitude = np.where(
        np.isnan(stack) | (stack <= 0),
        np.nan,
        np.sqrt(stack)
    )

    no_data_mask = np.all(np.isnan(amplitude), axis=0)

    # Step 2: magic = std / mean  (CV)
    mean_amp  = np.nanmean(amplitude, axis=0)
    std_amp   = np.nanstd(amplitude,  axis=0)
    mean_safe = np.where(mean_amp == 0, 1e-10, mean_amp)
    magic     = std_amp / mean_safe

    # Step 3: imax
    imax = np.nanmax(amplitude, axis=0)

    # Step 4: days — tid för maximum, t ∈ [0, 1]
    ds = max((end - start).days, 1)
    days_stack = []
    for img, date in zip(amplitude, dates):
        t        = (date - start).days / ds
        days_img = np.where((~np.isnan(img)) & (img >= imax), t, 0.0)
        days_stack.append(days_img)
    days = np.sum(days_stack, axis=0)

    # Step 5: magicnorm — EXAKT som GEE:
    # magicnorm = magic.subtract(mu).divide(stdmu.multiply(10)).clamp(0,1)
    sizepile = np.sum(~np.isnan(amplitude), axis=0).astype(np.float32)
    sizepile = np.where(sizepile < 1, 1.0, sizepile)

    mu     = 0.2286
    stdmu  = 0.1616 / np.sqrt(sizepile)

    magicnorm = (magic - mu) / (stdmu * 10.0)
    magicnorm = np.clip(magicnorm, 0.0, 1.0)

    # Step 6: Value channel — imax clamp(0,1) som GEE
    # Capella är i linjär amplitud efter sqrt, kan överstiga 1 för urban
    # Normalisera med p99 för att matcha GEE's GRD_FLOAT dynamik
    # Byt ut Step 6 mot detta:
    valid_imax = imax[~no_data_mask]
    if valid_imax.size > 0:
        p5  = np.percentile(valid_imax, 5)
        p95 = np.percentile(valid_imax, 95)
        v = np.clip((imax - p5) / (p95 - p5 + 1e-6), 0.0, 1.0)
    else:
        v = np.clip(imax, 0.0, 1.0)
    # -------------------------------------------------
    # HSV → RGB  — matchar GEE's .hsvToRgb() + gamma=2
    # H = days * croppalet
    # S = magicnorm
    # V = imax.clamp(0,1)
    # -------------------------------------------------

    croppalet = 0.6

    h_ch = np.nan_to_num(days * croppalet, nan=0.0)
    s_ch = np.nan_to_num(magicnorm,        nan=0.0)
    v_ch = np.nan_to_num(v,                nan=0.0)

    # Klipp till [0,1]
    h_ch = np.clip(h_ch, 0.0, 1.0)
    s_ch = np.clip(s_ch, 0.0, 1.0)
    v_ch = np.clip(v_ch, 0.0, 1.0)

    # Riktig HSV -> RGB konvertering
    rgb = hsv_to_rgb(h_ch, s_ch, v_ch)  # (3, H, W)

    # Gamma=2 som GEE's visparams gamma:2  =>  sqrt
    rgb = np.sqrt(np.clip(rgb, 0.0, 1.0))

    # Svarta pixlar där ingen data finns
    for c in range(3):
        rgb[c][no_data_mask] = 0.0

    # -------------------------------------------------
    # RETURN — (H, W, 3)
    # -------------------------------------------------

    H, W = rgb.shape[1], rgb.shape[2]
    if H == 0 or W == 0:
        return {"error": "Output image has zero dimensions"}

    rgb_hwc = np.transpose(rgb, (1, 2, 0))

    extent_out = [
        min(e[0] for e in extents),
        min(e[1] for e in extents),
        max(e[2] for e in extents),
        max(e[3] for e in extents),
    ]

    print(f"Output image: {W}x{H} pixels")

    return {
        "rgb":    rgb_hwc,
        "extent": extent_out
    }