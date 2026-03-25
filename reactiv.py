import os
import re
import numpy as np
import rasterio
import rasterio.windows
from rasterio.windows import from_bounds
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from datetime import datetime
from PIL import Image


def hsv_to_rgb(h, s, v):
    """Vectorized HSV -> RGB, matching GEE's hsvToRgb()."""
    h6 = h * 6.0
    i  = np.floor(h6).astype(np.int32) % 6
    f  = h6 - np.floor(h6)
    p  = v * (1.0 - s)
    q  = v * (1.0 - f * s)
    t  = v * (1.0 - (1.0 - f) * s)

    rgb = np.zeros((3, *h.shape), dtype=np.float32)
    for idx, (r, g, b) in enumerate([
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ]):
        mask = (i == idx)
        rgb[0][mask] = r[mask]
        rgb[1][mask] = g[mask]
        rgb[2][mask] = b[mask]
    return rgb


def reactiv_on_stack(stack, dates, start, end):
    """
    Run REACTIV algorithm on a (N, H, W) stack.
    Returns rgb (3, H, W), no_data_mask (H, W), amplitude (N, H, W), intensity (N, H, W)
    """
    intensity = stack.copy()  # linjär intensitet

    amplitude = np.where(
        np.isnan(stack) | (stack <= 0),
        np.nan,
        np.sqrt(stack)
    )

    no_data_mask = np.all(np.isnan(amplitude), axis=0)

    mean_amp  = np.nanmean(amplitude, axis=0)
    std_amp   = np.nanstd(amplitude,  axis=0)
    mean_safe = np.where(mean_amp == 0, 1e-10, mean_amp)
    magic     = std_amp / mean_safe

    imax = np.nanmax(amplitude, axis=0)

    ds = max((end - start).days, 1)
    days_stack = []
    for img, date in zip(amplitude, dates):
        t        = (date - start).days / ds
        days_img = np.where((~np.isnan(img)) & (img >= imax), t, 0.0)
        days_stack.append(days_img)
    days = np.sum(days_stack, axis=0)

    sizepile  = np.sum(~np.isnan(amplitude), axis=0).astype(np.float32)
    sizepile  = np.where(sizepile < 1, 1.0, sizepile)
    mu        = 0.2286
    stdmu     = 0.1616 / np.sqrt(sizepile)
    magicnorm = np.clip((magic - mu) / (stdmu * 10.0), 0.0, 1.0)

    valid_imax = imax[~no_data_mask]
    if valid_imax.size > 0:
        p5  = np.percentile(valid_imax, 5)
        p95 = np.percentile(valid_imax, 95)
        v   = np.clip((imax - p5) / (p95 - p5 + 1e-6), 0.0, 1.0)
    else:
        v = np.clip(imax, 0.0, 1.0)

    croppalet = 0.6
    h_ch = np.clip(np.nan_to_num(days * croppalet), 0.0, 1.0)
    s_ch = np.clip(np.nan_to_num(magicnorm),        0.0, 1.0)
    v_ch = np.clip(np.nan_to_num(v),                0.0, 1.0)

    rgb = hsv_to_rgb(h_ch, s_ch, v_ch)
    rgb = np.sqrt(np.clip(rgb, 0.0, 1.0))

    for c in range(3):
        rgb[c][no_data_mask] = 0.0

    return rgb, no_data_mask, amplitude, intensity


def run_reactiv(input_data: dict, data_folder: str = None) -> dict:
    """
    Run the REACTIV algorithm on Capella GEO GeoTIFF files using tile-based processing.
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
    # -------------------------------------------------

    tif_files = []
    for root_dir, dirs, files in os.walk(data_folder):
        for f in files:
            if not f.endswith(".tif") or "preview" in f.lower():
                continue
            if not f.startswith("CAPELLA_") or "_GEO_" not in f:
                continue
            parts = f.split("_")
            date_str = None
            for part in parts:
                if re.match(r"^\d{14}$", part):
                    date_str = part[:8]
                    break
            if not date_str:
                continue
            date = datetime.strptime(date_str, "%Y%m%d")
            if not (start <= date <= end):
                continue
            tif_files.append((os.path.join(root_dir, f), date))

        

    print(f"Found {len(tif_files)} Capella .tif files in date range")
    if not tif_files:
        return {"error": "No Capella GEO files found in data folder for the given date range"}

    tif_files = sorted(tif_files, key=lambda x: x[1])

    # -------------------------------------------------
    # DETERMINE OUTPUT GRID
    # Tile-baserad: TILE_SIZE pixlar per tile, max OUTPUT_SIZE totalt
    # -------------------------------------------------

    TILE_SIZE   = 1024   # pixlar per tile (RAM-säkert)
    OUTPUT_SIZE = 4096   # max upplösning på slutbild

    wgs84 = CRS.from_epsg(4326)

    # Beräkna önskad pixelstorlek i grader
    bbox_w  = east - west
    bbox_h  = north - south
    px_size = max(bbox_w, bbox_h) / OUTPUT_SIZE  # grader per pixel

    out_w = int(bbox_w / px_size)
    out_h = int(bbox_h / px_size)
    out_w = min(out_w, OUTPUT_SIZE)
    out_h = min(out_h, OUTPUT_SIZE)

    print(f"Output grid: {out_w}x{out_h} pixels, px_size={px_size:.6f} deg")

    # Antal tiles
    n_tiles_x = int(np.ceil(out_w / TILE_SIZE))
    n_tiles_y = int(np.ceil(out_h / TILE_SIZE))
    print(f"Tiles: {n_tiles_x}x{n_tiles_y} = {n_tiles_x * n_tiles_y} total")

    # Resultat-arrayer
    rgb_full      = np.zeros((3, out_h, out_w), dtype=np.float32)
    nodata_full   = np.ones((out_h, out_w), dtype=bool)
    amp_full      = np.full((len(tif_files), out_h, out_w), np.nan, dtype=np.float32)
    int_full      = np.full((len(tif_files), out_h, out_w), np.nan, dtype=np.float32)

    dates_used = []

    # -------------------------------------------------
    # TILE LOOP
    # -------------------------------------------------

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):

            # Pixelkoordinater för denna tile
            px0 = tx * TILE_SIZE
            py0 = ty * TILE_SIZE
            px1 = min(px0 + TILE_SIZE, out_w)
            py1 = min(py0 + TILE_SIZE, out_h)
            tw  = px1 - px0
            th  = py1 - py0

            # Geografiska koordinater för denna tile
            tile_west  = west  + px0 * px_size
            tile_east  = west  + px1 * px_size
            tile_north = north - py0 * px_size
            tile_south = north - py1 * px_size

            tile_images  = []
            tile_dates   = []
            tile_extents = []

            for file_idx, (path, date) in enumerate(tif_files):
                filename = os.path.basename(path)

                try:
                    with rasterio.open(path) as src:
                        src_crs = src.crs

                        if src_crs and src_crs != wgs84:
                            file_west_f, file_south_f, file_east_f, file_north_f = transform_bounds(
                                src_crs, wgs84, *src.bounds
                            )
                        else:
                            file_west_f, file_south_f, file_east_f, file_north_f = src.bounds

                        clip_west  = max(tile_west,  file_west_f)
                        clip_east  = min(tile_east,  file_east_f)
                        clip_south = max(tile_south, file_south_f)
                        clip_north = min(tile_north, file_north_f)

                        if clip_west >= clip_east or clip_south >= clip_north:
                            continue

                        if src_crs and src_crs != wgs84:
                            native_bounds = transform_bounds(
                                wgs84, src_crs, clip_west, clip_south, clip_east, clip_north
                            )
                        else:
                            native_bounds = (clip_west, clip_south, clip_east, clip_north)

                        window = from_bounds(*native_bounds, transform=src.transform)
                        file_window = rasterio.windows.Window(0, 0, src.width, src.height)
                        window = window.intersection(file_window)

                        if window.width <= 0 or window.height <= 0:
                            continue

                        try:
                            arr = src.read(
                                1,
                                window=window,
                                out_shape=(th, tw),
                                resampling=rasterio.enums.Resampling.average
                            ).astype(np.float32)
                        except Exception as e:
                            print(f"  Skipping corrupt tile in {filename}: {e}")
                            continue

                        nodata = src.nodata
                        if nodata is not None:
                            arr[arr == nodata] = np.nan
                        arr[arr < -1e6] = np.nan
                        arr[arr == 0]   = np.nan

                        valid = arr[~np.isnan(arr)]
                        if valid.size == 0:
                            continue

                        p99_dn = np.percentile(valid, 99)
                        if p99_dn > 0:
                            arr = (arr / p99_dn) ** 2

                        if arr.size == 0 or np.all(np.isnan(arr)):
                            continue

                        tile_images.append((file_idx, arr))
                        if date not in tile_dates:
                            tile_dates.append(date)
                        tile_extents.append((clip_west, clip_south, clip_east, clip_north))

                except Exception as e:
                    print(f"  Error opening {filename}: {e}")
                    continue

            if len(tile_images) < 2:
                continue

            # Bygg stack för denna tile
            stack = np.full((len(tile_images), th, tw), np.nan, dtype=np.float32)
            file_indices = []
            dates_for_stack = []
            for i, (file_idx, arr) in enumerate(tile_images):
                stack[i] = arr
                file_indices.append(file_idx)
                dates_for_stack.append(tif_files[file_idx][1])

            # Kör REACTIV på tile
            rgb_tile, nodata_tile, amp_tile, int_tile = reactiv_on_stack(
                stack, dates_for_stack, start, end
            )

            # Skriv in i resultat-arrayer
            rgb_full[:, py0:py1, px0:px1]    = rgb_tile
            nodata_full[py0:py1, px0:px1]     = nodata_tile
            for i, file_idx in enumerate(file_indices):
                amp_full[file_idx, py0:py1, px0:px1] = amp_tile[i]
                int_full[file_idx, py0:py1, px0:px1] = int_tile[i]

            # Samla datum
            for d in dates_for_stack:
                if d not in dates_used:
                    dates_used.append(d)

            print(f"  Tile ({tx},{ty}) done: {len(tile_images)} images")

    if not dates_used:
        return {"error": "No overlapping Capella images found in date range"}

    dates_used.sort()

    # Transponera rgb till (H, W, 3)
    rgb_hwc = np.transpose(rgb_full, (1, 2, 0))

    extent_out = [west, south, east, north]

    print(f"Output image: {out_w}x{out_h} pixels, {len(dates_used)} dates")

    return {
        "rgb":          rgb_hwc,
        "extent":       extent_out,
        "no_data_mask": nodata_full,
        "amplitude":    amp_full,
        "intensity":    int_full,
        "dates":        [d.isoformat() for d in dates_used],
    }


