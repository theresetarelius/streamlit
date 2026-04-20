import rasterio
import numpy as np

with rasterio.open("REACTIV_GEE_2025-01-01_2025-01-30.tif") as src:
    rgb = src.read([1, 2, 3]).astype(np.float32)
    bounds = src.bounds

reactiv_output = {
    "rgb": np.transpose(rgb, (1, 2, 0)),
    "extent": [bounds.left, bounds.bottom, bounds.right, bounds.top]
}

# Spara som ett dictionary, precis som reactiv_result.npy
np.save("/opt/saab/mex/streamlit/results/reactiv_result_gee.npy", reactiv_output)