import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import io
import base64
import plotly.graph_objects as go
import os
from reactiv import run_reactiv
from reactiv_evaluation import evaluate_reactiv

st.set_page_config(layout="wide", page_title="REACTIV Detection")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:
    st.title("REACTIV")
    st.markdown("---")

    st.subheader("Time interval")
    start_date = st.date_input("Start", datetime.now() - timedelta(days=30))
    end_date   = st.date_input("End",  datetime.now())

    st.markdown("---")
    data_folder = st.text_input("Data folder", value="/opt/saab/mex/streamlit/data")

    st.markdown("---")
    process_btn = st.button("▶ Run REACTIV", type="primary", use_container_width=True)
    clear_btn   = st.button("🗑 Clear results", use_container_width=True)

    if clear_btn:
        st.session_state.pop("reactiv_result", None)
        st.rerun()

    if "reactiv_result" in st.session_state:
        result = st.session_state["reactiv_result"]
        if "error" not in result:
            st.success("Resultat klart!")
            if st.button("💾 Save result to disk"):
                np.save("/opt/saab/mex/streamlit/results/reactiv_result.npy",
                        st.session_state["reactiv_result"])
                st.success("Saved as reactiv_result.npy")


    

# -------------------------------------------------
# LEGEND
# -------------------------------------------------

def make_legend(start_date, end_date, croppalet=0.6, n=256):
    """Skapa en HSV-färgskala som PNG base64."""
    h = np.linspace(0, croppalet, n)
    s = np.ones(n)
    v = np.ones(n)

    # HSV -> RGB för varje steg
    import colorsys
    colors = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(h[i], s[i], v[i])
        colors.append((int(r*255), int(g*255), int(b*255)))

    # Skapa en 30px hög legend-bild
    legend_h = 30
    legend_arr = np.zeros((legend_h, n, 3), dtype=np.uint8)
    for i, (r, g, b) in enumerate(colors):
        legend_arr[:, i, :] = [r, g, b]

    pil = Image.fromarray(legend_arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

if "reactiv_result" in st.session_state and "error" not in st.session_state["reactiv_result"]:
    legend_b64 = make_legend(start_date, end_date)
    st.markdown("**Legend:**")
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.caption(str(start_date))
    with col2:
        st.markdown(
            f'<img src="data:image/png;base64,{legend_b64}" style="width:100%;height:30px;border-radius:4px;">',
            unsafe_allow_html=True
        )
    with col3:
        st.caption(str(end_date))
    st.markdown("---")

# -------------------------------------------------
# KARTA
# -------------------------------------------------

st.subheader("Map")
st.caption("Zoom to your area of choice  and press Run to run REACTIV")


m = folium.Map(
    location=[39, -100],
    zoom_start=5,
    tiles=None
)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Satellit",
    overlay=False,
    control=True
).add_to(m)

folium.TileLayer("OpenStreetMap", name="OSM", overlay=False, control=True).add_to(m)

# REACTIV-overlay
if "reactiv_result" in st.session_state:
    result = st.session_state["reactiv_result"]
    if "error" not in result:
        rgb_hwc      = result["rgb"]
        ext          = result["extent"]
        no_data_mask = result["no_data_mask"]

        img_uint8 = (np.clip(rgb_hwc, 0, 1) * 255).astype(np.uint8)
        alpha     = np.where(no_data_mask, 0, 255).astype(np.uint8)
        img_rgba  = np.dstack([img_uint8, alpha])
        pil_img   = Image.fromarray(img_rgba, mode="RGBA")
        buf       = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=[[ext[1], ext[0]], [ext[3], ext[2]]],
            opacity=0.85,
            name="REACTIV"
        ).add_to(m)

folium.LayerControl().add_to(m)

map_data = st_folium(
    m,
    width="100%",
    height=600,
    returned_objects=["bounds", "last_clicked", "zoom"],
    key="main_map",
)

# Spara bbox
if map_data and map_data.get("bounds"):
    b = map_data["bounds"]
    sw = b.get("_southWest")
    ne = b.get("_northEast")
    if sw and ne and sw.get("lat") is not None and ne.get("lat") is not None:
        st.session_state["bbox"] = [
            sw["lng"], sw["lat"], ne["lng"], ne["lat"],
        ]
        st.session_state["map_center"] = [
            (sw["lat"] + ne["lat"]) / 2,
            (sw["lng"] + ne["lng"]) / 2,
        ]

if map_data and map_data.get("zoom"):
    st.session_state["map_zoom"] = map_data["zoom"]

# -------------------------------------------------
# TIDSPROFIL VID KLICK
# -------------------------------------------------

if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    click_lat = clicked["lat"]
    click_lon = clicked["lng"]

    if "reactiv_result" in st.session_state:
        result = st.session_state["reactiv_result"]
        if "error" not in result and "amplitude" in result:
            amplitude = result["amplitude"]   # (N, H, W)
            dates     = result["dates"]        # list of ISO strings
            ext       = result["extent"]       # [west, south, east, north]

            H, W = amplitude.shape[1], amplitude.shape[2]

            # Konvertera lat/lon -> pixelkoordinat
            px = int((click_lon - ext[0]) / (ext[2] - ext[0]) * W)
            py = int((ext[3] - click_lat) / (ext[3] - ext[1]) * H)

            if 0 <= px < W and 0 <= py < H:
                values = amplitude[:, py, px]
                dates_dt = [datetime.fromisoformat(d) for d in dates]

                # Filtrera bort NaN
                valid = [(d, v) for d, v in zip(dates_dt, values) if not np.isnan(v)]

                if valid:
                    valid_dates, valid_vals = zip(*valid)

                    st.markdown(f"**Temporal profile ({click_lat:.4f}, {click_lon:.4f})**")

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(valid_dates),
                        y=list(valid_vals),
                        mode="lines+markers",
                        marker=dict(size=6, color="steelblue"),
                        line=dict(width=2, color="steelblue"),
                        name="Amplitud"
                    ))
                    fig.update_layout(
                        xaxis_title="Datum",
                        yaxis_title="Amplitud (normaliserad)",
                        height=300,
                        margin=dict(l=40, r=20, t=20, b=40),
                        plot_bgcolor="white",
                        paper_bgcolor="white"
                    )
                    fig.update_xaxes(showgrid=True, gridcolor="#eee")
                    fig.update_yaxes(showgrid=True, gridcolor="#eee")

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Ingen data för denna pixel.")
            else:
                st.info("Klickad punkt är utanför REACTIV-resultatet.")

# -------------------------------------------------
# PROCESS
# -------------------------------------------------

if process_btn:
    if "bbox" not in st.session_state:
        st.warning("Panorera kartan en gång så att en vy registreras, försök sedan igen.")
    elif start_date >= end_date:
        st.error("Ogiltigt datumintervall.")
    else:
        from reactiv import run_reactiv
        bbox = st.session_state["bbox"]
        with st.spinner("Kör REACTIV-algoritmen..."):
            input_data = {
                "startDate": start_date.isoformat(),
                "endDate":   end_date.isoformat(),
                "bbox":      bbox,
            }
            result = run_reactiv(input_data, data_folder=data_folder)

        st.session_state["reactiv_result"] = result

        if "error" in result:
            st.error(f"Fel: {result['error']}")
        else:
            st.success("Done! The results have been added to the map.")


