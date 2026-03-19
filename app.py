import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import io
import base64

st.set_page_config(layout="wide", page_title="REACTIV Detection")

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:
    st.title("REACTIV")
    st.markdown("---")

    st.subheader("Detection interval")
    start_date = st.date_input("Start", datetime.now() - timedelta(days=30))
    end_date   = st.date_input("End",  datetime.now())

    st.markdown("---")
    data_folder = st.text_input("Data folder", value="/opt/saab/mex/streamlit/data")

    st.markdown("---")
    process_btn = st.button("▶ Run REACTIV", type="primary", use_container_width=True)
    clear_btn   = st.button("🗑 Rensa resultat", use_container_width=True)

    if clear_btn:
        st.session_state.pop("reactiv_result", None)
        st.rerun()

# -------------------------------------------------
# KARTA
# -------------------------------------------------

st.subheader("Map")

m = folium.Map(
    location=[19.42, -155.28],
    zoom_start=11,
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

# REACTIV-overlay om det finns
if "reactiv_result" in st.session_state:
    result = st.session_state["reactiv_result"]
    if "error" not in result:
        rgb_hwc = result["rgb"]
        ext     = result["extent"]

        img_uint8 = (np.clip(rgb_hwc, 0, 1) * 255).astype(np.uint8)
        pil_img   = Image.fromarray(img_uint8)
        buf       = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}",
            bounds=[[ext[1], ext[0]], [ext[3], ext[2]]],
            opacity=0.8,
            name="REACTIV"
        ).add_to(m)

folium.LayerControl().add_to(m)

map_data = st_folium(m, width="100%", height=650, returned_objects=["bounds"])

# Spara bbox från kartans vy
if map_data and map_data.get("bounds"):
    b = map_data["bounds"]
    st.session_state["bbox"] = [
        b["_southWest"]["lng"],
        b["_southWest"]["lat"],
        b["_northEast"]["lng"],
        b["_northEast"]["lat"],
    ]

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
            st.success("Klar! Resultatet visas nu på kartan.")
            st.rerun()