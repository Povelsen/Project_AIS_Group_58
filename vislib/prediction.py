# vislib/prediction.py
import folium
import numpy as np
import os

def make_map(inputs, preds, lat_range, lon_range, save_path):
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    center = [(lat_min+lat_max)/2, (lon_min+lon_max)/2]

    m = folium.Map(location=center, zoom_start=8)
    folium.PolyLine(inputs, color="blue").add_to(m)
    folium.PolyLine(preds, color="red").add_to(m)
    m.save(save_path)
