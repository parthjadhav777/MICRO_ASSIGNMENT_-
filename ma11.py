import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, Polygon, box
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
# ----------------------------------------------------
# 1. Generate Dummy GeoJSON Data
# ----------------------------------------------------
def create_dummy_data():
    # Points (GPS probes)
    points = [
        Point(np.random.uniform(72.8, 73.0), np.random.uniform(19.0, 19.2))
        for _ in range(10)
    ]
    gpd.GeoDataFrame(geometry=points, crs="EPSG:4326").to_file(
        "points.geojson", driver="GeoJSON"
    )

    # Line (trajectory)
    lines = [
        LineString([(72.8, 19.0), (72.9, 19.1), (72.85, 19.2)])
    ]
    gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326").to_file(
        "lines.geojson", driver="GeoJSON"
    )

    # Polygon (AOI / building)
    polys = [
        Polygon([(72.8, 19.0), (72.9, 19.0), (72.9, 19.1), (72.8, 19.1)])
    ]
    gpd.GeoDataFrame(geometry=polys, crs="EPSG:4326").to_file(
        "polygons.geojson", driver="GeoJSON"
    )

create_dummy_data()

# ----------------------------------------------------
# 2. Load Data
# ----------------------------------------------------
files = {
    "Points": "points.geojson",
    "Lines": "lines.geojson",
    "Polygons": "polygons.geojson",
}

gdfs = {name: gpd.read_file(path) for name, path in files.items()}

# ----------------------------------------------------
# 3. Metadata + Sanity Checks
# ----------------------------------------------------
print(f"{'Type':<10} | {'CRS':<15} | {'Count':<5} | {'Bounds'}")
print("-" * 75)

for name, gdf in gdfs.items():
    bounds = np.round(gdf.total_bounds, 4)
    crs_type = "Unset"
    if gdf.crs:
        crs_type = "Projected" if gdf.crs.is_projected else "Geographic"
    print(f"{name:<10} | {crs_type:<15} | {len(gdf):<5} | {bounds}")

# ----------------------------------------------------
# 4. Helper: plot layer with its bounding box
# ----------------------------------------------------

def plot_with_bbox(gdf, title, color, geom_type):
    minx, miny, maxx, maxy = gdf.total_bounds

    bbox_geom = gpd.GeoDataFrame(
        geometry=[box(minx, miny, maxx, maxy)],
        crs=gdf.crs
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    gdf.plot(
        ax=ax,
        color=color,
        markersize=50 if geom_type == "point" else None,
        linewidth=2 if geom_type == "line" else None,
        alpha=0.7
    )

    bbox_geom.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linestyle="--",
        linewidth=2
    )

    # ---- Proper legend proxies ----
    legend_elements = []

    if geom_type == "point":
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=color, markersize=8, label='Points')
        )
    elif geom_type == "line":
        legend_elements.append(
            Line2D([0], [0], color=color, lw=2, label='Line')
        )
    else:  # polygon
        legend_elements.append(
            Patch(facecolor=color, edgecolor=color, alpha=0.7, label='Polygon')
        )

    legend_elements.append(
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Bounding Box')
    )

    ax.legend(handles=legend_elements)

    ax.set_xlim(minx - 0.01, maxx + 0.01)
    ax.set_ylim(miny - 0.01, maxy + 0.01)
    ax.set_aspect("equal")
    ax.set_title(title)

    plt.show()




# ----------------------------------------------------
# 6. Useful / Important GeoPandas Operations
# ----------------------------------------------------

# A. Reproject to metric CRS (critical for distance/area)
gdfs_m = {k: v.to_crs(epsg=32643) for k, v in gdfs.items()}  # UTM zone for India

# B. Geometry validity check
for name, gdf in gdfs.items():
    print(f"{name} valid geometries:", gdf.is_valid.all())

# C. Spatial relationship: which points fall inside polygon?
points_in_poly = gpd.sjoin(
    gdfs["Points"],
    gdfs["Polygons"],
    predicate="within",
    how="inner"
)
print("Points inside polygon:", len(points_in_poly))

# D. Length & area (metric CRS only)
line_length_m = gdfs_m["Lines"].length.iloc[0]
polygon_area_m2 = gdfs_m["Polygons"].area.iloc[0]

print(f"Line length (meters): {line_length_m:.2f}")
print(f"Polygon area (sq meters): {polygon_area_m2:.2f}")

# E. Distance from points to line
distances = gdfs_m["Points"].distance(gdfs_m["Lines"].geometry.iloc[0])
print("Point-to-line distances (m):")
print(np.round(distances.values, 2))
# ----------------------------------------------------
# 5. Individual Proper Plots
# ----------------------------------------------------
plot_with_bbox(gdfs["Points"], "Points + Bounding Box", "red", "point")
plot_with_bbox(gdfs["Lines"], "Lines + Bounding Box", "green", "line")
plot_with_bbox(gdfs["Polygons"], "Polygons + Bounding Box", "blue", "polygon")