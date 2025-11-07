# Sea-Bottom Gravity Data and Processing – Gulf of Manfredonia (SW Adriatic Sea)

This repository contains the sea-bottom gravity data, external datasets, and processing codes used for the analysis presented in:

> **Zampa, L.S., Lodolo, E., Creati, N., Busetti, M., Madrussani, G., Forlin, E., & Camerlenghi, A. (2022)**  
> *A Comparison Between Sea-Bottom Gravity and Satellite Altimeter-Derived Gravity in Coastal Environments: A Case Study of the Gulf of Manfredonia (SW Adriatic Sea).*  
> Earth and Space Science, 9, e2020EA001572.  
> https://doi.org/10.1029/2020EA001572

---

## 🌍 Study Area

The Gulf of Manfredonia is located along the southwestern Adriatic Sea (Italy), between the Gargano Promontory and the Apulian foreland.  
It represents an ideal test site because:

- The seafloor is relatively flat and shallow (≤ 90 m);
- High-resolution sea-bottom gravity data exist (OGS60 & OGS83 surveys);
- The area covers the offshore continuation of the Gondola Fault Zone;
- Satellite altimeter gravity (DTU13, S&S) shows strong coastal noise up to ~17 km from the shoreline.

---

## 📁 Repository Structure

SeaBottomGravManfredonia/
├── data/
│ ├── raw/ # Original measurements (not all included here)
│ ├── processed/ # Bouguer anomalies, residuals, grids
│ ├── external/ # DTU13, S&S, GEBCO, EMODnet (linked or via script)
│ └── README.md # Description and citation of datasets
│
├── codes/
│ ├── preprocessing/ # Drift correction, Free-water, Bouguer reduction
│ ├── modelling/ # Forward gravity modelling (Parker, ISVD, Tilt)
│ ├── plotting/ # Scripts to reproduce figures of the paper
│ └── README.md
│
├── docs/
│ ├── paper/ # Reference article (PDF)
│ ├── workflow.md # Explanation of processing steps (optional)
│ └── figures/ # Maps, profiles (optional)
│
├── .gitignore
├── .gitattributes # Git LFS tracking of large files
└── README.md # You are here


> ⚠ **Large files notice:**  
> Some datasets (e.g., GEBCO bathymetry, DTBM TIFF files, shapefiles >100 MB) are not stored in the commit history.  
> They should be downloaded manually or via the scripts in `data/external/`.

---

## ⚙️ Data Processing Summary

The sea-bottom gravity data were corrected and processed following these main steps:

## ⚙️ Processing Workflow (as implemented in the script)

The main Python script performs the following steps:

1. **Import modules and set project paths**  
   Loads base Python modules and custom functions from `modules/`.  
   Creates folders (`figures/`) and defines working directories.

2. **Define study area**  
   Geographic limits (lon/lat WGS84) are manually set and converted to UTM (EPSG:32633).

3. **Import gravity datasets**
   - Sea-bottom gravity (OGS60 campaign, 1960) from CSV  
   - Satellite free-air gravity (Sandwell & Smith) from TXT  
   Bad data points (e.g., station 1311) are removed.

4. **Load DTBM (Digital Terrain/Bathymetry Model)**  
   Raster file `ManfredoniaDBTM.tif` is used for elevation and plotting.

5. **Plot raw DTBM and gravity station distribution**

6. **Correct station depth using DTBM**  
   If |h_obs − h_dtm| > (2 m + 5% of depth), the DTBM depth is used.

7. **Compute Free-Water Anomaly (sea-bottom data only)**  

8. **Topographic Effect (Te)**  
Calculated using `TePrismTess.te()` with:
- Local DTBM  
- GEBCO bathymetry for regional effect  
- Tesseroids for far-field, prisms for near-field  
Stored as `gd['ogs60']['te']`, `gd['ss']['te']`.

9. **Bouguer anomaly**
BA = FA – Te (sea-bottom data)
BA = free_air_sat – Te (satellite data)


10. **Gridding and final maps**  
 Bouguer anomalies are interpolated to a 1 km grid and exported as maps in `/figures`.

---

## 💾 Data Availability

| Dataset |
|---------|
| Sea-bottom gravity (OGS60) |
| Satellite gravity (DTU13, S&S) |
| Bouguer anomaly data | 
| Bathymetry (GEBCO, EMODnet) |
| Shapefiles (coastline, faults) |

---

## 📚 Citation

Zampa, L.S., Lodolo, E., Creati, N., Busetti, M., Madrussani, G., Forlin, E. & Camerlenghi, A. (2022). *A comparison between sea-bottom gravity and satellite altimeter-derived gravity in coastal environments: A case study of the Gulf of Manfredonia (SW Adriatic Sea).* Earth and Space Science, 9, e2020EA001572. https://doi.org/10.1029/2020EA001572

---

## ⚖ License & Usage

This work is released under the **CC BY-NC 4.0** license (scientific, non-commercial use only).  
Third-party datasets referenced must be cited according to their own usage terms.

---

For questions or collaboration, please open an issue or contact: **lzampa@ogs.it**

