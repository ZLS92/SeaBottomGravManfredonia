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

| Dataset | Type | Source / Reference |
|---------|------|---------------------|
| Sea-bottom gravity (OGS60) | Observed gravity | Ciani, Morelli & Gantar (1960) |
| Satellite gravity (S&S) | Free Air Altimetry-derived Gravity | Sandwell et al. (2014) |
| Bathymetry (DTM/DTBM) | Topographic-Bathymetric model | GEBCO Compilation Group (2023) | CC BY 4 0 |
| EMODnet DTM (2020) | European seas bathymetry | EMODnet Bathymetry Consortium (2020) | 1/16° (~115m) | EMODnet terms |
---

## 📚 Citation

Zampa, L.S., Lodolo, E., Creati, N., Busetti, M., Madrussani, G., Forlin, E. & Camerlenghi, A. (2022). *A comparison between sea-bottom gravity and satellite altimeter-derived gravity in coastal environments: A case study of the Gulf of Manfredonia (SW Adriatic Sea).* Earth and Space Science, 9, e2020EA001572. https://doi.org/10.1029/2020EA001572

## 📁 Data Sources and Required Acknowledgements

### **🟦 Sea-bottom gravity data (OGS60 survey)**  
Original data acquired during the Italian continental shelf gravimetric survey:

**Ciani, A., Morelli, C., & Gantar, C. (1960).**  
*Rilievo gravimetrico sullo zoccolo epicontinentale dei mari Italiani.*  
Bollettino di Geofisica Teorica ed Applicata, **6**, 101.

### **🟩 Satellite-altimetry gravity data (CryoSat-2 & Jason-1)**  

**Sandwell, D. T., Müller, R. D., Smith, W. H. F., Garcia, E., & Francis, R. (2014).**  
*New global marine gravity model from CryoSat-2 and Jason-1 reveals buried tectonic structure.*  
Science, 346(6205), 65–67. https://doi.org/10.1126/science.1258213

### **🌍 Global Bathymetry / Topography – GEBCO_2023**

**GEBCO Compilation Group (2023).**  
*GEBCO 2023 Grid.*  
https://doi.org/10.5285/f98b053b-0cbc-6c23-e053-6c86abc0af7b  
Licensed under **CC BY 4.0**.

### **🇪🇺 Regional Bathymetry – EMODnet DTM**

If EMODnet data were used for coastal/topography refinement, please cite:

**EMODnet Bathymetry Consortium (2020).**  
*EMODnet Digital Bathymetry (DTM).*  
https://doi.org/10.12770/bb6a87dd-e579-4036-abe1-e649cea9881a

(Older grid versions: 2018 → DOI 10.12770/18ff0d48-b203-4a65-94a9-5fd8b0ec35f6;  
2016 → DOI 10.12770/c7b53704-999d-4721-b1a3-04ec60c87238)

### 🧩 Software Citation – Harmonica (Fatiando a Terra)

This repository makes use of the open-source **Harmonica** library (Fatiando a Terra project) for gravity data processing and topographic effect computation.

If you use this repository or reuse the processing workflow/code, please also cite:

Uieda, L., et al. (2020).  
**Harmonica: Forward modeling, inversion, and processing gravity and magnetic data.**  
Zenodo. https://doi.org/10.5281/zenodo.3628741

> *We kindly ask users to acknowledge both this repository and the Harmonica project in any derived scientific publication.*

---

## ⚖ License & Usage

This work is released under the **CC BY-NC 4.0** license (scientific, non-commercial use only).  
Third-party datasets referenced must be cited according to their own usage terms.

---

For questions or collaboration, please open an issue or contact: **lzampa@ogs.it**

