# Downloads data needed for data augmentation and training on auxiliary data.

# Political boundaries
curl --create-dirs -O --output-dir data/geocells https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ/geoBoundariesCGAZ_ADM2.geojson

# GADM Country Area Data
curl --create-dirs -O --output-dir data/gadm https://geodata.ucdavis.edu/gadm/gadm4.1/gadm_410-levels.zip
cd data/gadm
unzip gadm_410-levels.zip
cd ../..

# GHSL Population Density Data
curl --create-dirs -O --output-dir data/pop_density https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2022A/GHS_POP_E2020_GLOBE_R2022A_54009_1000/V1-0/GHS_POP_E2020_GLOBE_R2022A_54009_1000_V1_0.zip
cd data/pop_density
unzip GHS_POP_E2020_GLOBE_R2022A_54009_1000_V1_0.zip
cd ../..

# Koppen-Geiger Climate Zone Data
curl --create-dirs -O --output-dir data/koppen_geiger https://figshare.com/ndownloader/files/12407516
cd data/koppen_geiger
unzip Beck_KG_V1.zip
cd ../..