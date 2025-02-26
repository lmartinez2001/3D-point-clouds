#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# Download the ModelNet10 dataset
echo "Downloading ModelNet10 dataset..."
wget -P data/ http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip

# Download the ModelNet40 dataset
echo "Downloading ModelNet40 dataset..."
wget -P data/ http://modelnet.cs.princeton.edu/ModelNet40.zip

# Extract ModelNet10
echo "Extracting ModelNet10..."
unzip -q data/ModelNet10.zip -d data/

# Extract ModelNet40
echo "Extracting ModelNet40..."
unzip -q data/ModelNet40.zip -d data/

echo "Download and extraction complete."
echo "ModelNet10 and ModelNet40 datasets are now available in the data/ directory."
