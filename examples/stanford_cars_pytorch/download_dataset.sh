#!/bin/bash

# This script downloads the Stanford Cars dataset from a mirror 
# because the official hosting is currently broken.

set -e

echo "Starting Stanford Cars dataset download..."

# Ensure we are in the correct directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Create data directory
mkdir -p data

# Clone the mirror repository
echo "Cloning dataset mirror..."
git clone https://github.com/jhpohovey/StanfordCars.git

# Move data to the expected location
echo "Moving data to ./data/stanford_cars..."
mv StanfordCars/stanford_cars ./data/stanford_cars

# Cleanup
echo "Cleaning up..."
rm -rf StanfordCars

echo "Download complete! Data is located in examples/stanford_cars_pytorch/data/stanford_cars"
