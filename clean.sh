#!/bin/bash
cd "$(dirname "$0")"
echo "Limpando outputs/..."
rm -rf outputs/*
echo "Limpando data/synthetic_data/..."
rm -rf data/synthetic_data/*
echo "Limpando data/tfrecords/..."
rm -rf data/tfrecords/*
echo "Limpando models/unet/checkpoints/..."
rm -rf models/unet/checkpoints/*
echo "Limpando models/unet/unet.keras..."
rm -f models/unet/unet.keras
echo "Limpando __pycache__..."
rm -rf src/__pycache__
echo "Limpeza concluída."
