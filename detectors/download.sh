#!/usr/bin/env bash

echo "[+] install gdown"
pip install gdown

echo "[+] downloading weights"
mkdir pretrained_models && cd pretrained_models

weights=( "12mWjmWBv-4wra8hCNF71Wq0U2va0Ai8e" "1u2YTq1ttK1Nn-XWA0TbTjyerq7-LH_EE" "1vcB4PrS0Vpo48f27QpGtJPWo9iwGs5Vl" "14qpoyQWIirzUyLZHTxjZe-09AxiUtIxK")
models=( "Cascade Mask R-CNN HRNet" "Cascade Mask R-CNN MobileNet" "MGAN" "CSP" )

for i in 0 1 2 3
do
   echo "Donwloading ${models[i]} times"
   gdown https://drive.google.com/uc?id=${weights[i]}
done
