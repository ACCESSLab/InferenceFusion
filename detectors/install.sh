#!/usr/bin/env bash

echo "[+] cloning pedestron repository"
git clone https://github.com/hasanirtiza/Pedestron

echo "[+] building docker image"
cd Pedestron
docker build -t pedestron:2.0 .

echo "[+] building inference fusion docker"
cd ..
docker build -t pedestron:2.1 .

