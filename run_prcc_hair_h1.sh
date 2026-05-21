#!/usr/bin/env bash
set -u
cd /home/projects/HairFastGAN
export PATH=/home/envs/HairFastGAN/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
exec /home/envs/HairFastGAN/bin/python generate_prcc_hair.py --ref h1:7.png:8.png
