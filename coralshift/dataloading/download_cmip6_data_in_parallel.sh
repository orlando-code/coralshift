#!/bin/bash

mkdir -p logs/cmip6_download_logs/


# HadGEM3-GC31-HH
# python3 icenet/download_cmip6_data.py  --source_id HadGEM3-GC31-HH --member_id r1i1p1f1 > logs/cmip6_download_logs/HG_r1i1p1f1.txt 2>&1 &

# BCC-CSM2-HR
python3 coralshift/dataloading/download_cmip6_data.py  --source_id BCC-CSM2-HR --member_id r1i1p1f1 > logs/cmip6_download_logs/BCC-CSM2-HR.txt 2>&1 &

# EC-Earth3P-HR
# python3 icenet/download_cmip6_data.py  --source_id EC-Earth3P-HR --member_id r2i1p2f1 > logs/cmip6_download_logs/EC-Earth3P-HR.txt 2>&1 &
