 #!/bin/bash

mkdir -p testlogs/cmip6_download_logs/


# HadGEM3-GC31-HH
# python3 icenet/download_cmip6_data.py  --source_id HadGEM3-GC31-HH --member_id r1i1p1f1 > logs/cmip6_download_logs/HG_r1i1p1f1.txt 2>&1 &

# BCC-CSM2-HR
# python3 coralshift/dataloading/download_cmip6_data.py  --source_id BCC-CSM2-HR --member_id r1i1p1f1 > logs/cmip6_download_logs/BCC-CSM2-HR.txt 2>&1 &

# EC-Earth3P-HR
python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id mlotst --member_id r1i1p2f1 --lats '-50' --lats '0' --lons '130' --lons '170' --levs '0' --levs '20' --resolution 0.25 > testlogs/cmip6_download_logs/MLOTST.txt 2>&1 &
python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id tos --member_id r1i1p2f1 --lats '-50' --lats '0' --lons '130' --lons '170' --levs '0' --levs '20' --resolution 0.25 > testlogs/cmip6_download_logs/TOS.txt 2>&1 &
python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id rsdo --member_id r1i1p2f1 --lats '-50' --lats '0' --lons '130' --lons '170' --levs '0' --levs '20' --resolution 0.25 > testlogs/cmip6_download_logs/RSDO.txt 2>&1 &


# python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id mlotst --member_id r1i1p2f1 > testlogs/cmip6_download_logs/EC-Earth3P-HR_mlotst.txt 2>&1 &
# python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id so --member_id r1i1p2f1 > testlogs/cmip6_download_logs/EC-Earth3P-HR_so.txt 2>&1 &
# python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id thetao --member_id r1i1p2f1 > testlogs/cmip6_download_logs/EC-Earth3P-HR_thetao.txt 2>&1 &
# python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id uo --member_id r1i1p2f1 > testlogs/cmip6_download_logs/EC-Earth3P-HR_uo.txt 2>&1 &
# python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id vo --member_id r1i1p2f1 > testlogs/cmip6_download_logs/EC-Earth3P-HR_vo.txt 2>&1 &
# python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id tos --member_id r1i1p2f1 > testlogs/cmip6_download_logs/EC-Earth3P-HR_tos.txt 2>&1 &
