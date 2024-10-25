 #!/bin/bash

mkdir -p testlogs/cmip6_process_logs/

# EC-Earth3P-HR
python3 cmipper/parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id mlotst --experiment_id hist-1950 --member_id r1i1p2f1 --command process > testlogs/cmip6_process_logs/mlotst_process.log 2>&1 &
# python3 cmipper/parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id tos --member_id r1i1p2f1 > testlogs/cmip6_download_logs/tos_download.log 2>&1 &
# python3 cmipper/parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id rsdo --member_id r1i1p2f1 > testlogs/cmip6_download_logs/rsdo_download.log 2>&1 &
# python3 cmipper/parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id so --member_id r1i1p2f1 > testlogs/cmip6_download_logs/so_download.log 2>&1 &
# python3 cmipper/parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id thetao --member_id r1i1p2f1 > testlogs/cmip6_download_logs/thetao_download.log 2>&1 &
# python3 cmipper/parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id hfds --member_id r1i1p2f1 > testlogs/cmip6_download_logs/hfds_download.log 2>&1 &
# python3 cmipper/parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id umo --member_id r1i1p2f1 > testlogs/cmip6_download_logs/umo_download.log 2>&1 &
# python3 cmipper/parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id uo --member_id r1i1p2f1 > testlogs/cmip6_download_logs/uo_download.log 2>&1 &
# python3 cmipper/parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id vmo --member_id r1i1p2f1 > testlogs/cmip6_download_logs/vmo_download.log 2>&1 &
# python3 cmipper/parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id vo --member_id r1i1p2f1 > testlogs/cmip6_download_logs/vo_download.log 2>&1 &
# python3 cmipper/parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id wfo --member_id r1i1p2f1 > testlogs/cmip6_download_logs/wfo_download.log 2>&1 &


# POTENTIAL AUTOMATION
# # Read values from YAML config file
# yaml_config_file_path="download_config.yaml"

# source_id = EC-Earth3P-HR
# variable_ids=$(yaml_config_file_path variable_ids)
# member_ids=$(yaml_config_file_path member_ids)

# # Iterate over variable IDs
# for variable_id in $variable_ids; do
#     # Iterate over member IDs
#     for member_id in $member_ids; do
#         # Generate command dynamically
#         command="python3 cmipper/parallelised_download_and_process.py --source_id $source_id --variable_id $variable_id --member_id $member_id > testlogs/cmip6_download_logs/${member_id}/${variable_id}_download.txt 2>&1 &"
        
#         # Execute the command
#         eval $command
#     done
# done

# python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id mlotst --member_id r1i1p2f1 > testlogs/cmip6_download_logs/EC-Earth3P-HR_mlotst.txt 2>&1 &
# python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id so --member_id r1i1p2f1 > testlogs/cmip6_download_logs/EC-Earth3P-HR_so.txt 2>&1 &
# python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id thetao --member_id r1i1p2f1 > testlogs/cmip6_download_logs/EC-Earth3P-HR_thetao.txt 2>&1 &
# python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id uo --member_id r1i1p2f1 > testlogs/cmip6_download_logs/EC-Earth3P-HR_uo.txt 2>&1 &
# python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id vo --member_id r1i1p2f1 > testlogs/cmip6_download_logs/EC-Earth3P-HR_vo.txt 2>&1 &
# python3 coralshift/dataloading/download_cmip6_data_parallel.py --source_id EC-Earth3P-HR --variable_id tos --member_id r1i1p2f1 > testlogs/cmip6_download_logs/EC-Earth3P-HR_tos.txt 2>&1 &
