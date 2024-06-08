 #!/bin/bash


# python3 run_model.py --model_code log_reg > /maps/rt582/coralshift/logs/runs/log_reg/IDX2_cmip6_unep_gebco.log 2>&1 &
# python3 run_model.py --model_code gb_reg > /maps/rt582/coralshift/logs/runs/gb_reg/IDX2_cmip6_unep_gebco.log 2>&1 &
# python3 run_model.py --model_code mlp_reg > /maps/rt582/coralshift/logs/runs/mlp_reg/IDX2_cmip6_unep_gebco.log 2>&1 &
python3 run_model.py --model_code rf_reg > /maps/rt582/coralshift/logs/runs/rf_reg/IDX3_cmip6_unep_gebco.log 2>&1 &
# python3 run_model.py --model_code xgb_reg > /maps/rt582/coralshift/logs/runs/xgb_reg/IDX2_cmip6_unep_gebco.log 2>&1 &





# # EC-Earth3P-HR
# # python3 parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id mlotst --member_id r1i1p2f1 > logs/cmip6_download_logs/mlotst_download.log 2>&1 &
# python3 parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id tos --member_id r1i1p2f1 > logs/cmip6_download_logs/tos_download.log 2>&1 &
# # python3 parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id rsdo --member_id r1i1p2f1 > logs/cmip6_download_logs/rsdo_download.log 2>&1 &
# # python3 parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id so --member_id r1i1p2f1 > logs/cmip6_download_logs/so_download.log 2>&1 &
# # python3 parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id thetao --member_id r1i1p2f1 > logs/cmip6_download_logs/thetao_download.log 2>&1 &
# # python3 parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id hfds --member_id r1i1p2f1 > logs/cmip6_download_logs/hfds_download.log 2>&1 &
# # python3 parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id umo --member_id r1i1p2f1 > logs/cmip6_download_logs/umo_download.log 2>&1 &
# # python3 parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id uo --member_id r1i1p2f1 > logs/cmip6_download_logs/uo_download.log 2>&1 &
# # python3 parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id vmo --member_id r1i1p2f1 > logs/cmip6_download_logs/vmo_download.log 2>&1 &
# # python3 parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id vo --member_id r1i1p2f1 > logs/cmip6_download_logs/vo_download.log 2>&1 &
# python3 parallelised_download_and_process.py --source_id EC-Earth3P-HR --variable_id wfo --member_id r1i1p2f1 > logs/cmip6_download_logs/wfo_download.log 2>&1 &


# # POTENTIAL AUTOMATION
# # # Read values from YAML config file
# # yaml_config_file_path="download_config.yaml"

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
