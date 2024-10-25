# file ops
import argparse

# from pathlib import Path
import warnings

# from multiprocessing import Process

import numpy as np

# from concurrent.futures import ThreadPoolExecutor
from dask.distributed import Client, LocalCluster
import subprocess

# import functools

# custom
from coralshift.utils import file_ops, config
from coralshift.processing import ml_processing
from coralshift.machine_learning import static_models, ml_results

# from coralshift.plotting import model_results

# import sys
# import logging
import time

# nans in statistics calculations (not a functional problem)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def run_model(
    model_code: str,
    trains,
    tests,
    vals,
    config_info: dict,
    ds_info: dict = None,
    client=None,
    cluster=None,
    extent: tuple[float] = None
):
    tic = time.time()
    # TODO: comment
    model, latest_config_fp = static_models.RunStaticML(
        model_code,
        trains=trains,
        tests=tests,
        vals=vals,
        config_info=config_info,
        additional_info=ds_info if ds_info else None,
        client=client,
        cluster=cluster,
    ).run_model()

    ml_results.AnalyseResults(
        model=model,
        model_code=model_code,
        trains=trains,
        tests=tests,
        vals=vals,
        config_info=file_ops.read_yaml(latest_config_fp),
        extent=extent
    ).analyse_results()

    print(f"\nWriting run results to {str(config.runs_csv)}...\n")
    # write details to csv file
    yaml_dict = file_ops.read_yaml(latest_config_fp)
    # append model_code to results
    yaml_dict["model_code"] = model_code
    # write updated yaml to csv
    file_ops.write_dict_to_csv(yaml_dict, config.runs_csv)

    dur = time.time() - tic

    print(f"\n\nTOTAL DURATION: {np.floor(dur / 60):.0f}m:{dur % 60:.0f}s\n")


def main(
    model_code: str,
    trains,
    tests,
    vals,
    config_info: dict,
):
    if model_code not in [
        "log_reg",
        "lin_reg",
        "max_ent",
        "rf_cf",
        "gb_cf",
        "rf_reg",
        "gb_reg",
        "xgb_cf",
        "xgb_reg",
        "mlp_cf",
        "mlp_reg",
    ]:
        raise ValueError(f"Model type {model_code} not recognised")

    run_model(
        model_code=model_code,
        trains=trains,
        tests=tests,
        vals=vals,
        config_info=file_ops.read_yaml(config.config_dir / "config_test.yaml"),
        # client=client,
        # cluster=cluster,
    )


def execute_command_in_parallel(command):
    process = subprocess.Popen(command, shell=True)
    return process


# make script callable with bash script which specifies model code and config file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model with specified configuration file"
    )
    parser.add_argument(
        "--model_code",
        help="String specifying model architecture",
        default="log_reg",
    )
    parser.add_argument(
        "--config_fp",
        help="Path to the configuration file",
        default=config.config_dir / "config_test.yaml",
    )
    args = parser.parse_args()
    model_code = args.model_code

    # time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    # print log start time
    print(f"START: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}\n")

    # generate data
    (trains, tests, vals), ds_info = ml_processing.ProcessMLData(
        config_info=file_ops.read_yaml(args.config_fp)
    ).generate_ml_ready_data()

    cluster = LocalCluster(n_workers=4, dashboard_address=f":{8786}")
    client = Client(cluster)
    # run model
    main(
        model_code=model_code,
        trains=trains,
        tests=tests,
        vals=vals,
        config_info=file_ops.read_yaml(args.config_fp),
    )

    # subprocess.Popen(" ".join(command), shell=True)

    client.close()
    cluster.close()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Run the model with specified configuration file"
#     )
#     parser.add_argument(
#         "--model_codes",
#         nargs="*",
#         help="String specifying model architectures",
#         default=["rf_reg", "log_reg"],
#     )
#     parser.add_argument(
#         "--config-fp",
#         help="Path to the configuration file",
#         default=config.config_dir / "config_test.yaml",
#     )
#     args = parser.parse_args()

#     # generate data
#     (trains, tests, vals), ds_info = ml_processing.ProcessMLData(
#         config_info=file_ops.read_yaml(args.config_fp)
#     ).generate_ml_ready_data()

#     cluster = LocalCluster(
#         n_workers=len(args.model_codes) * 4, dashboard_address=f":{8786}"
#     )
#     client = Client(cluster)

#     # Run each model code in parallel
#     processes = []
#     for model_code in args.model_codes:
#         log_dir = config.run_logs_dir / model_code
#         if not log_dir.exists():
#             log_dir.mkdir(parents=True, exist_ok=True)
#         log_name = (
#             static_models.RunStaticML(
#                 model_code=model_code, config_info=file_ops.read_yaml(args.config_fp)
#             )
#             .generate_latest_unique_fp()
#             .replace("_CONFIG", "")
#         )
#         log_name = f"{log_name}.log"
#         log_file = log_dir / log_name
#         print(log_file)
#         process = Process(
#             target=main,
#             args=(
#                 model_code,
#                 trains,
#                 tests,
#                 vals,
#                 args.config_fp,
#                 log_file,
#                 # client,
#                 # cluster,
#             ),
#         )
#         process.start()
#         processes.append(process)

# # Wait for all processes to finish
# for process in processes:
#     process.join()
# logging.shutdown()
# client.close()
# cluster.close()


# def main(
#     model_code: str = "log_reg",
#     config_fp: str | Path = config.config_dir / "config_test.yaml",
# ):
#     if model_code not in [
#         "log_reg",
#         "rf_cf",
#         "gb_cf",
#         "rf_reg",
#         "gb_reg",
#         "xgb_cf",
#         "xgb_reg",
#         "mlp_cf",
#         "mlp_reg",
#     ]:
#         raise ValueError(f"Model type {model_code} not recognised")

#     config_info = file_ops.read_yaml(Path(config_fp))
#     run_model(model_code, config_info)


# if __name__ == "__main__":
#     # sys.argv.extend(["log_reg", str(config.config_dir / "config_test.yaml")])
#     parser = argparse.ArgumentParser(
#         description="Run the model with specified configuration file"
#     )
#     parser.add_argument(
#         "model_code",
#         nargs="*",
#         help="String specifying model architecture",
#         default="log_reg",
#     )
#     parser.add_argument(
#         "config_fp",
#         help="Path to the configuration file",
#         nargs="*",
#         default=config.config_dir / "config_test.yaml",
#     )

#     args = parser.parse_args()

#     # set up logging
#     log_dir = config.run_logs_dir / args.model_code
#     log_dir.mkdir(parents=True, exist_ok=True)
#     log_name = (
#         static_models.RunStaticML(
#             model_code=args.model_code, config_info=file_ops.read_yaml(args.config_fp)
#         )
#         .generate_latest_unique_fp()
#         .replace("_CONFIG", "")
#     )
#     log_name = f"{log_name}.log"
#     parser.add_argument(
#         "--log-file",
#         help="Path to the log file",
#         default=log_dir / log_name,
#     )

#     args = parser.parse_args()

#     # Configure logging
#     logging.basicConfig(
#         filename=args.log_file,
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#     )
#     # Redirect stdout and stderr to the log file
#     sys.stdout = open(args.log_file, "a")
#     sys.stderr = open(args.log_file, "a")

#     main(model_code=args.model_code, config_fp=args.config_fp)


# # def main(
# #     config_fp: str | Path = config.config_dir / "config_test.yaml", *model_codes: str
# # ):
# #     if not model_codes:
# #         model_codes = ["log_reg"]  # Default model code if none provided

# #     for model_code in model_codes:
# #         if model_code not in [
# #             "log_reg",
# #             "rf_cf",
# #             "gb_cf",
# #             "rf_reg",
# #             "gb_reg",
# #             "xgb_cf",
# #             "xgb_reg",
# #             "mlp_cf",
# #             "mlp_reg",
# #         ]:
# #             raise ValueError(f"Model type {model_code} not recognised")

# #     config_info = file_ops.read_yaml(Path(config_fp))

# #     # Run the model in parallel for each specified model code
# #     with ThreadPoolExecutor() as executor:
# #         futures = []
# #         for model_code in model_codes:
# #             futures.append(executor.submit(run_model, model_code, config_info))
# #         for future in futures:
# #             future.result()


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(
# #         description="Run the model with specified configuration file"
# #     )
# #     parser.add_argument(
# #         "config_fp",
# #         help="Path to the configuration file",
# #         default=config.config_dir / "config_test.yaml",
# #     )
# #     parser.add_argument(
# #         "model_codes",
# #         nargs="*",
# #         help="List of model codes to run",
# #     )
# #     args = parser.parse_args()
# #     main(config_fp=args.config_fp, *args.model_codes)

# #     # Run the model in parallel for each specified model code
# #     with ThreadPoolExecutor() as executor:
# #         futures = []
# #         for model_code in model_codes:
# #             futures.append(executor.submit(run_model_wrapper, model_code, config_info))
# #         for future in futures:
# #             future.result()
