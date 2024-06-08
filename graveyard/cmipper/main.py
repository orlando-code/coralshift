import concurrent.futures

from cmipper import utils, config, parallelised_download_and_process


def main():
    model_info_dict = utils.read_yaml(config.model_info)
    download_config_dict = utils.read_yaml(config.download_config)

    all_download_futures = []  # Store futures from the first set of loops

    # TODO: parallelise by source as well?
    source_ids = ["EC-Earth3P-HR"]
    member_ids = model_info_dict[source_ids[0]]["member_ids"]
    variable_ids = download_config_dict["variable_ids"]

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    utils.execute_functions_in_threadpool,
                    [(source_id, member_id, variable_id)],
                )
                for source_id in source_ids
                for member_id in member_ids
                for variable_id in variable_ids
            ]

            # Wait for all futures to complete
            concurrent.futures.wait(futures)
    except Exception as e:
        print(f"An error occurred: {e}")

    for source_id in model_info_dict.keys():
        for member_id in model_info_dict[source_id]["member_ids"]:
            for var in download_config_dict["variable_ids"]:
                print(f"Processing {source_id}, {member_id}, {var}")
                log_fp = (
                    config.logging_dir
                    / source_id
                    / member_id
                    / "_".join([var, "download.log"])
                )
                if not log_fp.parent.exists():
                    log_fp.parent.mkdir(parents=True)
                # utils.redirect_stdout_stderr_to_file(log_fp)
                futures = utils.execute_functions_in_threadpool(
                    parallelised_download_and_process.download_cmip_variable_data,
                    [source_id, member_id, var],
                )
                all_download_futures.extend(futures)  # Add futures to the list
                utils.reset_stdout_stderr()
                # utils.handle_errors(futures)

    # wait until downloads are complete
    concurrent.futures.wait(all_download_futures)

    # postprocessing of files after all downloaded
    for source_id in model_info_dict.keys():
        for experiment_id in model_info_dict[source_id]["experiment_ids"]:
            for member_id in model_info_dict[source_id]["member_ids"]:
                for var in download_config_dict["variable_ids"]:
                    print(
                        f"Processing {source_id}, {experiment_id}, {member_id}, {var}"
                    )
                    log_fp = (
                        config.logging_dir / source_id / member_id / "processing.log"
                    )
                    if not log_fp.parent.exists():
                        log_fp.parent.mkdir(parents=True)
                    # utils.redirect_stdout_stderr_to_file(log_fp)
                    futures = utils.execute_functions_in_threadpool(
                        parallelised_download_and_process.process_cmip6_data,
                        [(source_id, experiment_id, member_id, var)],
                    )
                    utils.reset_stdout_stderr()
                    # utils.handle_errors(futures)


if __name__ == "__main__":
    main()
