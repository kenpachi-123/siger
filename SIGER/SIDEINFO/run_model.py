import argparse
import time

from recbole.quick_start import run_recbole


def _parse_cli_value(raw_value):
    lower = raw_value.lower()
    if lower == "none":
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        if (
            raw_value.startswith("0")
            and raw_value != "0"
            and not raw_value.startswith("0.")
        ):
            raise ValueError
        return int(raw_value)
    except ValueError:
        pass

    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def _parse_unknown_args(unknown_args):
    config_overrides = {}
    index = 0

    while index < len(unknown_args):
        token = unknown_args[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected positional argument: {token}")

        key = token[2:]
        if "=" in key:
            key, value = key.split("=", 1)
            config_overrides[key] = _parse_cli_value(value)
            index += 1
            continue

        if index + 1 < len(unknown_args) and not unknown_args[index + 1].startswith(
            "--"
        ):
            config_overrides[key] = _parse_cli_value(unknown_args[index + 1])
            index += 2
        else:
            config_overrides[key] = True
            index += 1

    return config_overrides


if __name__ == "__main__":
    begin = time.time()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--model", "-m", type=str, default="DIFF", help="name of models"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="Amazon_Beauty", help="name of datasets"
    )
    parser.add_argument(
        "--config_files",
        type=str,
        default="configs/Amazon_Beauty_diff.yaml",
        help="config files",
    )
    args, unknown_args = parser.parse_known_args()

    parameter_dict = {"neg_sampling": None}
    parameter_dict.update(_parse_unknown_args(unknown_args))

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    run_result = run_recbole(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=parameter_dict,
    )
    end = time.time()
    print(end - begin)
