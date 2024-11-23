"""Test script to check parameters coverage in examples
"""
import json
import os
from pathlib import Path
from sparc.io import _read_inpt

def load_parameters_json(json_path):
    """
    Load the parameters from the parameters.json file.
    """
    with open(json_path, 'r') as f:
        parameters = json.load(f)
    if "parameters" not in parameters:
        raise KeyError("The 'parameters' field is missing in parameters.json")
    return parameters["parameters"]

def check_missing_parameters(test_dir, parameters_json_path):
    """
    Check for missing parameters in the documentation.
    test_dir must be structured in <name>/standard/<name>.inpt
    """
    test_dir = Path(test_dir)
    documented_parameters = load_parameters_json(parameters_json_path)

    # Iterate through the .inpt files and check for missing parameters
    report = {}
    for match_file in test_dir.glob("*/standard/*.inpt"):
        test_name = match_file.stem
        try:
            inpt_data = _read_inpt(match_file)
            params_in_file = inpt_data["inpt"]["params"]
        except Exception:
            # Something could be buggy with SPARC-X-API?
            pass

        # Check missing or typo parameters
        missing_params = [
            param for param in params_in_file
            if (param.upper() not in documented_parameters)
            # TODO: Obsolete BOUNDARY_CONDITION keyword
            and (param.upper() != "BOUNDARY_CONDITION")
        ]
        if missing_params:
            report[test_name] = missing_params

    # Generate report and return error if missing parameters are found
    if report:
        print("Missing / Incorrect Parameters Report:")
        print("-" * 60)
        for file_path, missing_params in report.items():
            print(f"Test name: {file_path}")
            print(f"Missing Parameters: {', '.join(missing_params)}")
            print("-" * 60)
        return False
    else:
        print("All parameters are documented correctly.")
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Check for missing / incorrect parameters in SPARC examples."
    )
    parser.add_argument(
        "test_directory",
        type=str,
        help="Path to the directory containing test .inpt files."
    )
    parser.add_argument(
        "parameters_json",
        type=str,
        help="Path to the parameters.json file."
    )

    args = parser.parse_args()

    # Run the check
    success = check_missing_parameters(args.test_directory,
                                       args.parameters_json)
    if not success:
        exit(1)
    else:
        exit(0)
if __name__ == "__main__":
    main()
