"""Test script to check if SPARC package version is older than
the latest Changelog entry

We generally do not check the other way around since sometimes a trivial bump of SPARC version may be necessary
"""
import re
import json
from datetime import datetime
from pathlib import Path


def load_parameters_json(json_path):
    """
    Load the parameters from the parameters.json file.
    """
    with open(json_path, 'r') as f:
        parameters = json.load(f)
    if "sparc_version" not in parameters:
        raise KeyError("The 'sparc_version' field is missing in parameters.json")
    return parameters["sparc_version"]


def extract_latest_date_from_changelog(changelog_path):
    """
    Extracts the latest date from the changelog file.
    """
    date_patterns = [
        r"(?P<date>\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{1,2}, \d{4})",
        r"(?P<date>\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4})",
    ]

    latest_date = None
    changelog_path = Path(changelog_path)

    with changelog_path.open("r") as f:
        content = f.read()

    for pattern in date_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            try:
                # Convert matched date to datetime object
                parsed_date = datetime.strptime(match, "%b %d, %Y") if "," in match else datetime.strptime(match, "%B %d, %Y")
                if latest_date is None or parsed_date > latest_date:
                    latest_date = parsed_date
            except ValueError:
                continue  # Skip invalid date formats

    if latest_date is None:
        raise ValueError("No valid date found in the changelog.")
    return latest_date


def check_version_against_changelog(parameters_json_path, changelog_path):
    """
    Check if the package version in parameters.json is older than the latest changelog date.
    """
    # Load sparc_version from parameters.json
    sparc_version = load_parameters_json(parameters_json_path)
    version_date = datetime.strptime(sparc_version, "%Y.%m.%d")

    # Extract the latest date from the changelog
    latest_changelog_date = extract_latest_date_from_changelog(changelog_path)

    if version_date < latest_changelog_date:
        print("Version Check Report:")
        print("-" * 60)
        print(f"ERROR: SPARC version ({version_date.strftime('%Y.%m.%d')}) "
              f"is older than the latest changelog date ({latest_changelog_date.strftime('%Y.%m.%d')}).")
        print("Please update initialization.c!")
        print("-" * 60)
        return False
    else:
        print("Version Check Report:")
        print("-" * 60)
        print("SUCCESS:")
        print("-" * 60)
        return True


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Check if package version is up-to-date with the changelog."
    )
    parser.add_argument(
        "parameters_json",
        type=str,
        help="Path to the parameters.json file."
    )
    parser.add_argument(
        "changelog",
        type=str,
        help="Path to the changelog file."
    )

    args = parser.parse_args()

    # Run the version check
    success = check_version_against_changelog(args.parameters_json, args.changelog)
    if not success:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
