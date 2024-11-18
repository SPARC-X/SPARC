#!/bin/bash
PWD=$(pwd)
for dir in $PWD/*/
do
    dir=${dir%*/}

    if [[ -f "$dir/test_all.py" ]]; then
        echo "Running test_all.py in $dir"
        # Change into the directory and run the script
        (cd "$dir" && python test_all.py)
    fi
done
