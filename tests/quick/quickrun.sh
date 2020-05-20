#!/bin/bash

declare -a arr=("BaTiO3" "H2O_sheet" "H2O_wire" "SiH4")

for i in "${arr[@]}"
do
    echo -e "$i \c" | tee -a result
    # remove any existing output
    rm -f ${i}.log ${i}.out* ${i}.aimd* ${i}.geopt* ${i}.static* 
    # run sparc tests
    mpirun -np 1 ../../lib/sparc -name ${i} > ${i}.log
    # check if program runs successfully
    if [ $(grep "The program took" ${i}.log | wc -l) = 1 ]; then
        echo -e "\e[32mPASS\e[0m" | tee -a result
    else
        echo -e "\e[31mFAIL\e[0m" | tee -a result
    fi
done

