#!/usr/bin/env python
import os
import sys
import subprocess
import time

curdir = os.getcwd()

if len(sys.argv) > 1:
    tests_to_run = sys.argv[1]
else:
    tests_to_run = 'all'

failed = False

fails = []  
for root, dirs, files in os.walk(".", topdown=False):
    for name in dirs:
        #if name not in ['relax_LBFGS_test']:
        #    print(name)
        #    continue
        for use_gpu in [0,1]:
            for np in [1,2]:
                print(f"GPU: {use_gpu}, np: {np}")
                os.chdir(name)
                fils = os.listdir('.')
                fil = [a for a in fils if '.inpt' in a][0]
                old_outs = [a for a in fils if '.out' in a]
                for old_out in old_outs:
                    os.remove(old_out)
                fil = fil.split('.')[0]
                # figure out what kind of test it is
                if 'MD' in name:
                    test_type = 'MD'
                elif 'relax' in name:
                    test_type = 'relax'
                else:
                    test_type = 'default'
                # only run the tests that are asked for
                if tests_to_run == 'all':
                    pass
                elif tests_to_run != test_type:
                    os.chdir(curdir)
                    continue
                print(name)
                # run the tests
                start = time.time()
                exit = subprocess.call(f'LIBPCE_USE_GPU={use_gpu} LIBPCE_DEBUG_LEVEL=5 mpirun -np {np}  ../../lib/sparc -name ' + fil + f' > testout_ab_fixed_np_{np}_gpu_{use_gpu}',
                                shell = True,
                                cwd=os.getcwd())
                end = time.time()
                print("Time : {}".format(end - start))
                # read the tests
                subprocess.call('../read_tests.py ' + test_type,
                                shell = True,
                                cwd=os.getcwd())
                if exit:
                    failed = True
                    fails.append(name)
                os.chdir(curdir)

print(fails)
if failed:
    raise Exception('at least one test failed')
