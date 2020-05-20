#!/usr/bin/env python
import os
import shutil
import re
import sys


def read_results(label):
        output = label

        # read and parse output
        with open(output,'r') as f:
            log_text = f.read()
        log_text = log_text.split('SPARC')[-1]  # isolate the most recent run
        body = log_text.rsplit('Timing info')[-2]

        # check that the SCF cycle converged
        converged = re.findall('^.*did not converge to desired.*$', body, re.MULTILINE)
        converged = converged == []

        # Parse out the total energy, energies are in Ha
        eng = re.findall('^.*Free energy per atom               :.*$', body, re.MULTILINE)
        eng = eng[-1].split()[-2]
        eng_per_atom = float(eng)
        # read forces, forces are in Ha/Bohr
        """
        if 'PRINT_FORCES: 1' in log_text:
            forces = []
            for i, force in enumerate(energy_force_block[9:-5]):
                forces.append([float(a) for a in force.split()])

        """
        del f
        
        max_force = re.findall('^.*Maximum force.*$', body, re.MULTILINE)[-1]
        max_force = float(max_force.split()[3])

        # This is the order these values are in in the output        

        results = {
                'energy': eng_per_atom,
                'free_energy': eng_per_atom,
                'converged': converged,
                'max_force': max_force,
                }

        # get number of SCF steps
        f = open(label,'r')
        out = f.read()
        out = out.split('SCF#')
        steps_block = out[-1].split('=' * 68)[1]
        num_steps = len(steps_block.split('\n')) - 3
        results['num_steps'] = num_steps

        # get run time
        time = re.findall('^.*Total walltime.*$', log_text, re.MULTILINE)[-1]
        # f = os.popen('grep "Total walltime" ' + label)
        time = time.split(':')[1]
        time = float(time.split('sec')[0].strip())
        f.close()
        results['time'] = time

        return results

def check_test(label):
    if os.path.isfile(label + '.out'):
        with open(label + '.out', 'r') as f:
            raw_file = f.read()
        terminated_properly = re.findall('^.*Qimen Xu, Abhiraj Sharma.*$', raw_file, re.MULTILINE)
        #s = os.popen('grep "Qimen Xu, Abhiraj Sharma" ' + label + '.out')
        #terminated_properly = s.read() != ''
        terminated_properly = len(terminated_properly) != 0
        if not terminated_properly:
            print('FAILED')
            #print('it looks like {}.out did not terminate properly'.format(label))
            return None
        cur_dir = os.getcwd()
        #run_system(label)
        if label + '.ref' not in os.listdir('.'):
            #os.system('cp {} {}'.format(label + '.out',label + '.ref'))
            shutil.copy(label + '.out', label + '.ref')
            print('No {}.ref was found, copied the .out file that was found to {}.ref'.format(label, label))
            out = read_results(label + '.out')

            conv_check = out['converged']
            f = open(label + '.results','w')
            f.write(label)
            f.write('\n')
                
            f.write('total energy: {} ha\n'.format(out['energy']))
            f.write('max force: {} ha/bohr\n'.format(out['max_force']))
            f.write('timing: {} s\n'.format(out['time']))
            if not conv_check:
                print('Warning this run did not converge\n')
                f.write('convergence check: Warning this run did not converge\n')
            else:
                f.write('convergence check: passed')

        else:
            out = read_results(label + '.out')
            out_ref = read_results(label + '.ref')

            f = open(label + '.results','w')
            f.write(label)
            f.write('\n')
            f.write('total energy: '+str(out['energy'])+' ha\n')
            f.write('max force: '+str(out['max_force'])+' ha/bohr\n')
            f.write('timing: '+str(out['time'])+' s\n')
            f.write('\n')

            # compare results to reference
            eng_diff = abs((out['energy'] - out_ref['energy']))
            max_force_diff = abs((out['max_force'] - out_ref['max_force']))
            time_diff = (out['time'] - out_ref['time'])

            eng_percent_diff = eng_diff
            max_force_percent_diff = max_force_diff
            time_percent_diff = time_diff/out_ref['time']


            if len(sys.argv) > 1: # check if it's a special test type
                test_type = sys.argv[1]
            else:
                test_type = 'default'
            # run checks
            checks = []
            conv_check = out['converged']
            checks.append(conv_check)
            if test_type != 'MD':
                eng_check =  abs(eng_diff) < 0.0001 # 1e-4 ha/atom
                checks.append(eng_check)
            if test_type not in ['MD', 'relax']:
                frc_check = abs(max_force_diff) < 0.001 #
                checks.append(frc_check)
            time_check = time_diff > 0.2
            #print(conv_check, eng_check, frc_check, time_check)
            if checks == [True] * len(checks):
                print('passed')
            else:
                print('FAILED')
            ## convergence:
            if not conv_check:
                f.write('convergence check: Warning this run did not converge\n')
            else:
                f.write('convergence check: passed\n')
            # energy
            if test_type != 'MD':
                if not eng_check:
                    f.write('energy check: Warning the energies are off'
                            ' by '+str(eng_diff * 100)+'%\n')
                else:
                    f.write('energy check: passed\n')
            # forces
            if test_type not in ['MD', 'relax']:
                if not frc_check:
                    f.write('force check: Warning the forces are off by '+str(max_force_diff * 100)+'%\n')
                else:
                    f.write('force check: passed\n')
            # timing
            if time_check:
                f.write('timing check: Warning the timing changed '+str(time_diff * 100)+'% This may not be an issue, but may indicate you are using suboptimal compilation settings\n')
            else:
                f.write('timing check: passed\n')
                #print('timing: passed')
        os.chdir(cur_dir)

cur_dir = os.getcwd()

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if name.endswith('.out'):
            check_test(name[:-4])
            os.chdir(cur_dir)

