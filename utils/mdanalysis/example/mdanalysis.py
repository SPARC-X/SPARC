import os
import re
import sys
import numpy as np
from scipy.signal import correlate
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from datetime import datetime
import time
import math

class RunInfo:
    def __init__(self):
        self.cell = None
        self.dt_step = None
        self.nelem = None
        self.natom = None
        self.natom_elem = None
        self.mass = None
        self.ion_temp = None
        self.volume = None

class MDTrajInfo:
    def __init__(self):
        self.P_el = None
        self.P_io = None
        self.uen = None
        self.ken = None
        self.pos = None
        self.velocity = None
        self.stress_el = None
        self.stress_io = None

class ParamsClass:
    def __init__(self):
        self.system_name = None
        self.n_folders = None
        self.folder_path_sims = None
        self.n_sims_folders = None

        self.n_equil = None

        self.pcf_flag = 0
        self.range_pcf = None
        self.size_hist_pcf = None

        self.selfD_flag = 0
        self.selfD_block_len = None


        self.interD_flag = 0
        self.interD_block_len = None


        self.viscosity_flag = 0
        self.viscosity_block_len = None

        self.ien_flag = 1
        self.pres_flag = 0

def read_parameters(file_path='./input.params'):
    with open(file_path,'r') as f_params:
        f_params_content = [ line.strip() for line in f_params ]

    for line in f_params_content:
        if re.findall('SYSTEM_NAME:', line) ==['SYSTEM_NAME:']:
            system_name = line.split(':')[1].split()[0]
            break

    for line in f_params_content:
        if re.findall('N_FOLDERS:', line) ==['N_FOLDERS:']:
            n_folders = int(line.split(':')[1].split()[0])
            break

    for line in f_params_content:
        if re.findall('FOLDER_PATH_SIMULATIONS:', line) ==['FOLDER_PATH_SIMULATIONS:']:
            folder_path_sims = line.split(':')[1].split('#')[0].split()
            break

    for line in f_params_content:
        if re.findall('N_SIMULATIONS_FOLDER:', line) ==['N_SIMULATIONS_FOLDER:']:
            n_sims_folders = line.split(':')[1].split('#')[0].split()
            for i in range(len(n_sims_folders)):
                n_sims_folders[i] = int(n_sims_folders[i])
            break

    for line in f_params_content:
        if re.findall('N_EQUIL:', line) ==['N_EQUIL:']:
            n_equil = int(line.split(':')[1].split()[0])
            break

    for line in f_params_content:
        if re.findall('PCF_FLAG:', line) ==['PCF_FLAG:']:
            pcf_flag = int(line.split(':')[1].split()[0])
            break


    for line in f_params_content:
        if re.findall('RANGE_PCF:', line) ==['RANGE_PCF:']:
            range_pcf = float(line.split(':')[1].split()[0])
            break

    for line in f_params_content:
        if re.findall('SIZE_HIST_PCF:', line) ==['SIZE_HIST_PCF:']:
            size_hist_pcf = int(line.split(':')[1].split()[0])
            break

    for line in f_params_content:
        if re.findall('SELF_DIFFUSION_FLAG:', line) ==['SELF_DIFFUSION_FLAG:']:
            selfD_flag = int(line.split(':')[1].split()[0])
            break

    for line in f_params_content:
        if re.findall('BLOCK_LENGTH_SELF_DIFFUSION:', line) ==['BLOCK_LENGTH_SELF_DIFFUSION:']:
            # selfD_block_len = int(line.split(':')[1].split()[0])
            selfD_block_len = list(map(int, re.findall(r'\d+', line)))
            break

    for line in f_params_content:
        if re.findall('INTER_DIFFUSION_FLAG:', line) ==['INTER_DIFFUSION_FLAG:']:
            interD_flag = int(line.split(':')[1].split()[0])
            break

    for line in f_params_content:
        if re.findall('BLOCK_LENGTH_INTER_DIFFUSION:', line) ==['BLOCK_LENGTH_INTER_DIFFUSION:']:
            interD_block_len = int(line.split(':')[1].split()[0])
            break

    for line in f_params_content:
        if re.findall('VISCOSITY_FLAG:', line) ==['VISCOSITY_FLAG:']:
            viscosity_flag = int(line.split(':')[1].split()[0])
            break

    for line in f_params_content:
        if re.findall('BLOCK_LENGTH_VISCOSITY:', line) ==['BLOCK_LENGTH_VISCOSITY:']:
            viscosity_block_len = int(line.split(':')[1].split()[0])
            break

    for line in f_params_content:
        if re.findall('BLOCK_LENGTH_VISCOSITY:', line) ==['BLOCK_LENGTH_VISCOSITY:']:
            viscosity_block_len = int(line.split(':')[1].split()[0])
            break

    for line in f_params_content:
        if re.findall('INTERNAL_ENERGY_FLAG:', line) ==['INTERNAL_ENERGY_FLAG:']:
            ien_flag = int(line.split(':')[1].split()[0])
            break

    for line in f_params_content:
        if re.findall('PRESSURE_FLAG:', line) ==['PRESSURE_FLAG:']:
            pres_flag = int(line.split(':')[1].split()[0])
            break

    parameters = ParamsClass()

    parameters.system_name = system_name
    parameters.n_folders = n_folders
    parameters.folder_path_sims = folder_path_sims
    parameters.n_sims_folders = n_sims_folders

    parameters.n_equil = n_equil

    parameters.pcf_flag = pcf_flag
    parameters.range_pcf = range_pcf
    parameters.size_hist_pcf = size_hist_pcf

    parameters.selfD_flag = selfD_flag
    parameters.selfD_block_len = selfD_block_len


    parameters.interD_flag = interD_flag
    parameters.interD_block_len = interD_block_len


    parameters.viscosity_flag = viscosity_flag
    parameters.viscosity_block_len = viscosity_block_len

    parameters.ien_flag = ien_flag
    parameters.pres_flag = pres_flag

    return parameters

def read_out(fname):
    with open(fname,'r') as f_out:
        f_out_content = [ line.strip() for line in f_out ]

    count = 0
    for line in f_out_content:
        if re.findall('MD_TIMESTEP', line) ==['MD_TIMESTEP']:
            temp = re.findall(r'\b\d+(?:\.\d+)?\b', line)
            dt_step = float(temp[0])
            break
        count=count+1

    count = 0
    for line in f_out_content:
        if re.findall('Total number of atom types', line) ==['Total number of atom types']:
            temp = re.findall('\d+', line)
            nelem = int(temp[0])
            break
        count=count+1

    natom = 0
    natom_elem = []
    mass = []
    cell = []

    count = 0
    for line in f_out_content:
        if re.findall('CELL:', line) ==['CELL:']:
            temp = re.findall('\d+\.\d+', line)
            cell = [float(temp[0]), float(temp[1]), float(temp[2])]
            break
        count=count+1

    count = 0
    for line in f_out_content:
        if re.findall('Total number of atoms', line) ==['Total number of atoms']:
            temp = re.findall('\d+', line)
            natom = int(temp[0])
            break
        count=count+1

    if (nelem > 1):
        count = 0
        for line in f_out_content:
            if re.findall('Number of atoms of type', line) ==['Number of atoms of type']:
                temp = re.findall('\d+', line)
                natom_elem.append(int(temp[1]))
                if (len(natom_elem)==nelem):
                    break
            count=count+1

        if sum(natom_elem) != natom:
            print(natom_elem, natom)
            print("Total number of atoms not matching the sum of atoms of each atom type!\n")
            exit(1)
    else:
        natom_elem.append(natom)

    mass = []
    for line in f_out_content:
        if re.findall('Atomic mass', line) ==['Atomic mass']:
            temp = re.findall(r'\b\d+(?:\.\d+)?\b', line)
            mass.append(float(temp[0]))
            break
        count=count+1

    count = 0
    for line in f_out_content:
        if re.findall('ION_TEMP', line) ==['ION_TEMP']:
            temp = re.findall(r'\b\d+(?:\.\d+)?\b', line)
            ion_temp = float(temp[0])
            break
        count=count+1

    count = 0
    for line in f_out_content:
        if re.findall('Volume', line) ==['Volume']:
            temp = re.findall('-?\d\.\d+[Ee][+\-]\d\d?', line)
            volume = float(temp[0])
            break
        count=count+1

    run_info = RunInfo()

    run_info.cell =  cell
    run_info.dt_step =  dt_step
    run_info.nelem =  nelem
    run_info.natom =  natom
    run_info.natom_elem =  natom_elem
    run_info.mass =  mass
    run_info.ion_temp =  ion_temp
    run_info.volume =  volume

    return run_info

def read_aimd(fname, natom):
    with open(fname,'r') as f_aimd:
        f_aimd_content = [ line.strip() for line in f_aimd ]

    os.system("grep ':PRES:' "+fname+" > press_el.txt")
    os.system("sed -i 's/:PRES://g' press_el.txt")

    os.system("grep ':PRESIG:' "+fname+" > press_io.txt")
    os.system("sed -i 's/:PRESIG://g' press_io.txt")

    os.system("grep ':UEN:' "+fname+" > uen.txt")
    os.system("sed -i 's/:UEN://g' uen.txt")

    os.system("grep ':KENIG:' "+fname+" > ken.txt")
    os.system("sed -i 's/:KENIG://g' ken.txt")


    os.system("grep -A "+str(natom)+" ':R:' "+fname+" > pos.txt")
    os.system("sed -i 's/:R://g' pos.txt")
    os.system("sed -i 's/--//g' pos.txt")


    os.system("grep -A "+str(natom)+" ':V:' "+fname+" > vel.txt")
    os.system("sed -i 's/:V://g' vel.txt")
    os.system("sed -i 's/--//g' vel.txt")

    os.system("grep -A 3 ':STRESS:' "+fname+" > str_el.txt")
    os.system("sed -i 's/:STRESS://g' str_el.txt")
    os.system("sed -i 's/--//g' str_el.txt")

    os.system("grep -A 3 ':STRIO:' "+fname+" > str_io.txt")
    os.system("sed -i 's/:STRIO://g' str_io.txt")
    os.system("sed -i 's/--//g' str_io.txt")

    P_el = np.loadtxt('press_el.txt')
    P_io = np.loadtxt('press_io.txt')

    uen = np.loadtxt('uen.txt')
    ken = np.loadtxt('ken.txt')

    pos = np.loadtxt('pos.txt')
    velocity = np.loadtxt('vel.txt')

    stress_el = np.loadtxt('str_el.txt')
    stress_io = np.loadtxt('str_io.txt')

    os.system("rm press_el.txt press_io.txt uen.txt ken.txt pos.txt vel.txt str_el.txt str_io.txt")
    # os.system("rm press_el.txt press_io.txt pos.txt vel.txt str_el.txt str_io.txt")

    md_info = MDTrajInfo()

    md_info.P_el = P_el
    md_info.P_io = P_io

    md_info.uen = uen
    md_info.ken = ken

    md_info.pos = pos
    md_info.velocity = velocity

    md_info.stress_el = stress_el
    md_info.stress_io = stress_io

    return md_info
def get_Pel_out(fname):


    os.system("grep 'Pressure' "+fname+" > press_el.txt")
    os.system("sed -i 's/Pressure//g' press_el.txt")
    os.system("sed -i 's/(GPa)//g' press_el.txt")
    os.system("sed -i 's/://g' press_el.txt")

    P_el = np.loadtxt('press_el.txt')

    return P_el


def get_equilibriation_step(data, tol=0.1):
    cumulative_avg = np.cumsum(data) / np.arange(1, len(data) + 1)

    equilibration_time_index = np.where(np.abs(np.diff(cumulative_avg)) < tol)[0][0]

    # N_step = len(data)
    # array = list(range(100, 2100, 100))
    # mean_old = np.mean(data)
    # for i in range(len(array)):
    #     mean  = np.mean(data[array[i]:-1])
    #     print(mean_old, mean, np.abs(mean-mean_old), np.abs(mean)*tol*0.01)
    #     if np.abs(mean-mean_old)< np.abs(mean)*tol*0.01:
    #         Equil_step = array[i]
    #         break
    #     mean_old = mean
    return equilibration_time_index

    #TODO

def get_mean(data):
    return np.mean(data)
    #TODO

def blocking_method(data, block_size):
    n = len(data)
    # block_size = n // num_blocks
    num_blocks = n // block_size
    block_means = np.zeros(num_blocks)
    
    for i in range(num_blocks):
        start = i * block_size
        end = start + block_size
        block_means[i] = np.mean(data[start:end])
    
    mean_of_blocks = np.mean(block_means)
    variance_of_blocks = np.var(block_means, ddof=1)  # Unbiased variance estimator
    
    std_dev = np.sqrt(variance_of_blocks / num_blocks)
    
    return mean_of_blocks, std_dev

def find_stagnation_point(arr, threshold=0.05):
    n = len(arr)
    for i in range(1, n):
        # Calculate the percentage change
        percentage_change = (arr[i] - arr[i-1]) / arr[i-1]
        # Check if the growth is less than or equal to the threshold or if it decreases
        if percentage_change <= threshold:
            return i
    return -1  # Return -1 if no stagnation point is found


def get_error_bar_scalar_blocking_method(data):

    total_len = len(data)

    max_b_size = int(math.log(total_len) / math.log(2)) - 1
    bsize = list(range(0, max_b_size, 1))
    errors = []
    means = []


    block_error_data = []
    for i in range(len(bsize)):
        mean_of_blocks, std_dev = blocking_method(data, int(pow(2, bsize[i])))
        block_error_data.append([int(pow(2, bsize[i])), mean_of_blocks, std_dev])
        errors.append(std_dev)
        means.append(mean_of_blocks)



    thresh = np.arange(0.01, 0.41, 0.01)
    for i in range(len(thresh)):
        idx = find_stagnation_point(errors, thresh[i])
        if (idx != -1):
            continue
    error_bar = errors[idx]


    return error_bar, block_error_data

def get_vacf_selfD(data, natom, dt, numStepLength, startStepInter):
    totalNumStepLength = data.shape[0]
    firstStart = 0
    numStartStep = (totalNumStepLength - firstStart - numStepLength) // startStepInter + 1

    lengthStep = (dt*1E-15)/(2.4188843265857E-17) # 1fs=?atu
    unitCoeff = 1.157676

    # Initialize arrays
    vacfWithTime = np.zeros((numStepLength, numStartStep))
    vacfWithTime_norm1 = np.zeros((numStepLength, numStartStep))
    diffWithTime = np.zeros((numStepLength, numStartStep))

    D_collect = []
    # Calculate VACF and diffusion
    for n in range(numStartStep):
        startStep = firstStart + startStepInter * n
        for i in range(startStep, startStep + numStepLength):
            # Calculate the VACF for this step
            vacfWithTime[i - startStep, n] = np.sum(np.sum(data[i] * data[startStep], axis=1), axis=0)
            vacfWithTime_norm1[i - startStep, n] = np.sum(np.sum(data[i] * data[startStep], axis=1), axis=0) / \
                                                   np.sum(np.sum(data[startStep] * data[startStep], axis=1), axis=0)
        
        # Calculate the diffusion with time
        diffWithTime[:, n] = np.cumsum(vacfWithTime[:, n]) * lengthStep / (3 * natom) * unitCoeff
        D_collect.append(diffWithTime[int(numStepLength*0.95), n])

    # Averages
    avgDiffWithTime = np.sum(diffWithTime, axis=1) / numStartStep
    vacfWithTime_norm = np.sum(vacfWithTime_norm1, axis=1) / numStartStep
    avgVacf = np.sum(vacfWithTime, axis=1) / (numStartStep * natom)

    # Cumulative sum for plotting
    diffWithTimeAvg = np.cumsum(avgVacf) * lengthStep / 3 * unitCoeff

    # Time array for plotting
    time_array = np.arange(numStepLength) * dt

    # Get D at 95% of the block length
    D_mean = np.mean(D_collect)
    error_bar_D, block_error_data_D = get_error_bar_scalar_blocking_method(D_collect)


    return time_array, avgVacf, vacfWithTime_norm, diffWithTimeAvg, D_mean, error_bar_D, block_error_data_D

# https://journals.aps.org/pra/abstract/10.1103/PhysRevA.36.1779
def get_vacf_interD(data, natom, cE1, cE2, dt, numStepLength, startStepInter):
    firstStart = 0
    totalNumStepLength = data.shape[0]
    numStartStep = (totalNumStepLength - firstStart - numStepLength) // startStepInter + 1

    lengthStep = (dt*1E-15)/(2.4188843265857E-17) # 1fs=?atu
    unitCoeff = 1.157676/(cE1 * cE2)

    # Initialize arrays
    vacfWithTime = np.zeros((numStepLength, numStartStep))
    vacfWithTime_norm1 = np.zeros((numStepLength, numStartStep))
    diffWithTime = np.zeros((numStepLength, numStartStep))

    D_collect = []
    # Calculate VACF and diffusion
    for n in range(numStartStep):
        startStep = firstStart + startStepInter * n
        for i in range(startStep, startStep + numStepLength):
            # Calculate the VACF for this step
            vacfWithTime[i - startStep, n] = np.sum(data[i] * data[startStep])
            vacfWithTime_norm1[i - startStep, n] = np.sum(data[i] * data[startStep]) / \
                                                   np.sum(data[startStep] * data[startStep])
        
        # Calculate the diffusion with time
        diffWithTime[:, n] = np.cumsum(vacfWithTime[:, n]) * lengthStep / (3* natom) * unitCoeff
        D_collect.append(diffWithTime[int(numStepLength*0.95), n])

    # Averages
    avgDiffWithTime = np.sum(diffWithTime, axis=1) / numStartStep
    vacfWithTime_norm = np.sum(vacfWithTime_norm1, axis=1) / numStartStep
    avgVacf = np.sum(vacfWithTime, axis=1) / (numStartStep* natom)

    # Cumulative sum for plotting
    diffWithTimeAvg = np.cumsum(avgVacf) * lengthStep / 3 * unitCoeff

    # Time array for plotting
    time_array = np.arange(numStepLength) * dt

    # Get D at 95% of the block length
    D_mean = np.mean(D_collect)
    error_bar_D, block_error_data_D = get_error_bar_scalar_blocking_method(D_collect)

    return time_array, avgVacf, diffWithTimeAvg, D_mean, error_bar_D, block_error_data_D


def get_sacf_viscosity(data, dt, volumeBohr, ion_T, numStepLength, startStepInter):

    # Constants
    Bohr3 = (5.29177210903e-11) ** 3  # 1 Bohr^3 = ? m^3
    volume = volumeBohr * Bohr3  # in m^3
    K_BmulT = 1.380649e-23 * ion_T  # J/K*K, in J
    unit = 10 ** 3  # (GPa^2*fs)/J*m^3 = 10^3 Pa*s

    firstStart = 0
    totalNumStepLength = data.shape[0]  # 24000
    numStartStep = (totalNumStepLength  - numStepLength) // startStepInter + 1

    # Initialize arrays
    sacfWithTime = np.zeros((numStepLength, numStartStep))
    viscWithTime = np.zeros((numStepLength, numStartStep))

    eta_collect = []
    # Calculate SACF and viscosity
    for n in range(numStartStep):
        startStep = firstStart + startStepInter * n
        for i in range(startStep, startStep + numStepLength):
            sacfWithTime[i - startStep, n] = np.dot(data[i], data[startStep])
        
        # Integrate over time
        intsacf = np.cumsum(sacfWithTime[:, n]) * dt
        viscWithTime[:, n] = intsacf / 5 * (volume / K_BmulT) * unit  # in Pa*s
        eta_collect.append(viscWithTime[int(numStepLength*0.95), n])

    # Averages
    avgSacf = np.sum(sacfWithTime, axis=1) / (numStartStep * 5)
    avgViscWithTime = np.sum(viscWithTime, axis=1) / numStartStep
    # Time array for plotting
    time_array = np.arange(numStepLength) * dt

    # Get eta at 95% of the block length
    eta_mean = np.mean(eta_collect)
    error_bar_eta, block_error_data_eta = get_error_bar_scalar_blocking_method(eta_collect)

    return time_array, avgSacf, avgViscWithTime, eta_mean, error_bar_eta, block_error_data_eta

# Pair Correlation Function =>
# Input variables:
# equilStep = time step at which equilibration is achieved (will collect
# data after this step)
# stepPcf = intervals at which data is to be used to plot pcf
# limitPcf = total number of time steps to be used for pcf
# histPcf = count the number of atoms in a spherical shell of thickness
# deltaR
# sizeHistPcf = total number of spherical shells to be considered
# rangePcf = maximum radial distance to be  considered 
# cells = vector comprising of length of unit cells
# pos = an array of size [tn_atoms*n_ts,3] containing atomic positions for all time steps
# n_types = number of atom types
# typ_natm = an array containing number of atoms of each type

# Output variables:
# Pcf: pair correlation function for different radial distances
def pcf(equilStep, stepPcf, limitPcf, sizeHistPcf, rangePcf, cells, pos, n_types, typ_natm):
    tn_atoms = sum(typ_natm)
    n_ts = pos.shape[0] // tn_atoms
    last = stepPcf * limitPcf + equilStep
    if last > n_ts:
        raise ValueError(f'Error: "{limitPcf} time steps for calculating PCF @ steps of {stepPcf} are unavailable!"')

    Traj = pos.reshape((n_ts, tn_atoms, 3))
    atm_filter = Traj[equilStep:last:stepPcf]
    deltaR = rangePcf / sizeHistPcf
    npcf = n_types * n_types
    histPcf = np.zeros((sizeHistPcf, npcf))
    Pcf = np.zeros((sizeHistPcf, npcf))
    seq = np.arange(0.5, sizeHistPcf, 1)
    volume = np.prod(cells)
    count = 0


    for typ in range(n_types):
        for j1 in range(count, count + typ_natm[typ]):
            for j2 in range(j1 + 1, count + typ_natm[typ]):
                dr = np.abs(atm_filter[:, j1, :] - atm_filter[:, j2, :])
                dr[:, 0][dr[:, 0] > 0.5 * cells[0]] -= cells[0]
                dr[:, 1][dr[:, 1] > 0.5 * cells[1]] -= cells[1]
                dr[:, 2][dr[:, 2] > 0.5 * cells[2]] -= cells[2]
                rr = np.sqrt(np.sum(dr**2, axis=1))
                n = np.ceil(rr[rr < rangePcf] / deltaR).astype(int)
                if len(n) > 0:
                    element, counts = np.unique(n, return_counts=True)
                    histPcf[element - 1, typ] += counts

        count += typ_natm[typ]
        normfac = volume / (2 * np.pi * (deltaR**3) * typ_natm[typ]**2 * limitPcf)
        Pcf[:, typ] = (histPcf[:, typ] * normfac) / (seq**2)


    cc = n_types
    count = 0

    for typ1 in range(n_types):
        count2 = 0
        for typ2 in range(n_types):
            if typ1 != typ2:
                for j1 in range(count, count + typ_natm[typ1]):
                    for j2 in range(count2, count2 + typ_natm[typ2]):
                        dr = np.abs(atm_filter[:, j1, :] - atm_filter[:, j2, :])
                        dr[:, 0][dr[:, 0] > 0.5 * cells[0]] -= cells[0]
                        dr[:, 1][dr[:, 1] > 0.5 * cells[1]] -= cells[1]
                        dr[:, 2][dr[:, 2] > 0.5 * cells[2]] -= cells[2]
                        rr = np.sqrt(np.sum(dr**2, axis=1))
                        n = np.ceil(rr[rr < rangePcf] / deltaR).astype(int)
                        if len(n) > 0:
                            element, counts = np.unique(n, return_counts=True)
                            histPcf[element - 1, cc] += counts

                normfac = volume / (2 * np.pi * (deltaR**3) * typ_natm[typ1] * typ_natm[typ2] * limitPcf)
                Pcf[:, cc] = (histPcf[:, cc] * normfac) / (seq**2)
                cc += 1

            count2 += typ_natm[typ2]
        count += typ_natm[typ1]

    radial_grid = np.arange(0, rangePcf, deltaR)
    return Pcf, radial_grid

def merge_md_info(md_info_collect, nsims):
    md_info_merged = MDTrajInfo()

    i = 0
    md_info_merged.P_el = md_info_collect[i].P_el
    md_info_merged.P_io = md_info_collect[i].P_io
    md_info_merged.uen = md_info_collect[i].uen
    md_info_merged.ken = md_info_collect[i].ken
    md_info_merged.pos = md_info_collect[i].pos
    md_info_merged.velocity = md_info_collect[i].velocity
    md_info_merged.stress_el = md_info_collect[i].stress_el
    md_info_merged.stress_io = md_info_collect[i].stress_io

    for i in range(nsims-1):
        md_info_temp = md_info_collect[i+1]

        md_info_merged.P_el.append(md_info_temp.P_el)
        md_info_merged.P_io.append(md_info_temp.P_io)
        md_info_merged.fen.append(md_info_temp.fen)
        md_info_merged.ken.append(md_info_temp.ken)
        md_info_merged.pos.append(md_info_temp.pos)
        md_info_merged.velocity.append(md_info_temp.velocity)
        md_info_merged.stress_el.append(md_info_temp.stress_el)
        md_info_merged.stress_io.append(md_info_temp.stress_io)

    return md_info_merged

def plotter(x, y, xlabel, ylabel, plot_title, fig_name):
    plt.figure()  # Create a new figure
    plt.plot(x, y, linewidth=1.4)  # Plot x and y with circle markers
    plt.title(plot_title)  # Set the plot title
    plt.xlabel(xlabel)  # You can customize the labels
    plt.ylabel(ylabel)
    
    plt.grid(True)  # Enable grid
    plt.savefig(fig_name, dpi=300)  # Save the figure with the provided file name
    plt.close()  # Close the plot to free up memory


if __name__ == "__main__":

    homedir = os.getcwd()
    # Reading input parameter file
    params_input_fname = sys.argv[1:][0]
    
    parameters = read_parameters(params_input_fname)


    

    fid_out = open(parameters.system_name+".MDanalysis.out","w")

    now = datetime.now()
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

    fid_out.write("*************************************************************************** \n")
    fid_out.write("*                   MD analysis (Version 05 Sept 2024)                    *\n*                      Date:  "+date_time+"                        * \n")
    fid_out.write("*************************************************************************** \n")

    fid_out.write("Inputs:\n")
    with open(params_input_fname, 'r') as file_in:
        content = file_in.read()
    fid_out.write(content)
    fid_out.write("*************************************************************************** \n")




    # Number of steps in the beginning for equilibriation
    N_equil = parameters.n_equil

    # Reading .out and .aimd files
    list_run_info = []
    list_md_info = []


    for i in range(parameters.n_folders):
        os.chdir(parameters.folder_path_sims[i])
        fout_names = [f"{parameters.system_name}.out"] + [f"{parameters.system_name}.out_{i:02d}" for i in range(1, parameters.n_sims_folders[i])]
        faimd_names = [f"{parameters.system_name}.aimd"] + [f"{parameters.system_name}.aimd_{i:02d}" for i in range(1, parameters.n_sims_folders[i])]

        run_info = read_out(fout_names[0])

        list_run_info.append(run_info)

        md_info_collect = []

        for j in range(parameters.n_sims_folders[i]):
            P_el_temp = get_Pel_out(fout_names[j])
            md_info = read_aimd(faimd_names[j], run_info.natom)
            # md_info.P_el = P_el_temp
            md_info_collect.append(md_info)

        md_info_merged = merge_md_info(md_info_collect, parameters.n_sims_folders[i])
        list_md_info.append(md_info_merged)
        os.chdir(homedir)

    # Internal energy = UEN + KENIG
    if (parameters.ien_flag):
        fid_out.write("Internal energy (UEN + KENIG):\n")
        if (parameters.n_folders==1):

            error_bar_ien, block_error_data_ien = get_error_bar_scalar_blocking_method(list_md_info[0].uen[N_equil:] + list_md_info[0].ken[N_equil:])
            mean_ien = np.mean(list_md_info[0].uen[N_equil:] + list_md_info[0].ken[N_equil:])
            fid_out.write(f"Mean of internal energy [ha/atom]: {mean_ien:.9E}\n")
            fid_out.write(f"Error bar of internal energy [ha/atom]: {error_bar_ien:.3E}\n")
            fid_temp = open("block_error_IEN.txt","w")
            fid_temp.write("block-length    mean    std-dev\n")
            for i in range(len(block_error_data_ien)):
                fid_temp.write(f"{block_error_data_ien[i][0]:.3E} {block_error_data_ien[i][1]:.3E} {block_error_data_ien[i][2]:.3E}\n")
            fid_temp.close()

        else:
            means_ien_collect = []
            for i in range(parameters.n_folders):
                means_ien_collect.append(np.mean(list_md_info[i].uen[N_equil:] + list_md_info[i].ken[N_equil:]))

            mean_ien = np.mean(means_ien_collect)
            error_bar_ien = np.std(means_ien_collect, ddof=1)/np.sqrt(parameters.n_folders)


            fid_out.write(f"Mean of internal energy [ha/atom]: {mean_ien:.9E}\n")
            fid_out.write(f"Error bar of internal energy [ha/atom]: {error_bar_ien:.3E}\n")

            fid_out.write("Means from different folder runs [ha/atom]:\n")
            for i in range(parameters.n_folders):
                fid_out.write(f"{means_ien_collect[i]:.9E}, ")
            fid_out.write("\n")

    
    # Pressure = PRES + PRESIG
    if (parameters.pres_flag):
        fid_out.write("*************************************************************************** \n")
        fid_out.write("Pressure (PRES + PRESIG):\n")
        if (parameters.n_folders==1):
            error_bar_pres, block_error_data_pres = get_error_bar_scalar_blocking_method(
                list_md_info[0].P_el[N_equil:] + list_md_info[0].P_io[N_equil:]
            )
            mean_pres = np.mean(list_md_info[0].P_el[N_equil:] + list_md_info[0].P_io[N_equil:])
            fid_out.write(f"Mean of pressure [GPa]: {mean_pres:.9E}\n")
            fid_out.write(f"Error bar of pressure [GPa]: {error_bar_pres:.3E}\n")
            fid_temp = open("block_error_pressure.txt","w")
            fid_temp.write("block-length    mean    std-dev\n")
            for i in range(len(block_error_data_ien)):
                fid_temp.write(f"{block_error_data_pres[i][0]:.3E} {block_error_data_pres[i][1]:.6E} {block_error_data_pres[i][2]:.3E}\n")
            fid_temp.close()
        else:
            means_pres_collect = []
            for i in range(parameters.n_folders):
                means_pres_collect.append(np.mean(list_md_info[i].P_el[N_equil:] + list_md_info[i].P_io[N_equil:]))

            mean_pres = np.mean(means_pres_collect)
            error_bar_pres = np.std(means_pres_collect, ddof=1)/np.sqrt(parameters.n_folders)

            fid_out.write(f"Mean of pressure [GPa]: {mean_pres:.9E}\n")
            fid_out.write(f"Error bar of pressure [GPa]: {error_bar_pres:.3E}\n")

            fid_out.write("Means from different folder runs [GPa]:\n")
            for i in range(parameters.n_folders):
                fid_out.write(f"{means_pres_collect[i]:.9E}, ")
            fid_out.write("\n")

    
    # Self-diffusion coefficient

    if (parameters.selfD_flag):
        fid_out.write("*************************************************************************** \n")
        fid_out.write("Self diffusion coefficient:\n")
        selfD_mean = np.zeros(list_run_info[0].nelem)
        selfD_errorbar = np.zeros(list_run_info[0].nelem)

        if (parameters.n_folders==1):
            N_MD = len(list_md_info[0].uen)
            velocity = list_md_info[0].velocity.reshape(N_MD, list_run_info[0].natom, 3)
            velocity = velocity[N_equil: ,:, :]

            vacf_mean = []
            D_t_mean = []

            time_array_collect = []
            start_idx = 0
            for i in range(list_run_info[0].nelem):
                end_idx = start_idx + list_run_info[0].natom_elem[i]
                time_array, avgVacf, vacfWithTime_norm, diffWithTimeAvg, D_mean, error_bar_D, block_error_data_D = get_vacf_selfD(
                    velocity[:, start_idx:end_idx,:], 
                    list_run_info[0].natom_elem[i], 
                    list_run_info[0].dt_step, 
                    parameters.selfD_block_len[i], 
                    int(0.1*parameters.selfD_block_len[i])
                )
                vacf_mean.append(avgVacf)
                D_t_mean.append(diffWithTimeAvg)
                time_array_collect.append(time_array)
                start_idx = end_idx
                selfD_mean[i] = D_mean
                selfD_errorbar[i] = error_bar_D

            # fid_vacf = open("vacf.txt","w")
            # fid_vacf.write("t[fs]    VACF-for-different-elementsCF\n")
            # for i in range(len(time_array)):
            #     fid_vacf.write(f"{time_array[i]:.4E} ")
            #     for j in range(list_run_info[0].nelem):
            #         fid_vacf.write(f"{vacf_mean[j][i]:.4E} ")
            #     fid_vacf.write("\n")
            # fid_vacf.close()

            for i in range(list_run_info[0].nelem):
                fid_vacf = open("vacf_"+str(i)+".txt","w")
                fid_vacf.write("t[fs]    VACF\n")
                for j in range(len(time_array_collect[i])):
                    fid_vacf.write(f"{time_array_collect[i][j]:.4E} {vacf_mean[i][j]:.4E}\n")
                fid_vacf.close()

            for i in range(list_run_info[0].nelem):
                fid_D_t = open("D_t_"+str(i)+".txt","w")
                fid_D_t.write("t[fs]    D_t\n")
                for j in range(len(time_array_collect[i])):
                    fid_D_t.write(f"{time_array_collect[i][j]:.4E} {D_t_mean[i][j]:.4E}\n")
                fid_D_t.close()



            # fid_D_t = open("D_t.txt","w")
            # fid_D_t.write("t[fs]    D(t) for elements\n")
            # for i in range(len(time_array_collect[i])):
            #     fid_D_t.write(f"{time_array_collect[i]:.4E} ")
            #     for j in range(list_run_info[0].nelem):
            #         fid_D_t.write(f"{diffWithTimeAvg[j][i]:.4E} ")
            #     fid_D_t.write("\n")
            # fid_D_t.close()

            for j in range(list_run_info[0].nelem):
                plotter(time_array_collect[j], D_t_mean[j], 'time [fs]', 'D(t) for element '+str(j), 'D_t_'+str(j), 'D_t_'+str(j))

            fid_out.write("Self diffusion coefficients [cm^2/s]: \n")
            for i in range(list_run_info[0].nelem):
                fid_out.write(f"{selfD_mean[i]:.4E}, ")
            fid_out.write("\n")
            fid_out.write("Self diffusion coefficients error bar[cm^2/s]: \n")
            for i in range(list_run_info[0].nelem):
                fid_out.write(f"{selfD_errorbar[i]:.4E}, ")
            fid_out.write("\n")
            fid_block_D = open("block_error_D.txt","w")
            for i in range(len(block_error_data_D)):
                fid_block_D.write(f"{block_error_data_D[i][0]:.6E}   {block_error_data_D[i][1]:.6E}   {block_error_data_D[i][2]:.6E}\n")
            fid_block_D.close()

        else:
            selfD_mean_temp = np.zeros((list_run_info[0].nelem, parameters.n_folders))
            

            vacf_collect = []
            D_t_collect = []
            for j in range(parameters.n_folders):
                N_MD = len(list_md_info[j].uen)
                velocity = list_md_info[j].velocity.reshape(N_MD, list_run_info[j].natom, 3)
                velocity = velocity[N_equil: ,:, :]
                vacf_temp = []
                D_t_temp = []
                start_idx = 0
                time_array_collect = []
                for i in range(list_run_info[0].nelem):
                    end_idx = start_idx + list_run_info[0].natom_elem[i]
                    time_array, avgVacf, vacfWithTime_norm, diffWithTimeAvg, D_mean, error_bar_D, block_error_data_D = get_vacf_selfD(
                        velocity[:, start_idx:end_idx,:], 
                        list_run_info[0].natom_elem[i], 
                        list_run_info[0].dt_step, 
                        parameters.selfD_block_len[i], 
                        int(0.1*parameters.selfD_block_len[i])
                    )
                    time_array_collect.append(time_array)
                    vacf_temp.append(avgVacf)
                    D_t_temp.append(diffWithTimeAvg)
                    start_idx = end_idx
                    selfD_mean_temp[i, j] = D_mean
                vacf_collect.append(vacf_temp)
                D_t_collect.append(D_t_temp)

            vacf_mean = []
            D_t_mean = []

            for i in range(list_run_info[0].nelem):
                vacf_temp = np.zeros(len(vacf_collect[j][i]))
                D_t_temp = np.zeros(len(D_t_collect[j][i]))
                for j in range(parameters.n_folders):
                    vacf_temp = vacf_temp + vacf_collect[j][i]
                    D_t_temp = D_t_temp + D_t_collect[j][i]
                vacf_temp =  vacf_temp/parameters.n_folders
                D_t_temp =  D_t_temp/parameters.n_folders
                vacf_mean.append(vacf_temp)
                D_t_mean.append(D_t_temp)

            for i in range(list_run_info[0].nelem):
                fid_vacf = open("vacf_"+str(i)+".txt","w")
                fid_vacf.write("t[fs]    VACF\n")
                for j in range(len(time_array_collect[i])):
                    fid_vacf.write(f"{time_array_collect[i][j]:.4E} {vacf_mean[i][j]:.4E}\n")
                fid_vacf.close()


            # fid_vacf = open("vacf.txt","w")
            # fid_vacf.write("t[fs]    VACF-for-different-elements\n")
            # for i in range(len(time_array)):
            #     fid_vacf.write(f"{time_array[i]:.4E} ")
            #     for j in range(list_run_info[0].nelem):
            #         fid_vacf.write(f"{vacf_mean[j][i]:.4E} ")
            #     fid_vacf.write("\n")
            # fid_vacf.close()

            for j in range(list_run_info[0].nelem):
                plotter(time_array_collect[j], D_t_mean[j], 'time [fs]', 'D(t) for element '+str(j), 'D_t_'+str(j), 'D_t_'+str(j))


            selfD_mean = np.zeros(list_run_info[0].nelem)
            selfD_errorbar = np.zeros(list_run_info[0].nelem)

            for i in range(list_run_info[0].nelem):
                selfD_mean[i] = np.mean(selfD_mean_temp[i,:])
                selfD_errorbar[i] = np.std(selfD_mean_temp[i,:])/np.sqrt(parameters.n_folders)

            fid_out.write("Self diffusion coefficients [cm^2/s]: \n")
            for i in range(list_run_info[0].nelem):
                fid_out.write(f"{selfD_mean[i]:.4E}, ")
            fid_out.write("\n")

            fid_out.write("Self diffusion coefficients error bar[cm^2/s]: \n")
            for i in range(list_run_info[0].nelem):
                fid_out.write(f"{selfD_errorbar[i]:.4E}, ")
            fid_out.write("\n")

            # fid_out.write("Self diffusion coefficients from different folders [cm^2/s]: \n")
            # for i in range(list_run_info[0].nelem):
            #     fid_out.write("element "+str(i)+": ")
            #     for j in range(parameters.n_folders):
            #         fid_out.write(f"{selfD_mean_temp[i][j]:.4E}, ")
            #     fid_out.write("\n")

    
    

    # Inter-diffusion coefficient
    if (parameters.interD_flag):
        fid_out.write("*************************************************************************** \n")
        fid_out.write("Inter diffusion coefficient:\n")
        if (list_run_info[0].nelem != 2):
            print('Inter diffusion coefficient is only implemented for binary systems!!\n')
            exit(1)
        c_E1 = run_info.natom_elem[0]/run_info.natom
        c_E2 = run_info.natom_elem[1]/run_info.natom

        if (parameters.n_folders==1):
            velocity = list_md_info[0].velocity.reshape(N_MD, list_run_info[0].natom, 3)
            Vd = (c_E2* np.sum(
                velocity[:, 0:list_run_info[0].natom_elem[0], :], axis=1
            ) - c_E1* np.sum(
                velocity[:, list_run_info[0].natom_elem[0]:, :], axis=1
            ))

            time_array, avgVacf_interD, InterDiffWithTimeAvg, interD_mean, error_bar_interD, block_error_data_interD = get_vacf_interD(
                Vd, 
                list_run_info[0].natom,
                c_E1,
                c_E2, 
                list_run_info[0].dt_step, 
                parameters.interD_block_len, 
                int(0.3*parameters.interD_block_len)
            )

            fid_vacf = open("vacf_interD.txt","w")
            fid_vacf.write("t[fs]    VACF\n")
            for i in range(len(time_array)):
                fid_vacf.write(f"{time_array[i]:%.4E}  {avgVacf_interD[i]:%.4E} ")
            fid_vacf.close()

            plotter(time_array, InterDiffWithTimeAvg, 'time [fs]', 'inter_D(t)', 'inter_D(t)', 'inter_D(t)')

            fid_out.write("Inter diffusion coefficients [cm^2/s]: \n")
            fid_out.write(f"{interD_mean[i]:.4E}, ")
            fid.out("\n")
            fid_out.write("Inter diffusion coefficients error bar[cm^2/s]: \n")
            fid_out.write(f"{error_bar_interD[i]:.4E}, ")
            fid_out.write("\n")

            fid_block_interD = open("block_error_interD.txt","w")
            for i in range(len(block_error_data_interD)):
                fid_block_interD.write(f"{block_error_data_interD[i][0]:.6E}   {block_error_data_interD[i][1]:.6E}   {block_error_data_interD[i][2]:.6E}\n")
            fid_block_interD.close()
        else:
            
            interD_mean_temp = np.zeros(parameters.n_folders)
            vacf_collect = []
            interD_t_collect = []
            for j in range(parameters.n_folders):
                N_MD = len(list_md_info[j].uen)
                velocity = list_md_info[j].velocity.reshape(N_MD, list_run_info[j].natom, 3)

                Vd = (c_E2* np.sum(
                velocity[:, 0:list_run_info[0].natom_elem[0], :], axis=1
                ) - c_E1* np.sum(
                velocity[:, list_run_info[0].natom_elem[0]:, :], axis=1
                ))

                time_array, avgVacf_interD, InterDiffWithTimeAvg, interD_mean, error_bar_interD, block_error_data_interD = get_vacf_interD(
                Vd, 
                list_run_info[0].natom,
                c_E1,
                c_E2,
                list_run_info[0].dt_step, 
                parameters.interD_block_len, 
                int(0.1*parameters.interD_block_len)
                )
                vacf_collect.append(avgVacf_interD)
                interD_t_collect.append(InterDiffWithTimeAvg)
                interD_mean_temp[j] = interD_mean

            interD_mean = np.mean(interD_mean_temp)
            # error_bar_interD = np.std(interD_mean_temp)/np.sqrt(parameters.n_folders)
            error_bar_interD = np.std(interD_mean_temp)/parameters.n_folders

            vacf_mean = np.zeros(len(vacf_collect[0]))
            interD_t_mean = np.zeros(len(interD_t_collect[0]))
            for i in range(parameters.n_folders):
                vacf_mean = vacf_mean + vacf_collect[i]
                interD_t_mean = interD_t_mean + interD_t_collect[i]

            vacf_mean=vacf_mean/parameters.n_folders
            interD_t_mean=interD_t_mean/parameters.n_folders
            fid_vacf = open("vacf_interD.txt","w")
            fid_vacf.write("t[fs]    VACF\n")
            for i in range(len(time_array)):
                fid_vacf.write(f"{time_array[i]:.4E}  {vacf_mean[i]:.4E} ")
            fid_vacf.close()
            plotter(time_array, interD_t_mean, 'time [fs]', 'inter_D_t', 'inter_D_t', 'inter_D_t')

            fid_out.write("Inter diffusion coefficients [cm^2/s]: \n")
            fid_out.write(f"{interD_mean:.4E}, ")
            fid_out.write("\n")
            fid_out.write("Inter diffusion coefficients error bar[cm^2/s]: \n")
            fid_out.write(f"{error_bar_interD:.4E}, ")

            # fid_out.write("Inter diffusion coefficients from different folders [cm^2/s]: \n")
            # for j in range(parameters.n_folders):
            #     fid_out.write(f"{interD_mean_temp[j]:.4E}, ")
            fid_out.write("\n")


    
    # Shear viscosity
    if (parameters.viscosity_flag):
        fid_out.write("*************************************************************************** \n")
        fid_out.write("Shear viscosity:\n")
        if (parameters.n_folders==1):
            N_MD = len(list_md_info[0].uen)
            stress_viscosity = np.zeros((N_MD, 5))
            stress_viscosity[:, 0] = list_md_info[0].stress_el[0::3, 1] + list_md_info[0].stress_io[0::3, 1]
            stress_viscosity[:, 1] = list_md_info[0].stress_el[1::3, 2] + list_md_info[0].stress_io[1::3, 2]
            stress_viscosity[:, 2] = list_md_info[0].stress_el[0::3, 2] + list_md_info[0].stress_io[0::3, 2]
            stress_viscosity[:, 3] = 0.5*(list_md_info[0].stress_el[0::3, 0] - list_md_info[0].stress_el[1::3, 1]) + 0.5*(list_md_info[0].stress_io[0::3, 0] - list_md_info[0].stress_io[1::3, 1])
            stress_viscosity[:, 4] = 0.5*(list_md_info[0].stress_el[1::3, 1] - list_md_info[0].stress_el[2::3, 2]) + 0.5*(list_md_info[0].stress_io[1::3, 1] - list_md_info[0].stress_io[2::3, 2])

            time_array_sacf, avgSacf, avgViscWithTime, eta_mean, error_bar_eta, block_error_data_eta = get_sacf_viscosity(
                stress_viscosity[N_equil:, :], 
                list_run_info[0].dt_step, 
                list_run_info[0].volume, 
                list_run_info[0].ion_temp, 
                parameters.viscosity_block_len, 
                int(0.3*parameters.viscosity_block_len)
            )

            fid_sacf = open("sacf.txt","w")
            fid_sacf.write("t[fs]    SACF\n")
            for i in range(len(time_array_sacf)):
                fid_sacf.write(f"{time_array_sacf[i]:.4E}  {avgSacf[i]:.4E}\n")
            fid_sacf.close()

            plotter(time_array_sacf, avgViscWithTime, 'time [fs]', 'eta_t', 'eta_t', 'eta_t')

            fid_out.write("Viscosity [Pa-s]: \n")
            fid_out.write(f"{eta_mean:.4E}, ")
            fid_out.write("\n")
            fid_out.write("Viscosity error bar[Pa-s]: \n")
            fid_out.write(f"{error_bar_eta:.4E}, ")
            fid_out.write("\n")

            fid_block_eta = open("block_error_viscosity.txt","w")
            for i in range(len(block_error_data_eta)):
                fid_block_eta.write(f"{block_error_data_eta[i][0]:.6E}   {block_error_data_eta[i][1]:.6E}   {block_error_data_eta[i][2]:.6E}\n")
            fid_block_eta.close()

        else:
            
            eta_mean_temp = np.zeros(parameters.n_folders)
            sacf_collect=[]
            eta_t_collect=[]
            for j in range(parameters.n_folders):
                N_MD = len(list_md_info[j].uen)
                stress_viscosity = np.zeros((N_MD, 5))
                stress_viscosity[:, 0] = list_md_info[j].stress_el[0::3, 1] + list_md_info[j].stress_io[0::3, 1]
                stress_viscosity[:, 1] = list_md_info[j].stress_el[1::3, 2] + list_md_info[j].stress_io[1::3, 2]
                stress_viscosity[:, 2] = list_md_info[j].stress_el[0::3, 2] + list_md_info[j].stress_io[0::3, 2]
                stress_viscosity[:, 3] = 0.5*(list_md_info[j].stress_el[0::3, 0] - list_md_info[j].stress_el[1::3, 1]) + 0.5*(list_md_info[j].stress_io[0::3, 0] - list_md_info[j].stress_io[1::3, 1])
                stress_viscosity[:, 4] = 0.5*(list_md_info[j].stress_el[1::3, 1] - list_md_info[j].stress_el[2::3, 2]) + 0.5*(list_md_info[j].stress_io[1::3, 1] - list_md_info[j].stress_io[2::3, 2])

                time_array_sacf, avgSacf, avgViscWithTime, eta_mean, error_bar_eta, block_error_data_eta = get_sacf_viscosity(
                    stress_viscosity[N_equil:, :], 
                    list_run_info[0].dt_step, 
                    list_run_info[0].volume, 
                    list_run_info[0].ion_temp, 
                    parameters.viscosity_block_len, 
                    int(0.08*parameters.viscosity_block_len)
                )
                sacf_collect.append(avgSacf)
                eta_t_collect.append(avgViscWithTime)

                eta_mean_temp[j] = eta_mean

            eta_mean = np.mean(eta_mean_temp)
            error_bar_eta = np.std(eta_mean_temp)/np.sqrt(parameters.n_folders)


            sacf_mean = np.zeros(len(sacf_collect[0]))
            eta_t_mean = np.zeros(len(eta_t_collect[0]))
            for i in range(parameters.n_folders):
                sacf_mean = sacf_mean + sacf_collect[i]
                eta_t_mean = eta_t_mean + eta_t_collect[i]
            sacf_mean=sacf_mean/parameters.n_folders
            eta_t_mean=eta_t_mean/parameters.n_folders
            fid_sacf = open("sacf.txt","w")
            fid_sacf.write("t[fs]    SACF\n")
            for i in range(len(time_array_sacf)):
                fid_sacf.write(f"{time_array_sacf[i]:.4E}  {sacf_mean[i]:.4E}\n")
            fid_sacf.close()
            plotter(time_array_sacf, eta_t_mean, 'time [fs]', 'eta_t', 'eta_t', 'eta_t')

            fid_out.write("Viscosity [Pa-s]: \n")
            fid_out.write(f"{eta_mean:.4E}, ")
            fid_out.write("\n")
            fid_out.write("Viscosity error bar[Pa-s]: \n")
            fid_out.write(f"{error_bar_eta:.4E}, ")
            fid_out.write("\n")

            # fid_out.write("Viscosity from different folders [Pa-s]: \n")
            # for j in range(parameters.n_folders):
            #     fid_out.write(f"{eta_mean_temp[j]:.4E}, ")
            fid_out.write("\n")
            
    
    if (parameters.pcf_flag):
        N_MD = len(list_md_info[0].uen)
        fid_out.write("*************************************************************************** \n")
        fid_out.write("Pair correlation function:\n")
        if (parameters.n_folders >1):
            print("WARNING: PCF is only calculated from the first folder!!\n")

        equilStep = N_equil
        stepPcf = 1
        limitPcf = len(list_md_info[0].uen)-N_equil
        sizeHistPcf = parameters.size_hist_pcf
        rangePcf = parameters.range_pcf
        cells = run_info.cell

        Pcf, radial_grid = pcf(
            equilStep, 
            stepPcf, 
            limitPcf, 
            sizeHistPcf, 
            rangePcf, 
            cells, 
            list_md_info[0].pos, 
            list_run_info[0].nelem, 
            list_run_info[0].natom_elem)

        fid_pcf = open("pcf_data.txt","w")
        fid_pcf.write("r[bohr]     PCF\n")
        for j in range(len(radial_grid)):
            fid_pcf.write(f"{radial_grid[j]:.6E}  ")
            for i in range(list_run_info[0].nelem*list_run_info[0].nelem):
                fid_pcf.write(f"{Pcf[j, i]:.6E}  ")
            fid_pcf.write("\n")
        fid_pcf.close()
        fid_out.write("Pair correlation function written!!\n")
    fid_out.close()

    
    # plt.figure(figsize=(8, 6))
    # plt.plot(time_array_sacf, avgSacf, label='avgSacf')
    # plt.xlabel('Time')
    # plt.ylabel('avgSacf')
    # plt.title('Time vs. avgSacf')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('time_vs_avgSacf.png')  # Save the plot as a PNG file
    # plt.close()

    # # Plot 2: time_array vs. diffWithTimeAvg
    # plt.figure(figsize=(8, 6))
    # plt.plot(time_array_sacf, avgViscWithTime, label='viscWithTimeAvg', color='orange')
    # plt.xlabel('Time')
    # plt.ylabel('viscWithTimeAvg')
    # plt.title('Time vs. viscWithTimeAvg')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('time_vs_viscWithTimeAvg.png')  # Save the plot as a PNG file
    # plt.close()

    
