import os
import sys
from shutil import copyfile
import time
import numpy as np

import hubbard_exact
import hubbard_scf
from input_proc import proc_inputfile
import dynamics

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print('Usage: python main.py [opts] inputfile')
        sys.exit()
    start_time = time.time()
    inputfile = args[-1]
    opt = args[0]
    # inputfile = './inputfile'
    sim = proc_inputfile(inputfile)
    sim.print_simulation_info()

    if sim.e_method=='exact':
        sim = hubbard_exact.init_hubbard_exact(sim)
    if sim.e_method=='scf':
        sim = hubbard_scf.init_hubbard_scf(sim)
    sim = dynamics.run_dynamics(sim)



