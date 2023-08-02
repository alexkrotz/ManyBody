
class simulation:
    def __init__(self, defaults):
        self.Ns0 = defaults['Ns0']
        self.Ns = 2*self.Ns0
        self.Np = defaults['Np']
        self.nprocs = defaults['nprocs']
        self.trials = defaults['trials']
        self.u = defaults['u']
        self.t = defaults['t']
        self.g = defaults['g']
        self.w = defaults['w']
        self.temp = defaults['temp']
        self.kB = defaults['kB']
        self.e_method = defaults['e_method']
        self.dyn_method = defaults['dyn_method']
        self.tmax = defaults['tmax']
        self.dt = defaults['dt']
        self.dt_bath = defaults['dt_bath']
        if self.nprocs >= 8:
            self.use_tqdm = False
        else:
            self.use_tqdm = True

        self.filename = './'+str(self.Ns)+'_'+str(self.Np)+'_'+str(self.u)+'_'+str(self.t)+'_'+str(self.g)\
                        +'_'+str(self.w)+'_'+str(self.temp)+'_'+str(self.e_method) + '_' + str(self.dyn_method)\
                        +'_'+str(self.tmax)+'_'+str(self.dt)+'_'+str(self.dt_bath)

    def print_simulation_info(self):
        print('##### SIMULATION INFO #####')
        print("tmax: ", self.tmax)
        print("output dt: ", self.dt)
        print("bath dt: ", self.dt_bath)
        print('temperature: ', self.temp)
        print('nprocs: ', self.nprocs)
        print('trials: ', self.trials)
        print('electronic structure method: ', self.e_method)
        print('dynamics method: ', self.dyn_method)
        print('u: ', self.u)
        print('t: ', self.t)
        print('g: ', self.g)
        print('w: ', self.w)
        print('Number of spinorbitals: ', self.Ns)
        print('Number of electrons: ', self.Np*2)
        print("###########################")






def proc_inputfile(inputfile):
    # defaults
    # E_method: "SCF" or "exact"
    # dyn_method: "MF" or "FSSH" or ....
    defaults = {"Ns0":4,"Np":1,"nprocs":8,"trials":1,"u":1,"t":1,"g":1,"w":1,"temp":1,"kB":1,"e_method":"SCF",\
                "dyn_method":"MF","tmax":10,"dt":0.1,"dt_bath":0.02}
    opts = list(defaults)
    input_params = {}
    with open(inputfile) as f:
        for line in f:
            exec(str(line), input_params)  # locals())
    inputs = list(input_params)
    for key in inputs:
        defaults[key] = input_params[key]
    sim = simulation(defaults)
    return sim