import numpy as np
import ray
import functions
import scipy
from os import path


@ray.remote
def run_traj_scf(sim, index, seed):
    np.random.seed(seed)
    tdat = np.arange(0, sim.tmax + sim.dt, sim.dt)
    tdat_bath = np.arange(0, sim.tmax + sim.dt_bath, sim.dt_bath)
    psi_db_mat = functions.fill_energy(sim)
    rhoDb_out = np.zeros((len(tdat), 2*sim.Np, sim.Ns, sim.Ns),dtype=complex)
    Ec_out = np.zeros(len(tdat))
    Eq_out = np.zeros(len(tdat))
    q, p = functions.init_classical(sim)
    t_ind = 0
    H = functions.H_e_SCF(psi_db_mat, sim) + functions.H_qc_SCF(q, p, sim)
    fq, fp = functions.get_F_SCF(psi_db_mat, sim)
    for t_bath_ind in range(len(tdat_bath)):
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * sim.dt_bath or t_bath_ind == len(tdat_bath) - 1:
            rhoDb_out[t_ind] = np.einsum('ji,li->ijl',np.conjugate(psi_db_mat),psi_db_mat)
            Eq_out[t_ind] = np.real(functions.op_MF(H, psi_db_mat, sim))
            Ec_out[t_ind] = np.sum((1 / 2) * ((sim.w ** 2) * q ** 2 + p ** 2))
            if np.abs(Eq_out[t_ind]+Ec_out[t_ind] - (Eq_out[0] + Ec_out[0])) > np.abs(0.01*(Eq_out[0]+Ec_out[0])):
                print('ERROR: energy conservation ')
            t_ind += 1
    p, q = functions.RK4(p, q, (fq, fp), sim)
    H = functions.H_e_SCF(psi_db_mat, sim) + functions.H_qc_SCF(q, p, sim)
    for i1 in range(2 * sim.Np):
        psi_db_mat[:, i1] = functions.timestepRK_Q(H, psi_db_mat[:, i1], sim.dt_bath)
    fq, fp = functions.get_F_SCF(psi_db_mat, sim)
    msg = 'Finished trial #' + str(index)
    return rhoDb_out, Eq_out, Ec_out, msg, index, seed

@ray.remote
def run_traj_exact(sim,index, seed):
    np.random.seed(seed)
    tdat = np.arange(0, sim.tmax+sim.dt,sim.dt)
    tdat_bath = np.arange(0, sim.tmax+sim.dt_bath, sim.dt_bath)
    rhoDb_out = np.zeros((len(tdat), len(sim.full_basis),len(sim.full_basis)),dtype=complex)
    Ec_out = np.zeros(len(tdat))
    Eq_out = np.zeros(len(tdat))
    q,p=functions.init_classical(sim)
    psi_db = scipy.sparse.coo_array((np.array([1 + 0.0j]), (np.array([0]), np.array([0]))),
                                    shape=(len(sim.full_basis), 1))
    t_ind = 0
    V = functions.V_mf_X(psi_db, sim)
    H_e = functions.H_k_sparse(sim)
    H = H_e + functions.H_qc_X(q, p, sim)
    fq, fp = functions.get_F(V, sim)
    for t_bath_ind in range(len(tdat_bath)):
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5*sim.dt_bath or t_bath_ind == len(tdat_bath)-1:
            rhoDb_out[t_ind] = np.outer(np.conjugate(psi_db.toarray().flatten()),psi_db.toarray().flatten())
            Eq_out[t_ind] = np.real(np.conjugate(psi_db).transpose().dot(H.dot(psi_db))).toarray()[0,0]
            Ec_out[t_ind] = np.sum((1/2)*((sim.w**2)*q**2 + p**2))
            if np.abs(Eq_out[t_ind]+Ec_out[t_ind] - (Eq_out[0] + Ec_out[0])) > np.abs(0.01*(Eq_out[0]+Ec_out[0])):
                print('ERROR: energy conservation ')
            t_ind += 1
        p, q = functions.RK4(p, q, (fq, fp), sim)
        H = H_e + functions.H_qc_X(q, p, sim)
        psi_db = functions.RK4_Q_sparse(H,psi_db,sim.dt_bath)#functions.timestepRK_Q(H, psi_db, sim.dt_bath)
        V = functions.V_mf_X(psi_db, sim)
        fq, fp = functions.get_F(V, sim)
        msg = 'Finished trial #' + str(index)
    return rhoDb_out, Eq_out, Ec_out, msg, index, seed

def run_dynamics(sim):
    print('Starting Calculation')
    r_ind = 0
    if sim.nprocs > sim.trials:
        sim.nprocs = sim.trials
    if path.exists(sim.filename + '_seed.csv'):
        last_seed = np.loadtxt(sim.filename + '_seed.csv', delimiter=",").astype(int)
        if len(np.shape(last_seed)) == 1:
            last_index = last_seed[0] + 1
        else:
            last_index = last_seed[-1, 0] + 1
    else:
        last_index = 0
    np.random.seed(1234)
    seeds = np.array([np.random.randint(100000) for n in range(sim.trials + last_index)])
    if sim.e_method == 'exact':
        for run in range(0, int(sim.trials / sim.nprocs)):
            results = [
                run_traj_exact.remote(sim, run * sim.nprocs + i + last_index, seeds[run * sim.nprocs + i + last_index])
                for
                i in range(sim.nprocs)]
            for r in results:
                rhoDb, Eq, Ec, msg, index, seed = ray.get(r)
                print(msg)
                if run == 0 and r_ind == 0:
                    rhoDbdat = np.zeros_like(rhoDb)
                    Eqdat = np.zeros_like(Eq)
                    Ecdat = np.zeros_like(Ec)
                    seed_list = np.array([[index, seed]]).astype(int)
                rhoDbdat += rhoDb
                Eqdat += Eq
                Ecdat += Ec
                if r_ind != 0 or run != 0:
                    seed_list = np.vstack((seed_list, np.array([[index, seed]])))
                r_ind += 1
        if path.exists(sim.filename + '_rhoDb.npy'):
            rhoDbdat += np.load(sim.filename + '_rhoDb.npy')
        if path.exists(sim.filename + '_Eq.csv'):
            Eqdat += np.loadtxt(sim.filename + '_Eq.csv', delimiter=",")
        if path.exists(sim.filename + '_Ec.csv'):
            Ecdat += np.loadtxt(sim.filename + '_Ec.csv', delimiter=",")
        if path.exists(sim.filename + '_seed.csv'):
            seed_list = np.vstack((np.loadtxt(sim.filename + '_seed.csv', delimiter=","), seed_list))

        np.save(sim.filename + '_rhoDb.npy', rhoDbdat)
        np.savetxt(sim.filename + '_Eq.csv', Eq, delimiter=",")
        np.savetxt(sim.filename + '_Ec.csv', Ec, delimiter=",")
        np.savetxt(sim.filename + '_seed.csv', seed_list, delimiter=",")
        sim.rhoDb = rhoDbdat
        sim.Eq = Eqdat
        sim.Ec = Ecdat
    if sim.e_method == 'scf':
        for run in range(0, int(sim.trials / sim.nprocs)):
            results = [run_traj_scf.remote(sim, run * sim.nprocs + i + last_index, seeds[run * sim.nprocs + i + last_index]) for
                       i in range(sim.nprocs)]
            for r in results:
                rhoDb, Eq, Ec, msg, index, seed = ray.get(r)
                print(msg)
                if run == 0 and r_ind == 0:
                    rhoDbdat = np.zeros_like(rhoDb)
                    Eqdat = np.zeros_like(Eq)
                    Ecdat = np.zeros_like(Ec)
                    seed_list = np.array([[index, seed]]).astype(int)
                rhoDbdat += rhoDb
                Eqdat += Eq
                Ecdat += Ec
                if r_ind != 0 or run != 0:
                    seed_list = np.vstack((seed_list, np.array([[index, seed]])))
                r_ind += 1
        if path.exists(sim.filename + '_rhoDb.npy'):
            rhoDbdat += np.load(sim.filename + '_rhoDb.npy')
        if path.exists(sim.filename + '_Eq.csv'):
            Eqdat += np.loadtxt(sim.filename + '_Eq.csv', delimiter=",")
        if path.exists(sim.filename + '_Ec.csv'):
            Ecdat += np.loadtxt(sim.filename + '_Ec.csv', delimiter=",")
        if path.exists(sim.filename + '_seed.csv'):
            seed_list = np.vstack((np.loadtxt(sim.filename + '_seed.csv', delimiter=","), seed_list))

        np.save(sim.filename + '_rhoDb.npy', rhoDbdat)
        np.savetxt(sim.filename + '_Eq.csv', Eq, delimiter=",")
        np.savetxt(sim.filename + '_Ec.csv', Ec, delimiter=",")
        np.savetxt(sim.filename + '_seed.csv', seed_list, delimiter=",")
        sim.rhoDb = rhoDbdat
        sim.Eq = Eqdat
        sim.Ec = Ecdat
    print('Finished')
    return sim