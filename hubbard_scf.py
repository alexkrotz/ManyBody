import functions
import numpy as np
from tqdm import tqdm
def init_hubbard_scf(sim): # initialize exact electronic structure
    def spin(ind):  # returns spin state of ind
        if ind < sim.Ns / 2:
            return 1
        if ind >= sim.Ns / 2:
            return 0

    def site(ind):  # returns orbital state of ind
        if ind < sim.Ns / 2:
            return int(ind)
        if ind >= sim.Ns / 2:
            return int(ind - sim.Ns / 2)

    spin_vec = np.vectorize(spin)
    site_vec = np.vectorize(site)

    def umklapp(ind):
        centers = np.arange(-2 * sim.Np, 2 * sim.Np) * sim.Ns0
        return ind - centers[np.argmin(np.abs(ind[np.newaxis] - centers[:, np.newaxis]), axis=0)]

    def umklapp_single(ind):
        centers = np.arange(-2 * sim.Np, 2 * sim.Np) * sim.Ns0
        return ind - centers[np.argmin(np.abs(ind - centers), axis=0)]

    N1_basis = functions.S(1, sim.Ns)  # generate the one particle basis only for one spin
    t = 1
    # energy truncation
    e_list_1p = np.zeros((len(N1_basis)))
    for i in range(len(N1_basis)):
        e_list_1p[i] = np.sum(-2 * t * np.cos(2 * np.pi * site_vec(np.arange(sim.Ns)[(N1_basis[i] == 1)]) / sim.Ns0))
    e_id_1p = e_list_1p.argsort()
    spin_sort_1p = e_id_1p.argsort()
    e_list_1p = e_list_1p[e_id_1p]
    N1_basis = np.copy(N1_basis[e_id_1p])

    print('Basis generation finished.')
    print('Generating indices')

    # calculate matrix elements of V_{q} = \sum_{k}c^{\dagger}_{k+q}c_{k} in the one electron basis
    V_mat = np.zeros((sim.Ns0, sim.Ns, sim.Ns))
    for i in range(sim.Ns):
        for j in range(sim.Ns):
            for q_n in range(sim.Ns0):
                vec_i = N1_basis[i]
                vec_j = N1_basis[j]
                ind_i = np.where(vec_i == 1)[0]
                ind_j = np.where(vec_j == 1)[0]
                site_i = umklapp_single(site(ind_i))
                site_j = umklapp_single(site(ind_j))
                spin_i = spin(ind_i)
                spin_j = spin(ind_j)
                q = umklapp_single(site(q_n))
                if spin_i == spin_j:
                    if umklapp_single(site_i - site_j) == q:
                        V_mat[q_n, i, j] = 1
    sim.N1_basis = N1_basis
    sim.spin = spin
    sim.site = site
    sim.spin_vec = spin_vec
    sim.site_vec = site_vec
    sim.umklapp = umklapp
    sim.umklapp_single = umklapp_single
    sim.V_mat = V_mat

    return sim