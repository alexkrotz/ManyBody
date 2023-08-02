import functions
import numpy as np
from tqdm import tqdm
def init_hubbard_exact(sim): # initialize exact electronic structure
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
    full_basis = functions.S(2 * sim.Np, sim.Ns)
    print('Size of full basis: ', len(full_basis))

    S_ind = (0 + ((np.sum(full_basis[:, sim.Ns0:], axis=1) - np.sum(full_basis[:, :sim.Ns0], axis=1)) / 2)).astype(
        int)  # spin polarization
    S_ind -= np.min(S_ind)
    K_ind = np.zeros((len(full_basis)), dtype=int)
    for i in tqdm(range(len(S_ind))):
        v = umklapp(site_vec(np.arange(sim.Ns)))[full_basis[i] == 1]
        K_ind[i] = np.arange(sim.Ns0)[umklapp_single((np.sum(v))).astype(int)]
    SK_ind = np.array([S_ind, K_ind], dtype=int).transpose()
    unique_SK, inverse_SK = np.unique(SK_ind, axis=0, return_inverse=True)
    pos_ind = np.arange(len(unique_SK))[inverse_SK]

    full_ind = np.array([S_ind, K_ind, pos_ind], dtype=int).transpose()
    idx = np.argsort(pos_ind)
    full_basis = np.copy(full_basis[idx])
    full_ind = np.copy(full_ind[idx])
    S_ind = np.copy(S_ind[idx])
    max_ind = full_ind[-1, -1]

    # energy truncation
    e_list = np.zeros((len(full_basis)))
    for i in range(len(full_basis)):
        e_list[i] = np.sum(-2 * t * np.cos(2 * np.pi * site_vec(np.arange(sim.Ns)[(full_basis[i] == 1)]) / sim.Ns0))
    e_id = e_list.argsort()
    e_list = e_list[e_id]
    full_basis = np.copy(full_basis[e_id])
    full_ind = np.copy(full_ind[e_id])
    S_ind = np.copy(S_ind[e_id])
    unique_index, counts = np.unique(full_ind, axis=0, return_counts=True)
    #print('Spin index, Momentum index, Block index, Block Dimension')
    #print(np.hstack((unique_index, np.array([counts]).transpose())))

    print('Basis generation finished.')
    print('Generating indices')
    for i in tqdm(range(len(full_basis))):
        for j in range(len(full_basis)):
            vec_i = full_basis[i]
            vec_j = full_basis[j]
            if np.sum(vec_i * vec_j) >= 2 * sim.Np - 1:  # check if states differ by at most one spinorbital
                diff = np.where(vec_i != vec_j)[0]
                if len(diff) == 0:  # no difference, i = j <i_{N}| \sum_{k} c^{\dagger}_{k+q, s} c_{k, s}|j_{N}> = (q=0, s_i = s_j = 0
                    if i == j and i == 0:
                        q = 0
                        m_q = 0
                        indices = np.array([i, j, q, m_q, 0, 0, sim.Np])
                        indices = np.vstack((indices, np.array([i, j, q, m_q, 1, 1, sim.Np])))

                        n_indices = np.concatenate((np.array([i, j]), vec_i * vec_j))
                    else:
                        q = 0
                        m_q = 0
                        indices = np.vstack((indices, np.array([i, j, q, m_q, 0, 0, sim.Np])))
                        indices = np.vstack((indices, np.array([i, j, q, m_q, 1, 1, sim.Np])))
                        n_indices = np.vstack((n_indices, np.concatenate((np.array([i, j]), vec_i * vec_j))))

                elif len(diff) == 2:  # difference by only one spinorbital
                    spin_diff = np.unique(spin_vec(diff))
                    if len(spin_diff) == 1:  # The spin of the states in which they differ must be the same
                        site_diff = site_vec(diff)
                        if spin_diff == 1:  # they differ by a spin up state
                            i_ind = diff[np.where(vec_i[diff] == 1)[0]]  # spinorbital index of the different electron
                            j_ind = diff[np.where(vec_j[diff] == 1)[0]]
                            i_site = site(i_ind)  # site index of the different electron
                            j_site = site(j_ind)
                            i_spin = spin(i_ind)  # spin index of the different electron
                            j_spin = spin(j_ind)
                            n_j = vec_j[j_ind]
                            c1vec_j = np.copy(vec_j)
                            c1vec_j[j_ind] -= 1  # destroy electron at position j_ind
                            c1vec_j = n_j * c1vec_j  # * ((-1)**qN(vec_j, j_ind[0]))  # multiply by exponent (determined from vec_j) and n_j
                            fac = n_j * ((-1) ** functions.qN(vec_j, j_ind[0]))
                            n_i = c1vec_j[i_ind]
                            c2c1vec_j = np.copy(c1vec_j)
                            c2c1vec_j[i_ind] += 1
                            c2c1vec_j = (1 - n_i) * c2c1vec_j  # (-1)**qN(c1vec_j, i_ind[0]) *
                            fac *= (1 - n_i) * (-1) ** functions.qN(c1vec_j, i_ind[0])
                            q = umklapp_single(
                                i_site - j_site)  # get the momentum difference between the initial and final state wrapped to the first BZ
                            m_q = umklapp_single(j_site - i_site)
                            indices = np.vstack((indices, np.array([i, j, q, m_q, i_spin, j_spin, fac[0]])))
                        if spin_diff == 0:
                            i_ind = diff[np.where(vec_i[diff] == 1)[0]]  # spinorbital index of the different electron
                            j_ind = diff[np.where(vec_j[diff] == 1)[0]]
                            i_site = site(i_ind)  # site index of the different electron
                            j_site = site(j_ind)
                            i_spin = spin(i_ind)  # spin index of the different electron
                            j_spin = spin(j_ind)
                            n_j = vec_j[j_ind]
                            c1vec_j = np.copy(vec_j)
                            c1vec_j[j_ind] -= 1  # destroy electron at position j_ind
                            # print(i_ind, j_ind,vec_i, vec_j, c1vec_j)
                            c1vec_j = n_j * c1vec_j  # * ((-1)**qN(vec_j, j_ind[0]))  # multiply by exponent (determined from vec_j) and n_j
                            fac = n_j * ((-1) ** functions.qN(vec_j, j_ind[0]))
                            n_i = c1vec_j[i_ind]
                            c2c1vec_j = np.copy(c1vec_j)
                            c2c1vec_j[i_ind] += 1
                            c2c1vec_j = (1 - n_i) * c2c1vec_j  # (-1)**qN(c1vec_j, i_ind[0]) *
                            fac *= (1 - n_i) * (-1) ** functions.qN(c1vec_j, i_ind[0])
                            q = umklapp_single(
                                i_site - j_site)  # get the momentum difference between the initial and final state wrapped to the first BZ
                            m_q = umklapp_single(j_site - i_site)
                            indices = np.vstack((indices, np.array([i, j, q, m_q, i_spin, j_spin, fac[0]])))
                else:
                    print('error', diff)

    sim.N1_basis = N1_basis
    sim.full_basis = full_basis
    sim.indices = indices
    sim.n_indices = n_indices
    sim.full_ind = full_ind
    sim.spin = spin
    sim.site = site
    sim.spin_vec = spin_vec
    sim.site_vec = site_vec
    sim.umklapp = umklapp
    sim.umklapp_single = umklapp_single

    return sim