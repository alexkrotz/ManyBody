import numpy as np
from sympy.utilities.iterables import multiset_permutations
import scipy
import scipy.integrate as it

def qN(N1, k): # fermionic antisymmetry exponent
    return np.sum(N1[:k])

def S(Np, Ns):  # generate occupation numbers for basis
    S0 = np.zeros(Ns, dtype=int)
    S0[:Np] = 1
    return np.array(list(multiset_permutations(S0)))

def init_classical(sim):
    p_out = np.random.normal(0, np.sqrt(sim.kB*sim.temp), sim.Ns0)
    q_out = np.random.normal(0, np.sqrt(sim.kB*sim.temp)/sim.w, sim.Ns0)
    return q_out, p_out


def H_qc_X(q, p, sim):  # does not generate the full matrix but instead generates a list of the nonzero matrix elements
    mels = np.zeros((len(sim.indices)), dtype=complex)
    row = np.zeros((len(sim.indices)))
    col = np.zeros((len(sim.indices)))
    for i in range(len(sim.indices)):
        k = sim.indices[i, 2]
        m_k = sim.indices[i, 3]
        fac = sim.indices[i, 6]
        row[i] = sim.indices[i, 0]
        col[i] = sim.indices[i, 1]
        # if row[i] == col[i]:
        #    fac = 1
        mels[i] = sim.g * np.sqrt(sim.w) / np.sqrt(2 * sim.Ns0) * (sim.w * (q[m_k] + q[k]) - 1.0j * (p[m_k] - p[k])) * fac

    sparse_mat = scipy.sparse.coo_array((mels, (row, col)), shape=(len(sim.full_basis), len(sim.full_basis)))
    return sparse_mat
def V_mf_X(psi_sparse, sim):
    psi = psi_sparse.toarray().flatten()
    V_out = np.zeros((sim.Ns0,2,2),dtype=complex)
    for i in range(len(sim.indices)):
        q = sim.indices[i,2]
        V_out[q, sim.indices[i,4], sim.indices[i,5]] += np.conjugate(psi[sim.indices[i,0]])*psi[sim.indices[i,1]]*sim.indices[i,6]
    return V_out

def H_k_MF(V, sim):
    #print(V[0])
    #print(V[1])
    mels = np.zeros((len(sim.indices)), dtype=complex)
    row = np.zeros((len(sim.indices)))
    col = np.zeros((len(sim.indices)))
    for i in range(len(sim.indices)):
        q = sim.indices[i,2]
        m_q = sim.indices[i,3]
        i_spin = sim.indices[i,4]
        j_spin = sim.indices[i,5]
        row[i] = sim.indices[i,0]
        col[i] = sim.indices[i,1]
        fac = sim.indices[i,6]
        if row[i] == col[i]:
            fac = 1
        mels[i] = (sim.u/sim.Ns0)*fac*V[m_q,i_spin,j_spin]
        if row[i] == col[i]:
            vec = np.arange(sim.Ns0)[sim.full_basis[int(row[i])][i_spin*sim.Ns0:(i_spin+1)*sim.Ns0]==1]
            mels[i] += np.sum(-2*sim.t*np.cos(2*np.pi*vec/sim.Ns0))
    sparse_mat = scipy.sparse.coo_array((mels, (row, col)), shape=(len(sim.full_basis), len(sim.full_basis)))
    return sparse_mat

def get_F(V, sim):
    Fq = np.zeros((sim.Ns0))
    Fp = np.zeros((sim.Ns0))
    Fq = sim.g*np.sqrt(2*(sim.w**3)/sim.Ns0)*np.real(V[:,0,0] + V[:,1,1])
    Fp = -sim.g*np.sqrt(2*sim.w/sim.Ns0)*np.imag(V[:,0,0] + V[:,1,1])
    return Fq, Fp

#@jit(nopython=True)
def RK4(p_bath, q_bath, QF, sim):
    Fq, Fp = QF
    K1 = sim.dt_bath *(p_bath + Fp)
    L1 = -sim.dt_bath * (sim.w**2 * q_bath + Fq)    # [wn2] is w_alpha ^ 2
    K2 = sim.dt_bath * ((p_bath + 0.5 * L1) + Fp)
    L2 = -sim.dt_bath * (sim.w**2 * (q_bath + 0.5 * K1) + Fq)
    K3 = sim.dt_bath * ((p_bath + 0.5 * L2) + Fp)
    L3 = -sim.dt_bath * (sim.w**2 * (q_bath + 0.5 * K2) + Fq)
    K4 = sim.dt_bath * ((p_bath + L3) + Fp)
    L4 = -sim.dt_bath * (sim.w**2 * (q_bath + K3) + Fq)
    q_bath = q_bath + 0.166667 * (K1 + 2 * K2 + 2 * K3 + K4)
    p_bath = p_bath + 0.166667 * (L1 + 2 * L2 + 2 * L3 + L4)
    return p_bath, q_bath

def timestepRK_Q(mat, cgrid, dt):
    def f_qho(t, c):
        return -1.0j * np.matmul(mat, c)
    soln4 = it.solve_ivp(f_qho, (0, dt), cgrid, method='RK45', max_step=dt, t_eval=[dt], rtol=1e-10, atol=1e-10)
    return soln4.y.flatten()

def RK4_Q_sparse(H, wavefn, dt):
    K1 = (-1j * H.dot(wavefn))
    K2 = (-1j * H.dot(wavefn + 0.5 * dt * K1))
    K3 = (-1j * H.dot(wavefn + 0.5 * dt * K2))
    K4 = (-1j * H.dot(wavefn + dt * K3))
    wavefn = wavefn + dt * 0.166667 * (K1 + 2 * K2 + 2 * K3 + K4)
    return wavefn
def n_k_X(psi,sim):
    n_k_out = np.zeros((sim.Ns))
    for i in range(len(sim.n_indices)):
        n_k_out += np.real(np.conjugate(psi[sim.n_indices[i,0]])*psi[sim.n_indices[i,1]]*sim.n_indices[i,2:])
    return n_k_out
def H_k_X(sim): # generalized hubbard Hamiltonian for any number of electrons in any number of sites, this generates the full Hamiltonian
    mat = np.zeros((len(sim.full_basis),len(sim.full_basis)))
    for i in range(len(sim.full_basis)):
        for j in range(len(sim.full_basis)):
            if sim.full_ind[i,0] == sim.full_ind[j,0] and sim.full_ind[i,1] == sim.full_ind[j,1]: # spin and momentum conservation
                i_vec = sim.full_basis[i]
                j_vec = sim.full_basis[j]
                diff_num = 2*sim.Np - np.sum(i_vec*j_vec)
                # can only differ by two or zero electrons
                if diff_num == 0: # diagonal case
                    k_vals = sim.umklapp(sim.site_vec(np.arange(sim.Ns)))[j_vec==1]
                    mat[i,j] += np.sum(-2*sim.t*np.cos(2*np.pi*k_vals/sim.Ns0))
                    #print(k_vals,j_vec,mat[i,j])
                    # if the diagonal case has at least one spin 0 and one spin 1 electron then there is a hubbard term
                    if np.sum(i_vec[:sim.Ns0])>0 and np.sum(i_vec[sim.Ns0:])>0:
                        mat[i,j] += sim.u/sim.Ns0

                if diff_num == 2: # differ by two electrons case
                    diff = np.where(i_vec != j_vec)[0]
                    i_diff = diff[i_vec[diff]==1]
                    j_diff = diff[j_vec[diff]==1]
                    if np.sum(j_diff < sim.Ns0)==1:# ensure that the two differences are of two electrons with different spins
                        diff_0 = np.where(i_vec[:sim.Ns0] != j_vec[:sim.Ns0])[0] # identify the spin down different electron indices
                        diff_1 = np.where(i_vec[sim.Ns0:] != j_vec[sim.Ns0:])[0] # identify the spin up different electron indices
                        j_ind_0 = diff_0[j_vec[:sim.Ns0][diff_0]==1][0] # identify the 0 spin electron index in j = k
                        i_ind_0 = diff_0[i_vec[:sim.Ns0][diff_0]==1][0] # identify the 0 spin electron index in i = k+q
                        j_ind_1 = (diff_1[j_vec[sim.Ns0:][diff_1]==1]+sim.Ns0)[0] # identify the 1 spin electron index in j = k'
                        i_ind_1 = (diff_1[i_vec[sim.Ns0:][diff_1]==1]+sim.Ns0)[0] # identify the 1 spin electron index in i = k'-q
                        fac = 1
                        # anihilate the electron at j_ind_0
                        c0j = np.copy(j_vec)
                        fac *= (-1)**(qN(j_vec,j_ind_0))*j_vec[j_ind_0]
                        c0j[j_ind_0] -= 1
                        # anihilate the electron at j_ind_1
                        c1c0j = np.copy(c0j)
                        fac *= (-1)**(qN(c0j,j_ind_1))*c0j[j_ind_1]
                        c1c0j[j_ind_1] -= 1
                        # create the electron at i_ind_1
                        cd1c1c0j = np.copy(c1c0j)
                        fac *= (-1)**(qN(c1c0j,i_ind_1))*(1-c1c0j[i_ind_1])
                        cd1c1c0j[i_ind_1] += 1
                        cd0cd1c1c0j = np.copy(cd1c1c0j)
                        fac *= (-1)**(qN(cd1c1c0j,i_ind_0))*(1-cd1c1c0j[i_ind_0])
                        cd0cd1c1c0j[i_ind_0] += 1
                        if np.sum(cd0cd1c1c0j*i_vec) != 2*sim.Np:
                            print('Error')
                        mat[i,j] += (sim.u/sim.Ns0)*fac
    return mat

def H_k_sparse(sim): # generalized hubbard Hamiltonian for any number of electrons in any number of sites, this generates the full Hamiltonian as a sparse matrix
    #mat = np.zeros((len(full_basis),len(full_basis)))
    mat_indices = np.array([])
    for i in range(len(sim.full_basis)):
        for j in np.arange(i,len(sim.full_basis)):
            if sim.full_ind[i,0] == sim.full_ind[j,0] and sim.full_ind[i,1] == sim.full_ind[j,1]: # spin and momentum conservation
                i_vec = sim.full_basis[i]
                j_vec = sim.full_basis[j]
                diff_num = 2*sim.Np - np.sum(i_vec*j_vec)
                # can only differ by two or zero electrons
                if diff_num == 0: # diagonal case
                    k_vals = sim.umklapp(sim.site_vec(np.arange(sim.Ns)))[j_vec==1]
                    if len(mat_indices)==0:
                        mat_indices = np.array([[i,j,np.sum(-2*sim.t*np.cos(2*np.pi*k_vals/sim.Ns0))]])
                    else:
                        mat_indices = np.vstack((mat_indices, np.array([i,j,np.sum(-2*sim.t*np.cos(2*np.pi*k_vals/sim.Ns0))])))
                    #mat[i,j] += np.sum(-2*t*np.cos(2*np.pi*k_vals/Ns0))
                    #print(k_vals,j_vec,mat[i,j])
                    # if the diagonal case has at least one spin 0 and one spin 1 electron then there is a hubbard term
                    if np.sum(i_vec[:sim.Ns0])>0 and np.sum(i_vec[sim.Ns0:])>0:
                        #mat[i,j] += u/Ns0
                        mat_indices = np.vstack((mat_indices, np.array([i,j,sim.u/sim.Ns0])))

                if diff_num == 2: # differ by two electrons case
                    diff = np.where(i_vec != j_vec)[0]
                    i_diff = diff[i_vec[diff]==1]
                    j_diff = diff[j_vec[diff]==1]
                    if np.sum(j_diff < sim.Ns0)==1:# ensure that the two differences are of two electrons with different spins
                        diff_0 = np.where(i_vec[:sim.Ns0] != j_vec[:sim.Ns0])[0] # identify the spin down different electron indices
                        diff_1 = np.where(i_vec[sim.Ns0:] != j_vec[sim.Ns0:])[0] # identify the spin up different electron indices
                        j_ind_0 = diff_0[j_vec[:sim.Ns0][diff_0]==1][0] # identify the 0 spin electron index in j = k
                        i_ind_0 = diff_0[i_vec[:sim.Ns0][diff_0]==1][0] # identify the 0 spin electron index in i = k+q
                        j_ind_1 = (diff_1[j_vec[sim.Ns0:][diff_1]==1]+sim.Ns0)[0] # identify the 1 spin electron index in j = k'
                        i_ind_1 = (diff_1[i_vec[sim.Ns0:][diff_1]==1]+sim.Ns0)[0] # identify the 1 spin electron index in i = k'-q
                        fac = 1
                        # anihilate the electron at j_ind_0
                        c0j = np.copy(j_vec)
                        fac *= (-1)**(qN(j_vec,j_ind_0))*j_vec[j_ind_0]
                        c0j[j_ind_0] -= 1
                        # anihilate the electron at j_ind_1
                        c1c0j = np.copy(c0j)
                        fac *= (-1)**(qN(c0j,j_ind_1))*c0j[j_ind_1]
                        c1c0j[j_ind_1] -= 1
                        # create the electron at i_ind_1
                        cd1c1c0j = np.copy(c1c0j)
                        fac *= (-1)**(qN(c1c0j,i_ind_1))*(1-c1c0j[i_ind_1])
                        cd1c1c0j[i_ind_1] += 1
                        cd0cd1c1c0j = np.copy(cd1c1c0j)
                        fac *= (-1)**(qN(cd1c1c0j,i_ind_0))*(1-cd1c1c0j[i_ind_0])
                        cd0cd1c1c0j[i_ind_0] += 1
                        if np.sum(cd0cd1c1c0j*i_vec) != 2*sim.Np:
                            print('Error')
                        #mat[i,j] += (u/Ns0)*fac
                        mat_indices = np.vstack((mat_indices, np.array([i,j,sim.u/sim.Ns0])))
                        mat_indices = np.vstack((mat_indices, np.array([j,i,sim.u/sim.Ns0])))
    sparse_mat = scipy.sparse.coo_array((mat_indices[:,2].astype(complex), (mat_indices[:,0], mat_indices[:,1])), shape=(len(sim.full_basis),len(sim.full_basis)))
    return sparse_mat

def wf_k_to_n(wf,sim): # transforms k space 2 electron wavefunction to real-space wavefunction
    wf_out = np.zeros(len(wf),dtype=complex)
    for i in range(len(sim.full_basis)):
        ind_vec = np.where(sim.full_basis[i]==1)[0]
        ind_1 = np.min(ind_vec)# ind_1 < ind_2 <-- kspace indices
        ind_2 = np.max(ind_vec)
        k_1 = sim.site(ind_1)
        k_2 = sim.site(ind_2)
        k_s_1 = sim.spin(ind_1)
        k_s_2 = sim.spin(ind_2)
        #print(ind_1, ind_2, k_1, k_2, s_1, s_2)
        for j in range(len(sim.full_basis)):
            n_ind_vec = np.where(sim.full_basis[j]==1)[0]
            n_ind_1 = np.min(n_ind_vec)
            n_ind_2 = np.max(n_ind_vec)
            n_1 = sim.site(n_ind_1)
            n_2 = sim.site(n_ind_2)
            n_s_1 = sim.spin(n_ind_1)
            n_s_2 = sim.spin(n_ind_2)
            if n_s_1 == k_s_1 and n_s_2 == k_s_2 and n_s_1 == n_s_2:
                wf_out[j] += wf[i]*(1/sim.Ns0)*(np.exp(1.0j*2*np.pi*(k_1*n_1+k_2*n_2)/sim.Ns0) - np.exp(1.0j*2*np.pi*(k_1*n_2+k_2*n_1)/sim.Ns0))
            if n_s_1 == k_s_1 and n_s_2 == k_s_2 and n_s_1 != n_s_2:
                wf_out[j] += wf[i]*(1/sim.Ns0)*(np.exp(1.0j*2*np.pi*(k_1*n_1+k_2*n_2)/sim.Ns0))
    return wf_out

def fill_energy(sim): # generates the zero temperature free particle grounds tate in k space for Np electrons
    psi_db_mat = np.zeros((sim.Ns, 2*sim.Np), dtype=complex)
    for n in range(2*sim.Np):
        psi_db_mat[n,n] = 1.0+0.0j
    return psi_db_mat


def op_MF(op_mat, psi_mat, sim):
    out = 0.0 + 0.0j
    for i in range(sim.Ns):
        for j in range(sim.Ns):
            vec_i = sim.N1_basis[i]
            vec_j = sim.N1_basis[j]
            ind_i = np.where(vec_i == 1)[0]
            ind_j = np.where(vec_j == 1)[0]
            site_i = sim.umklapp_single(sim.site(ind_i))
            site_j = sim.umklapp_single(sim.site(ind_j))
            spin_i = sim.spin(ind_i)
            spin_j = sim.spin(ind_j)
            out += op_mat[i, j] * np.sum(np.conjugate(psi_mat[i, :]) * psi_mat[j, :])

    return out


def V_SCF(psi_mat, sim):
    out = np.zeros((sim.Ns0, 2, 2), dtype=complex)
    for q_n in range(sim.Ns0):
        m_q = sim.umklapp_single(-1 * sim.umklapp_single(q_n))
        q = sim.umklapp_single(q_n)
        for i in range(sim.Ns):
            for j in range(sim.Ns):
                vec_i = sim.N1_basis[i]
                vec_j = sim.N1_basis[j]
                ind_i = np.where(vec_i == 1)[0]
                ind_j = np.where(vec_j == 1)[0]
                site_i = sim.umklapp_single(sim.site(ind_i))
                site_j = sim.umklapp_single(sim.site(ind_j))
                spin_i = sim.spin(ind_i)
                spin_j = sim.spin(ind_j)
                out[q, spin_i, spin_j] += sim.V_mat[q, i, j] * np.sum(np.conjugate(psi_mat[i, :]) * psi_mat[j, :])

    return out

def H_e_SCF(psi_mat, sim):
    V_list = V_SCF(psi_mat, sim)
    out_mat = np.zeros((sim.Ns,sim.Ns),dtype=complex)
    for i in range(sim.Ns):
        for j in range(sim.Ns):
            vec_i = sim.N1_basis[i]
            vec_j = sim.N1_basis[j]
            ind_i = np.where(vec_i == 1)[0]
            ind_j = np.where(vec_j == 1)[0]
            site_i = sim.umklapp_single(sim.site(ind_i))
            site_j = sim.umklapp_single(sim.site(ind_j))
            spin_i = sim.spin(ind_i)
            spin_j = sim.spin(ind_j)
            if spin_i == spin_j:
                q = sim.umklapp_single(site_i - site_j)
                m_q = sim.umklapp_single(-q)
                if i == j:
                    out_mat[i,j] += -2*sim.t*np.cos(2*np.pi*site_i/sim.Ns0) + (sim.u/sim.Ns0)*V_list[m_q,spin_i,spin_j]
                else:
                    out_mat[i,j] += (sim.u/sim.Ns0)*V_list[m_q,spin_i,spin_j]
    return out_mat
def H_qc_SCF(q,p, sim):
    out_mat = np.zeros((sim.Ns,sim.Ns),dtype=complex)
    for i in range(sim.Ns):
        for j in range(sim.Ns):
            vec_i = sim.N1_basis[i]
            vec_j = sim.N1_basis[j]
            ind_i = np.where(vec_i == 1)[0]
            ind_j = np.where(vec_j == 1)[0]
            site_i = sim.umklapp_single(sim.site(ind_i))
            site_j = sim.umklapp_single(sim.site(ind_j))
            spin_i = sim.spin(ind_i)
            spin_j = sim.spin(ind_j)
            if spin_i == spin_j:
                k = sim.umklapp_single(site_i - site_j)
                m_k = sim.umklapp_single(-k)
                out_mat[i,j] = sim.g * np.sqrt(sim.w)/np.sqrt(2*sim.Ns0) * (sim.w * (q[m_k] + q[k]) - 1.0j*(p[m_k] - p[k]))
    return out_mat

def get_F_SCF(psi_mat, sim):
    V_list = V_SCF(psi_mat, sim)
    Fq = np.zeros((sim.Ns0))
    Fp = np.zeros((sim.Ns0))
    Fq = sim.g*np.sqrt(2*(sim.w**3)/sim.Ns0)*np.real(V_list[:,0,0] + V_list[:,1,1])
    Fp = -sim.g*np.sqrt(2*sim.w/sim.Ns0)*np.imag(V_list[:,0,0] + V_list[:,1,1])
    return Fq, Fp