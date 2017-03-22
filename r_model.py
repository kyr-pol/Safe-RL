import numpy as np
import GPy
import pylab as pl
import sys
import data
import pdb
np.seterr(invalid='raise')
D = 4 #state space dimensions
F = 5 #state space + input dimensions

class GP_model(object):

    def __init__(self, read=False, eps=30,seed=0):
        self.read = read
        self.eps = eps
        self.testN = 2000
        self.steps = 200
        self.subs = 20 #subsampling freq
        self.seed = seed
        np.random.seed(self.seed)
        self.get_data(seed=self.seed)
        self.run()


    def get_data(self,seed=1):
            if self.read:
                all_u = np.genfromtxt('all_u.csv',delimiter=',')
                tr_X2= np.genfromtxt('tr_X.csv',delimiter=',')
                te_X2 = np.genfromtxt('te_X.csv',delimiter=',')
                tr_Y = np.genfromtxt('tr_Y.csv',delimiter=',')
                te_Y = np.genfromtxt('te_Y.csv',delimiter=',')
            else:
                all_u, tr_X2, te_X2, tr_Y, te_Y = data.create_data(
                                                    self.eps * self.steps,
                                                    self.testN,self.steps,
                                                    self.subs,seed=self.seed)
                self.seed += 1 #otherwise next time we'd get the same data

            #reformat data a bit
            tr_X = np.zeros((len(tr_X2),F))
            tr_X[:,0:D] = tr_X2[:,0:D]
            tr_X[:,D] = tr_X2[:,F]
            te_X = np.zeros((len(te_X2),F))
            te_X[:,0:D] = te_X2[:,0:D]
            te_X[:,D] = te_X2[:,F]
            tr_Y = tr_Y.reshape(len(tr_X),D)
            tr_l = len(tr_X)
            te_l = len(te_X)
            tr_U = all_u[0:self.eps]
            te_U = all_u[self.eps:self.eps+10]
            self.tr_X, self.tr_Y, self.te_X, self.te_Y,self.tr_U, self.te_U, self.all_u = \
            tr_X, tr_Y, te_X, te_Y, tr_U, te_U, all_u

    def get_more_data(self,extra_eps,seed=2):
        self.eps += extra_eps

        all_u, tr_X2, te_X2, tr_Y, te_Y = data.create_data(
                                            extra_eps * self.steps,
                                            self.testN,self.steps,
                                            self.subs,seed=self.seed)
        self.seed += 1
        #reformatting
        tr_X = np.zeros((len(tr_X2),F))
        tr_X[:,0:D] = tr_X2[:,0:D]
        tr_X[:,D] = tr_X2[:,F]
        tr_Y = tr_Y.reshape(len(tr_X),D)
        tr_U = all_u[0:extra_eps]
        self.tr_X = np.concatenate((self.tr_X, tr_X), axis=0)
        self.tr_Y = np.concatenate((self.tr_Y, tr_Y), axis=0)
        self.tr_U = np.concatenate((self.tr_U, tr_U), axis=0)

    #calculates the GP kernel matrix
    def K_rbf(self,l,sv,nv,data):
        N = len(data)
        K = np.zeros((N,N))
        R = np.reshape(data, (N,1,F)) - np.reshape(data, (1,N,F))
        l2 = np.diag(l **(-2))
        K = sv *                      \
            np.exp( -0.5 * np.sum(
                R.reshape((N,N,F,1)) * l2.reshape((1,1,F,F)) * R.reshape((N,N,1,F)),
                axis=(2,3)))          \
            + np.diag([float(nv)]*N)
        return K

    #returns the means predicted with uncertain inputs and q vectors used later
    def mean_pred(self, K1, pt, s_in, m1, vi, beta):
        tr_l = len(self.tr_X)
        #qi
        lamda = np.diag(m1.rbf.lengthscale ** 2 )
        ilamda = np.diag(m1.rbf.lengthscale ** (-2) )

        sl = s_in + lamda
        sl = np.linalg.inv(sl)

        tmp = m1.rbf.variance  / \
        np.power(np.linalg.det(np.dot(s_in,ilamda)+np.eye(F)),0.5)
        ex = np.zeros((tr_l,1))
        q = np.zeros((tr_l,1))
        for i in range(0,tr_l):
            r = np.reshape(vi[i],(F,1))
            q[i] =  tmp * np.exp(-0.5 * np.dot(np.dot(np.transpose(r),sl),r))
        prediction = np.dot(beta.transpose(),q)
        return prediction,q

    #performs predictions, retruns the Dx and x_t+1 mean and variance
    def predict_mv(self,models, Ks, invKs, lens, state_m, s_in, theta):
        tr_l = len(self.tr_X)
        te_l = len(self.te_X)
        vi = np.zeros(np.shape(self.tr_X))
        for i in range(0,len(vi)):
            vi[i] = self.tr_X[i,:] - state_m

        #all betas
        betas = [np.dot(invKs[i], self.tr_Y[:,i]) for i in range(0,D)]
        MUSQ = [self.mean_pred(Ks[i], state_m, s_in, models[i],vi,betas[i]) for i in range(0,D)]
        mus = [x[0] for x in MUSQ]
        qs = [x[1] for x in MUSQ]
        #variance prediction
        #all Rs
        R = np.zeros((D,D,F,F))
        for i in range(0,D):
            for j in range(0,D):
                R[i,j,:,:] = np.dot(s_in,lens[i] + lens[j]) + np.eye(F)

        #all zetas
        zetas =  np.zeros((D,tr_l,F))
        for i in range(0,D):
                for k in range(0,tr_l):
                    zetas[i,k] = np.dot(lens[i],vi[k])

        Q = np.zeros((D,D,tr_l,tr_l))
        EfDaDb = np.zeros((D,D))

        kas = np.zeros((D,tr_l))
        for i in range(0,D):
            for j in range(0,tr_l):
                kas[i,j] = np.log(models[i].rbf.variance) -    \
                0.5 * np.dot(np.dot(vi[j],lens[i]),vi[j])

        hlp = np.zeros(tr_l) + 1
        kas = [ np.outer(x,hlp) for x in kas]
        zeta_a = np.zeros((tr_l,tr_l,F))
        y = np.zeros((tr_l,tr_l,F)) + 1

        for i in range(0,D):
            for j in range(0,i+1):
                R_c = R[i,j,:,:]
                R_i = np.linalg.inv(R_c)

                zeta_a = np.reshape(zetas[i],(tr_l, 1, F)) + np.reshape(zetas[j],
                         (1, tr_l, F))

                Q[i,j] = kas[i] + np.transpose(kas[j]) +                          \
                         0.5 * np.sum( np.reshape(zeta_a,(tr_l, tr_l, F,1)) *     \
                                    np.reshape(np.dot(R_i,s_in),(1,1,F,F))  *     \
                                    np.reshape(zeta_a,(tr_l, tr_l, 1,F)),axis=(2,3))

                Q[i,j] = np.linalg.det(R_c)**(-0.5) * np.exp(Q[i,j])
                EfDaDb[i,j] = np.dot(np.dot(betas[i],Q[i,j]), betas[j])
                EfDaDb[j,i] = EfDaDb[i,j]

        s_ab = np.zeros((D,D))
        for i in range(0,D):
            for j in range(0,D):
                if i != j: s_ab[i,j] = EfDaDb[i,j] - mus[i] * mus[j]

        Exv = np.zeros(D)
        for i in range(0,D):
            Exv[i] = models[i].rbf.variance -                       \
                     np.trace(np.dot(invKs[i],Q[i,i]))

        s_aa = np.zeros(D)
        for i in range(0,D):
            s_aa[i] = Exv[i] + EfDaDb[i,i] - mus[i]**2

        s_out = np.zeros((D,D))
        s_out = s_ab + np.diag(s_aa)

        TMP = np.zeros((D,tr_l,F))
        for i in range(0,D):
            for n in range(tr_l):
                TMP[i,n] = betas[i][n] * float(qs[i][n]) *                     \
                           np.dot(np.dot(s_in, np.linalg.inv(s_in +            \
                           np.linalg.inv(lens[i]))), vi[n])

        cov = np.zeros((D,F))
        for i in range(D): cov[i] = sum(TMP[i])
        s_m = np.zeros(F)
        s_m[0:D] = state_m[0:D] + np.reshape(mus,D)
        s_m[D] = np.dot(theta,state_m[0:D]) * 0.1**4

        s_v = np.zeros((F,F))
        s_v[0:D,0:D] = s_in[0:D,0:D] + s_out + cov[0:D,0:D] +                  \
                       np.transpose(cov[0:D,0:D])
        #this is only valid because of the linear controller used
        for i in range(D):
          s_v[D,i] = 0.1**8  * theta[i]**2 * s_out[i,i]
          s_v[i,D] = s_v[D,i]
        s_v[D,D] = sum(s_v[D,0:D])
        return s_m, s_v, mus, s_out


    def run(self):
        #create the 4 models
        tr_X, tr_Y, te_X, te_Y, tr_U, te_U, all_u =                            \
        self.tr_X, self.tr_Y, self.te_X,self.te_Y,self.tr_U,self.te_U,self.all_u
        tr_l = len(tr_X)
        te_l = len(te_X)
        k = GPy.kern.RBF(F,ARD=True)
        l = GPy.likelihoods.Gaussian()

        models =[]
        for i in range(0,D):
            models += [GPy.core.GP(tr_X,np.reshape(self.tr_Y[:,i],(tr_l,1)),
                                   kernel=k.copy(), likelihood = l.copy())]
            models[i].optimize('BFGS',max_iters=20)

        #one step predictions
        pr_Y = np.asarray([x.predict(te_X) for x in models])
        pr_Y_m = np.reshape(np.asarray([x[0] for x in pr_Y]),(D,te_l))
        pr_Y_v = np.reshape(np.asarray([x[1] for x in pr_Y]),(D,te_l))
        res = te_Y - pr_Y_m.transpose()

        #calculate the 4 kernels
        Ks = [self.K_rbf(m.rbf.lengthscale, m.rbf.variance, m.Gaussian_noise.variance,
                tr_X) for m in models]
        invKs = []
        for k in Ks:
            L1 = np.linalg.cholesky(k)
            L2 = L1.T.conj()
            L2i = np.linalg.inv(L2)
            L1i = np.linalg.inv(L1)
            invKs += [np.dot(L2i,L1i)]

        #all Lambda^(-1)
        lens = [x.rbf.lengthscale for x in models]
        lens = np.asarray(lens) ** (-2)
        lens = [np.diag(x) for x in lens ]

        #Dx mean and v
        mus = np.zeros((len(te_Y),D))
        S = np.zeros((len(te_Y),D,D))
        #state and input related variables
        state_m = np.zeros((len(te_Y),F))
        state_v = np.zeros((len(te_Y),F,F))
        theta =np.zeros(D)
        #for every test episode i take steps j
        for i in range(0,10):
            state_m[9*i] = te_X[9 * i]
            state_v[9*i] = np.zeros((F,F))
            #simple gp prediction, no input distribution Dt1
            mus[9*i+1,:] = pr_Y_m[:,9*i]
            S[9*i+1] = np.diag(pr_Y_v[:,9*i])
            #theta for the whole episode
            theta[:] = te_U[i]
            #state mean variance and input for time 1
            state_m[9*i+1, 0:D] = state_m[9*i, 0:D] + mus[9*i+1]
            state_m[9*i+1,D] = np.dot(state_m[9*i,0:D],theta) * 0.1**4
            state_v[9*i+1,0:D,0:D] = S[9*i+1]
            #this is only valid because of the liner controller
            state_v[9*i+1,D,D] = 2 * 0.1**8 * np.dot(theta**2,np.diag(S[9*i+1]))
            for k in range(2,10):
                state_m[9*i+k], state_v[9*i+k], mus[9*i+k], S[9*i+k] =         \
                self.predict_mv(models, Ks, invKs, lens,
                                state_m[9*i+k-1], state_v[9*i+k-1], theta)

        #Accesible domains for others
        self.state_m = state_m
        self.state_v = state_v
        self.mus = mus
        self.S = S
        self.models = models
        self.step_pred_m = pr_Y_m
        self.step_pred_v = pr_Y_v
