#The dynamical system tha creates the data and whose dynamics are going to be
#modeled is implemented here
import numpy as np
import GPy
import pylab as pl

dt = 0.1
class System(object):
    defx = np.zeros([4,1])
    defx[0][0] = -100
    defx[1][0] = 10
    defx[2][0] = -100
    defx[3][0] = 10
    defu = np.zeros([1,4])

    def __init__(self, x=defx, y=-100, t=0, u=defu,m1 = 1000):
        self.x = x
        self.x[2] = y
        self.t = t
        self.u = u
        self.m1 = 1000
        self.A = np.array([[0, dt, 0, 0],[0, 0, 0, 0],[0, 0, 0, dt],[0,0,0,0]])
        self.B = np.array([[0], [dt/self.m1], [0], [0]])
        self.t = 0
        self.col = False
        self.min_dist = 200

    def step(self):
        x1 = self.x
        input = np.dot(np.outer(self.B, self.u), x1)

        #reasonable acceleration
        if input[1] > 0.2: input[1] = 0.2
        if input[1] < -0.2: input[1] = -0.2

        #noise
        noise = np.transpose(np.random.normal([0]*4,[0.001,0.02,0.001,0.02],(1,4)))

        self.x = x1 +np.dot(self.A,x1) + input + 1.2*noise
        self.t = self.t + dt
        if self.dist()<10: self.col = True
        if self.dist()<self.min_dist: self.min_dist = self.dist()

        return x1,input, self.x

    def done(self):
        if self.x[0] > 10 and self.x[2] > 10:
            return True
        else:
            return False

    def dist(self):
        return  np.sqrt(self.x[0]**2 + self.x[2]**2)

    def run(self):
        while not self.done:
            self.step()
        return self.t, self.min_dist


def create_data(KK,testN, steps, subs, write=False, seed=1):
    np.random.seed(seed)
    pts = KK/subs #training points
    pts2 = testN/subs
    all = np.zeros((KK,steps,3,4))
    all_u= np.random.normal([0]*4, [1.2,12,1.2,12], (200,4))

    for i in range(0,200):
        a = System(u=np.transpose(all_u[i]))
        for j in range(0,steps):
            x1,u,x2 = a.step()
            all[i][j][0] = np.transpose(x1)
            all[i][j][1] = np.transpose(u)
            all[i][j][2] = np.transpose(x2)

    data = np.zeros((KK,8))
    Y = np.zeros((KK,4))
    for i in range(0,KK/steps):
        for j in range(0,steps-subs):
            data[i*(steps-subs)+j][0:4] = all[i][j][0]
            data[i*(steps-subs)+j][4:8] = all[i][j][1]
            Y[i*(steps-subs)+j] = all[i][j+subs][0] - all[i][j][0] #+ \
                                 # np.random.normal([0]*4,[0.005]*4,(1,4))

    test_x = np.zeros((testN,8))
    test_y = np.zeros((testN,4))
    for i in range(KK/steps,(KK+testN)/steps):
        for j in range(0,steps-subs):
            test_x[(i-(KK/steps))*(steps-subs)+j][0:4] = all[i][j][0]
            test_x[(i-(KK/steps))*(steps-subs)+j][4:8] = all[i][j][1]
            test_y[(i-(KK/steps))*(steps-subs)+j] = all[i][j+subs][0] - all[i][j][0]

    data2 = np.zeros((pts,8))
    Y2 = np.zeros((pts,4))
    test_x2 = np.zeros((pts2,8))
    test_y2 = np.zeros((pts2,4))
    for i in range(0,pts):
        data2[i] = data[subs*i]
        Y2[i] = Y[subs*i]
    for i in range(0,pts2):
        test_x2[i] = test_x[subs*i]
        test_y2[i] = test_y[subs*i]

    tr_y1 = Y2
    if write:
        np.savetxt("tr_X.csv", data2, delimiter=",")
        np.savetxt("tr_Y.csv", Y2, delimiter=",")
        np.savetxt("te_X.csv", test_x2, delimiter=",")
        np.savetxt("te_Y.csv", test_y2, delimiter=",")
        np.savetxt("all_u.csv", all_u, delimiter=",")
    return all_u, data2, test_x2, Y2, test_y2
