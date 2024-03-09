import numpy as np
from plotlib import *
from kalmanFilter import *
from helpers import *
import csv

class IMU(object):
    def __init__(self, sampling):
        self.sampling = sampling
        self.accel    = np.zeros(3)
        self.gyro     = np.zeros(3)
        self.magne    = np.zeros(3)
        self.dt       = 1 / self.sampling

        self.x = np.zeros(9)
        self.P = np.eye(9) * 5000
        self.R = np.eye(3) * 1000000

        self.bias = np.array([0.04, 0.02, 0.02])
  
    # Get variance and bias from data
    def initVariance(self, data, noiseCoeff={'w': 100, 'a': 100, 'm': 10}):
        
        # Discard the first few readings due to fluctuation
        a = data[:, 0:3] # change here for data
        w = data[:, 3:6]
        m = data[:, 6:9]

        # ---- Gravity ----
        gn = -a.mean(axis=0)
        gn = gn[:, np.newaxis]
        # Save the initial magnitude of gravity
        g0 = np.linalg.norm(gn)

        # ---- Magnetic field ----
        mn = m.mean(axis=0)
        mn = normalized(mn)[:, np.newaxis]

        # ---- Noise covariance ---- 
        aVar = a.var(axis=0)
        wVar = w.var(axis=0)
        mVar = m.var(axis=0)
        print('acc var: %s, norm: %s' % (aVar, np.linalg.norm(aVar)))
        print('ang var: %s, norm: %s' % (wVar, np.linalg.norm(wVar)))
        print('mag var: %s, norm: %s' % (mVar, np.linalg.norm(mVar)))

        # ---- Sensor noise ----
        gyroNoise  = noiseCoeff['w'] * np.linalg.norm(wVar)
        gyroBias   = w.mean(axis=0)

        accelNoise = noiseCoeff['a'] * np.linalg.norm(aVar)
        accelBias  = a.mean(axis=0)

        magneNoise = noiseCoeff['m'] * np.linalg.norm(mVar)
        magneBias  = m.mean(axis=0)

        return (gn, g0, mn, gyroNoise, gyroBias, accelNoise, accelBias, magneNoise, magneBias)

    # Track attitude of vehicle
    def attitudeTracking(self, data, initVal):
        # ---- Calibrated data goes here! ----
        gn, g0, mn, gyroNoise, gyroBias, accelNoise, accelBias, magneNoise, magneBias = initVal
        a = data[:, 0:3] - gyroBias
        w = data[:, 3:6] 
        m = data[:, 6:9] 
        sampleSize = np.shape(data)[0]

        # ---- Data container ----
        aNav = []
        oriX = []
        oriY = []
        oriZ = []

        # ---- States and covariance matrix ----
        P = 1e-10 * np.eye(4)
        x = np.array([[1, 0, 0, 0]]).T      # Quaternion state column vector [[q0 q1 q2 q3]]
        initOri = np.eye(3)           # Rotation matrix

        # ---- Extended Kalman filter ----
        t = 0
        while t < sampleSize:
            # ---- Data prep ----
            wt = w[t, np.newaxis].T
            at = a[t, np.newaxis].T
            mt = normalized(m[t, np.newaxis].T)

            # ---- Predict ----
            # Use normalized measurements to reduce error
            F = gyroTransitionMatrix(wt[0][0], wt[1][0], wt[2][0], self.dt)
            W = jacobianW(x)
            Q = (gyroNoise * self.dt) ** 2 * W @ W.T
            x = normalized(F @ x)
            P = F @ P @ F.T + Q

            # ---- Update ----
            # ---- Acceleration and magnetic field prediction ---- 
            q  = x.T[0]
            pa = normalized(-quat2RotMatrix(q[0], q[1], q[2], q[3]) @ gn)
            pm = normalized(quat2RotMatrix(q[0], q[1], q[2], q[3]) @ mn)

            # Residual    
            y = np.vstack((normalized(at), mt)) - np.vstack((pa, pm))

            # Sensor noise matrix
            Ra = [(accelNoise/ np.linalg.norm(at))**2 + (1 - g0 / np.linalg.norm(at))**2] * 3
            Rm = [magneNoise**2] * 3
            R  = np.diag(Ra + Rm)

            # Kalman gain
            Ht = H(x, gn, mn)
            S  = Ht @ P @ Ht.T + R
            K  = P @ Ht.T @ np.linalg.inv(S)
            x = x + K @ y
            P = P - K @ Ht @ P

            # ---- Post update ----
            x = normalized(x)
            P = 0.5 * (P + P.T)         # Symmetric

            # ---- Navigation frame acceleration ----
            conj = -np.eye(4)
            conj[0][0] = 1
            an = rotate(conj @ x) @ at + gn
            orin = rotate(conj @ x) @ initOri

            # ---- Save data ----
            aNav.append(an.T[0])
            oriX.append(orin.T[0, :])
            oriY.append(orin.T[1, :])
            oriZ.append(orin.T[2, :])

            t += 1
        
        aNav = np.array(aNav)
        oriX = np.array(oriX)
        oriY = np.array(oriY)
        oriZ = np.array(oriZ)

        return (aNav, oriX, oriY, oriZ)

    def attitudeObtain(self, data):
        #  ---- Data preparation ----
        sampleSize = np.shape(data)[0]
        initOri = np.eye(3)
        euler = data[:, 0:3]
        euler = np.deg2rad(euler)
        a     = data[:, 3:6]

        aVar = a.mean(axis=0)
        accelNoise = np.linalg.norm(aVar) * 100

        # ---- Data container ----
        aNav = []
        oriX = []
        oriY = []
        oriZ = []
        
        t = 0
        while t < sampleSize:
            eulert = euler[t, np.newaxis].T
            at     = a[t, np.newaxis].T - self.bias[:, np.newaxis].T
            q = euler2Quat(eulert[1][0], eulert[2][0], eulert[0][0]) 
            q = normalized(q)

            # ---- Navigation frame acceleration ----
            conj = -np.eye(4)
            conj[0][0] = 1
            an = rotate(conj @ q) @ at 
            orin = rotate(conj @ q) @ initOri

            # ---- Save data ----
            aNav.append(an.T[0])
            oriX.append(orin.T[0, :])
            oriY.append(orin.T[1, :])
            oriZ.append(orin.T[2, :])
            t += 1
        
        aNav = np.array(aNav)
        oriX = np.array(oriX)
        oriY = np.array(oriY)
        oriZ = np.array(oriZ)
        return (aNav, oriX, oriY, oriZ, accelNoise)

    def posTrackWithConstantSpeed(self, aNav, data, accelNoise, count):
        #  ---- Data preparation ----
        sampleSize = np.shape(data)[0]
        euler = data[:, 0]
        euler = np.deg2rad(euler)
        # aVar = aNav.mean(axis=0)
        # accelNoise = np.linalg.norm(aVar) * 100

        # ---- Init kalman filter matrix ----
        # Transition matrix
        Ft = np.array([[1., self.dt, 0.5 * self.dt**2],
                       [0.,      1.,       self.dt   ],
                       [0.,      0.,       1.        ]])
        F = np.eye(9)
        F[0:3, 0:3] = Ft
        F[3:6, 3:6] = Ft
        F[6:9, 6:9] = Ft

        n = F.shape[1]

        Qt = accelNoise * np.array([[self.dt**5 / 20, self.dt**4 / 8, self.dt**3/ 6],
                                    [self.dt**4 / 8, self.dt**3 / 9, self.dt**2 / 2],
                                    [self.dt**3 / 6, self.dt**2 / 2, self.dt]])
        
        # print(accelNoise)
        Q = np.eye(9)
        Q[0:3, 0:3] = Qt
        Q[3:6, 3:6] = Qt
        Q[6:9, 6:9] = Qt

        H = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0],])
        
        # ---- Data container ----
        pos = []

        t = 0
        while t < sampleSize: 
            # ---- Predict step ----
            self.x[2] = aNav[t][0]
            self.x[5] = aNav[t][1]
            self.x[8] = aNav[t][2]

            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q

            # ---- Update step ----
            # Đưa speed xe về navigation frame
            # speed = yawRotationMatrix(euler[t][0]) @ speed  
            z = [0., 0., 1.] # Cho chạy x, y trong vòng 2.5 giây đầu 
            if count >= 565 / 2:
                z = [0., 0., -1.] # dừng lại 2.5 giây cuối 
            S = H @ self.P @ H.T + self.R
            Sinv = np.linalg.inv(S)
            K = self.P @ H.T @ Sinv

            y = z - H @ self.x
            self.x = self.x + K @ y

            I = np.eye(n)
            self.P = (I - K @ H) @ self.P
            self.P = 0.5 * (self.P + self.P.T)    

            pos.append([self.x[0], self.x[3], self.x[6]])
            t += 1
        pos = np.array(pos)
        print(self.P)
        fields = ['x', 'y', 'z']
        filename = "pos.csv"
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fields)
            writer.writerows(pos)
        return pos
        
    def removeAccErr(self, aNav, threshold=0.2, filter=False, wn=(0.01, 15)):
        sampleSize = np.shape(aNav)[0]
        tStart = 0
        for t in range(sampleSize):
            at = aNav[t]
            if np.linalg.norm(at) > threshold:
                tStart = t
                break
        
        tEnd = 0
        for t in range(sampleSize - 1, -1, -1):
            at = aNav[t]
            if np.linalg.norm(at - aNav[-1]) > threshold:
                tEnd = t
                break

        anDrift = aNav[tEnd:].mean(axis=0)
        anDriftRate = anDrift / (tEnd - tStart)

        for i in range(tEnd - tStart):
            aNav[tStart + i] -= (i + 1) * anDriftRate

        for i in range(sampleSize - tEnd):
            aNav[tEnd + i] -= anDrift

        if filter:
            filteredAccNav = filtSignal([aNav], dt=self.dt, wn=wn, btype='bandpass')[0]
            return filteredAccNav
        else: 
            return aNav
    
    # Applies zero velocity update (ZUPT) algorithm to accel data
    def zupt(self, aNav, threshold):
        sampleSize = np.shape(aNav)[0]
        vel = []
        prevt = -1
        stillPhase = False

        v = np.zeros((3, 1))
        t = 0
        while t < sampleSize:
            at = aNav[t, np.newaxis].T

            if np.linalg.norm(at) < threshold:
                if not stillPhase:
                    predictVel = v + at * self.dt
                    
                    velDriftRate = predictVel / (t - prevt)
                    for i in range(t - prevt - 1):
                        vel[prevt + 1 + i] -= (i + 1) * velDriftRate.T[0]

                v = np.zeros((3, 1))
                prevt = t
                stillPhase = True
            else:
                v = v + at * self.dt
                stillPhase = False

            vel.append(v.T[0])
            t += 1
        vel = np.asarray(vel)
        return vel
    
    def positionTracking(self, aNav, vel):
        sampleSize = np.shape(aNav)[0]
        pos = []
        p = np.array([[0., 0., 0.]]).T

        t = 0
        while t < sampleSize:
            at = aNav[t, np.newaxis].T
            vt = vel[t, np.newaxis].T 

            p = p + vt * self.dt + 0.5 * at + self.dt**2
            pos.append(p.T[0])
            t += 1

        pos = np.array(pos)
        fields = ['x', 'y', 'z']
        filename = "pos.csv"
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fields)
            writer.writerows(pos)
        return pos

def plot_trajectory():
    data = []
    with open("data_new.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            # rowt = np.array(row[1:10])
            data.append(row)
    data = np.array(data, dtype=np.float64)
    plot3([data[:, 3:6]])

    tracker = IMU(sampling=100)
    t = 0
    p_p = []
    a_nav_p = []
    while t < len(data):
        new_data = data[t:t+10]
        # print('initializing...')
        # init_list = tracker.initVariance(data[5:30])

        # print('--------')
        # print('processing...')
        
        # EKF step
        # a_nav, orix, oriy, oriz = tracker.attitudeTracking(data[30:], init_list)
        a_nav, orix, oriy, oriz, accelNoise = tracker.attitudeObtain(new_data)
        a_nav_p.append(a_nav)

        # Acceleration correction step
        
        # a_nav_filtered = tracker.removeAccErr(a_nav, filter=False)
        # plot3([a_nav, a_nav_filtered])
        

        # ZUPT step
        # v = tracker.zupt(a_nav_filtered, threshold=0.2)
        #plot3([v])
    
        # Integration Step
        # p = tracker.positionTracking(a_nav_filtered, v)
        p = tracker.posTrackWithConstantSpeed(a_nav, new_data, accelNoise, t)
        p_p.append(p)
        t = t + 10
        # plot3D([[p, 'position']])

    pp = np.vstack([row for row in p_p])
    aa = np.vstack([row for row in a_nav_p])
    print(len(pp))
    plot3([aa])
    plot3([pp])
    plot3D([[pp, 'position']])

if __name__ == "__main__":
    plot_trajectory()
    