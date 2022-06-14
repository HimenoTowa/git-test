"""Backend supported: tensorflow.compat.v1"""
from gc import callbacks

from numpy import dtype
import deepxde as dde
import numpy as np
from deepxde.backend import tf
import matplotlib.pyplot as plt
import math
dde.config.set_default_float('float64')

N = 10

y = []
x = []
with open('27.csv','r') as f:
    x = f.readline()
    x.strip()
    x = x.split(',')
    x = [float(i) for i in x]
    for line in f.readlines()[0:]:
        line.strip()
        t = line.split(',')
        t = [float(i) for i in t]
        y.append(t)

ob_t =  np.array(x).reshape(-1,1)
ob_y =  np.array(y).reshape(20,-1)
print(ob_t)
# print(ob_y)

def get_K(ob_y,N):
    k = []
    for i in range(N):
        k.append(np.mean(ob_y[i]))
    return k

K = []
K = get_K(ob_y,N*2)
print('K',K)




ob_y = ob_y.T
obs = [0] * (N*2)
for i in range(N*2):
    obs[i] =  dde.PointSetBC(ob_t, ob_y[:,i:i+1], component=i)


#固定的系数：
Nid = N;      Nimax = N;
Nvd = N;      Nvmax = N;
mDi = 1;       mDv = 1;
Ndef = Nimax+Nvmax;
MM=max(mDi,mDv);
# % Irradiation parameters %
Temp = 181 + 273;
Gnrt = 1.5E-4;
epr = 0.0;      epi = 0.00;     epv = 0.00;
fi2 = 0.46;     fi3 = 0.46;     fi4 = 0.08;
fv2 = 0.55;     fv3 = 0.27;     fv4 = 0.18;
# % Materials parameters %
Vat = 2.30E-23;     a0 = 2.86E-8;     Burg = 2.04E-8;
Matom = 56.0;        DenMat = 7.80;    
Grain = 4.00E3;      DenDis = 1.0E8;
Zi = 1.05;      Di0 = 4.00E-4;      EFi = 4.30;     EMi = 0.30;     EBi = 0.80;
Zv = 1.00;      Dv0 = 1.00E-0;      EFv = 1.60;     EMv = 1.30;     EBv = 0.20;
riv = 6.5E-8;     gamma = 6.25e+14; 
# % Constant parameters %
Kb = 8.625e-5;  acons = 6.02E23;
# % F-P parameters %
Cti = 1.01;     
Ctv = 1.01;    

at2cn = DenMat * acons / Matom

Gdef=np.zeros((Ndef,1));
Gdef[0] = Gnrt*(1 - epr)*(1 - epi)*at2cn;
Gdef[1] = Gnrt*(1 - epr)*fi2*epi*at2cn / 2;
Gdef[2] = Gnrt*(1 - epr)*fi3*epi*at2cn / 3;
Gdef[3] = Gnrt*(1 - epr)*fi4*epi*at2cn / 4;
Gdef[Nimax + 0] = Gnrt*(1 - epr)*(1 - epv)*at2cn;
Gdef[Nimax + 1] = Gnrt*(1 - epr)*fv2*epv*at2cn / 2;
Gdef[Nimax + 2] = Gnrt*(1 - epr)*fv3*epv*at2cn / 3;
Gdef[Nimax + 3] = Gnrt*(1 - epr)*fv4*epv*at2cn / 4;

#5秒时 N=10
y0 = [
1.39389e+14,
1.4029e+14,
1.28843e+14,
1.21139e+14,
1.15404e+14,
1.10873e+14,
1.07154e+14,
1.04014e+14,
1.01301e+14,
1.52847e-284,
5.8507e+17,
6.76615e+13,
4.37151e+09,
289314.0,
19.397,
0.00131078,
8.9032e-08,
6.06814e-12,
4.14521e-16,
1.68968e-100,
]
# y0 = np.array(ori_y0,dtype='float64')
# y0.astype(np.float64)
# K = [1e15, 1e15, 1e15, 1e15, 1e15, 1e15, 1e15, 1e15, 1e15, 1e-284,
#     1e17, 1e13,  1e9,  1e5,   1e1, 1e-3, 1e-7, 1e-11, 1e-15, 1e-100]



# for i in range(N*2):
#     y0[i] = y0[i]/K[i]

Ceq_i=np.zeros((Nimax,1));           Ceq_v=np.zeros((Nvmax,1)) ;
# I = [0]*Nimax;             V = [0]*Nvmax;
EFiloop=np.zeros((Nimax,1));         EFvloop=np.zeros((Nvmax,1));
EBiloop = np.zeros((Nimax,1));       EBvloop = np.zeros((Nvmax,1));
EBvoid = np.zeros((Nvmax,1));
uki = np.zeros((Nimax,1));    ukv = np.zeros((Nvmax,1));
xki = np.zeros((Nvmax,1));    xkv = np.zeros((Nvmax,1));

Di = np.zeros((mDi,1));             Dv = np.zeros((mDv,1));
Riloop = np.zeros((Nimax,1));       Rvloop = np.zeros((Nvmax,1));   
Rvoid = np.zeros((Nvmax,1));
Bias_iloop_i = np.zeros((Nimax,1)); Bias_iloop_v=np.zeros((Nimax,1));
Bias_vloop_i = np.zeros((Nvmax,1)); Bias_vloop_v=np.zeros((Nvmax,1));
Bias_void_i = np.zeros((Nvmax,1));  Bias_void_v=np.zeros((Nvmax,1));
AB_iloop_i = np.zeros((Nimax,mDi));  AB_iloop_v = np.zeros((Nimax,mDv));
AB_vloop_v = np.zeros((Nvmax,mDv));  AB_vloop_i = np.zeros((Nvmax,mDi));
AB_void_v = np.zeros((Nvmax,mDv));   AB_void_i = np.zeros((Nvmax,mDi));
EM_iloop_i = np.zeros((Nimax,1));    EM_vloop_v = np.zeros((Nvmax,1)); 
EM_void_v = np.zeros((Nvmax,1));


for i in range(0,mDi):
    Di[i] = Di0*np.exp(-EMi/Kb/Temp)/(i+1);
for i in range(0,mDv):
    Dv[i] = Dv0*np.exp(-EMv/Kb/Temp)/(i+1);

for i in range(0,Nid):
    xki[i] = i+1
for i in range(0,Nvd):
    xkv[i] = i+1

for i in range(Nid,Nimax):
    uki[i] = (i-Nid)*np.log(Cti)
for i in range(Nvd,Nvmax):
    ukv[i] = (i-Nvd)*np.log(Ctv)

for i in range(Nid,Nimax):
    xki[i] = Nid + (1 - np.exp(uki[i]))/(1-Cti)
for i in range(Nvd,Nvmax):
    xkv[i] = Nvd + (1 - np.exp(ukv[i]))/(1-Ctv)

for i in range(0,Nimax):
    Riloop[i] = np.sqrt(xki[i]*Vat/Burg/np.pi)
for i in range(0,Nvmax):
    Rvloop[i] = np.sqrt(xkv[i]*Vat/Burg/np.pi)
for i in range(0,Nvmax):
    Rvoid[i] = np.power(3*xkv[0]*Vat/(4*np.pi), 1/3)

EBiloop[1] = EBi;
EBvloop[1] = EBv;

for i in range(1,Nimax):
    EBiloop[i] = EFi + (EBiloop[1]-EFi)/(np.power(2, 2/3)-1)*(np.power(xki[i],2/3)-np.power(xki[i]-1, 2/3));
for i in range(1,Nvmax):
    EBvloop[i] = EFv + (EBvloop[1]-EFv)/(np.power(2, 2/3)-1)*(np.power(xkv[i],2/3)-np.power(xkv[i]-1, 2/3));

for i in range(0,Nimax):
    Bias_iloop_i[i] = Zi + (np.sqrt(Burg/8/np.pi/a0)*42-Zi)/np.power(xki[i],0.35);
    Bias_iloop_v[i] = Zv + (np.sqrt(Burg/8/np.pi/a0)*35-Zv)/np.power(xki[i],0.35);
for i in range(0,Nvmax):
    Bias_vloop_i[i] = Zi + (np.sqrt(Burg/8/np.pi/a0)*42-Zi)/np.power(xkv[i],0.35);
    Bias_vloop_v[i] = Zv + (np.sqrt(Burg/8/np.pi/a0)*35-Zv)/np.power(xkv[i],0.35);
    Bias_void_i[i] = np.power((48*np.pi*np.pi/Vat/Vat),1.0/3.0)*np.power(xkv[i],1.0/3.0)*Vat;
    Bias_void_v[i] = np.power((48*np.pi*np.pi/Vat/Vat),1.0/3.0)*np.power(xkv[i],1.0/3.0)*Vat;

for i in range(0,Nimax):
    for j in range(0,mDi):
        AB_iloop_i[i][j] = 2 * np.pi * Bias_iloop_i[i] * Riloop[i] * Di[j]
    for k in range(0,mDv):
        AB_iloop_v[i][k] = 2 * np.pi * Bias_iloop_v[i] * Riloop[i] * Dv[k]

for i in range(0,Nvmax):
    for j in range(0,mDi):
        AB_vloop_i[i][j] = 2*np.pi*Bias_vloop_i[i]*Rvloop[i]*Di[j]
        AB_void_i[i][j] = Bias_void_i[i] * Di[j]
    for k in range(0,mDv):
        AB_vloop_v[i][k] = 2*np.pi*Bias_vloop_v[i]*Rvloop[i]*Dv[k]
        AB_void_v[i][k] = Bias_void_v[i]*Dv[k]

for i in range(1,Nimax):
    EM_iloop_i[i] = AB_iloop_i[i-1][0] * np.exp(-EBiloop[i]/(Kb*Temp))/Vat
for i in range(1,Nvmax):
    EM_vloop_v[i] = AB_vloop_v[i-1][0] * np.exp(-EBvloop[i]/(Kb*Temp))/Vat
    EM_void_v[i] = AB_void_v[i][0] * np.exp(-EBvoid[i]/(Kb*Temp))/Vat

Rcom = 4*np.pi*riv*(Dv[0]+Di[0])

EFiloop[0] = EFi
EFvloop[0] = EFv
for i in range(1,Nimax):
    EFiloop[i] = (EFi - EBi)/(np.power(2, 2/3) - 1)*np.power(i+1, 2/3)
for i in range(1,Nvmax):
    EFvloop[i] = (EFv - EBv)/(np.power(2, 2/3) - 1)*np.power(i+1, 2/3);


def ode(x, y):  #x、y都是tensorflow类型，而IC/BC是numpy类型  
    """
    dC1i/dt = ...
    dC2i/dt = ...
    ...
    dC50i/dt = ...
    ...
    dC1v/dt = ...
    dC2v/dt = ...
    ...
    dC50v/dt = ...
    """
    #RT式子
    #左端项
    dcdt = [0] * Ndef
    for i in range(Ndef):
        dcdt[i] = dde.grad.jacobian(y, x, i=i)
    #右端项
    rhs = [0]*Ndef   

    # #gPINN
    # #左端项
    # dcdt_gPINN = [0] * Ndef
    # for i in range(Ndef):
    #     dcdt_gPINN[i] = dde.grad.hessian(y, x, component=i, i=0, j=0);
    # #右端项
    # rhs_gPINN = [0] * Ndef

    #因变量y1~yN  
    I = [0]*Nimax
    V = [0]*Nvmax
    for i in range(0,Nimax):
        I[i] = y[:, i:i+1]
    for i in range(0,Nvmax):
        V[i] = y[:, Nimax+i:Nimax+i+1]

    ab_ii = [([0]*MM) for i in range(Nimax)] # MM:列   Nimax:行
    ab_iv = [([0]*MM) for i in range(Nimax)]
    em_ii = [([0]*1) for i in range(Nimax)]
    ab_vv = [([0]*MM) for i in range(Nvmax)]
    ab_vi = [([0]*MM) for i in range(Nvmax)]
    em_vv = [([0]*1) for i in range(Nvmax)]

    for i in range(2*MM, Nid-MM):
        sum_i = 0
        for j in range(0, MM): #原版matlab代码以1开始，+j会有用，但这里以0开始，需要+1起到+j的作用（MM=1)
            sum_i = sum_i + AB_iloop_i[i-(j+1),j]*I[i-(j+1)]*I[j] + AB_iloop_v[i+(j+1),j]*I[i+(j+1)]*V[j] \
                            - AB_iloop_i[i,j]*I[i]*I[j]     - AB_iloop_v[i,j]*I[i]*V[j];
        rhs[i] = Gdef[i] + sum_i + EM_iloop_i[i+1]*I[i+1] - EM_iloop_i[i]*I[i]
        ab_ii[i][0] = AB_iloop_i[i][0]*I[i]
        ab_iv[i][0] = AB_iloop_v[i][0]*I[i]
        em_ii[i] = EM_iloop_i[i]*I[i]
        
    for i in range(2*MM, Nvd-MM):
        sum_v = 0
        for j in range(0, MM):
            sum_v = sum_v + AB_vloop_v[i-(j+1),j]*V[i-(j+1)]*V[j] + AB_vloop_i[i+(j+1),j]*V[i+(j+1)]*I[j] \
                            - AB_vloop_v[i,j]*V[i]*V[j]     - AB_vloop_i[i,j]*V[i]*I[j]
        rhs[Nimax+i] = Gdef[Nimax+i] + sum_v + EM_vloop_v[i+1]*V[i+1] - EM_vloop_v[i]*V[i]
        ab_vv[i][0] = AB_vloop_v[i,0]*V[i]
        ab_vi[i][0] = AB_vloop_i[i,0]*V[i]
        em_vv[i] = EM_vloop_v[i]*V[i]
        

    sum_defi = 0
    for e in ab_ii:
        sum_defi += e[0]
    for e in ab_vi:
        sum_defi += e[0] 
    sum_disi = Zi*DenDis;
    sum_gbi = 6 * tf.sqrt(sum_disi + sum_defi/Di[0])/Grain; 
    sum_toti = sum_disi*Di[0] + sum_gbi*Di[0] + sum_defi


    sum_defv = 0
    for e in ab_vv:
        sum_defv += e[0]
    for e in ab_iv:
        sum_defv += e[0]
    sum_disv = Zv*DenDis
    sum_gbv = 6 * tf.sqrt(sum_disv + sum_defv/Dv[0])/Grain
    sum_totv = sum_disv*Dv[0] + sum_gbv*Dv[0] + sum_defv;

    sum_em_ii = 0
    for e in em_ii:
        sum_em_ii += e[0]
    sum_em_vv = 0
    for e in em_vv:
        sum_em_vv += e[0]

    for i in range(0,Nimax):
        Ceq_i[i] = np.exp(-EFiloop[i] / (Kb*Temp))*at2cn
    for i in range(0,Nvmax):
        Ceq_v[i] = np.exp(-EFvloop[i] / (Kb*Temp))*at2cn

    rhs[0] = Gdef[0] - Rcom*I[0]*V[0] + AB_iloop_v[1][0]*I[1]*V[0] \
        - 4*AB_iloop_i[0][0]*I[0]*I[0] \
        - sum_toti*I[0] + sum_toti*Ceq_i[0] + sum_em_ii + 3*EM_iloop_i[1]*I[1]

    rhs[Nimax] = Gdef[Nimax] - Rcom*I[0]*V[0] + AB_vloop_i[1][0]*V[1]*I[0] \
        - 4*AB_vloop_v[0][0]*V[0]*V[0] \
        - sum_totv*V[0] + sum_totv*Ceq_v[0] + sum_em_vv + 3*EM_vloop_v[1]*V[1]

    rhs[1] = Gdef[1] + 2*AB_iloop_i[0][0]*I[0]*I[0] - 2*EM_iloop_i[1]*I[1] \
        - AB_iloop_i[1][0]*I[1]*I[0] + EM_iloop_i[2]*I[2] \
        - AB_iloop_v[1][0]*I[1]*V[0] + AB_iloop_v[2][0]*I[2]*V[0]

    rhs[Nimax+1] = Gdef[Nimax+1] + 2*AB_vloop_v[0][0]*V[0]*V[0] \
        - 2*EM_vloop_v[1]*V[1] - AB_vloop_v[1][0]*V[1]*V[0] \
        + EM_vloop_v[2]*V[2] - AB_vloop_i[1][0]*V[1]*I[0] \
        + AB_vloop_i[2][0]*V[2]*I[0]

    


    res= [0]*Ndef
    for i in range(Ndef):
        res[i] = dcdt[i]-rhs[i]
         


    return res




def boundary(_, on_initial):
    return on_initial


geom = dde.geometry.TimeDomain(5.0, 600.0)

ics = [0] * Ndef
for i in range(Ndef):
    ics[i] = dde.IC(geom, lambda X: y0[i], boundary, component = i)




# data = dde.data.PDE(geom, ode, ics, num_domain=100, num_boundary=40, num_test=100, anchors=fix_points)
# data = dde.data.PDE(geom, ode, obs, num_domain=100, num_boundary=20, num_test=100, anchors=ob_t)
data = dde.data.PDE(geom, ode, [], num_domain=100, num_boundary=20, num_test=100)


layer_size = [1] + [80] * 4 + [N*2]
activation = "swish"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

def feature_transform(t):
    t = t * 0.01
    return tf.concat(
        (
            t
            # tf.exp(-t)
        ),
        axis=1,
    )
net.apply_feature_transform(feature_transform)

# Y0 = np.array(y0)/np.array(K)

def output_transform(inputs, outputs):
    # t = inputs[:,:1]
    # out = [0] * (N*2)
    # for i in range(N*2):
    #     out[i] = (t-0) * outputs[:,i:i+1] + Y0[i]
    #     out[i] = out[i] * K[i]
    # return tf.concat(out,axis=1)
    return (
            y0 + tf.math.tanh(inputs) * tf.constant(K,dtype='float64') * outputs
        )


net.apply_output_transform(output_transform)


model = dde.Model(data, net)

def get_initial_loss(model):
    model.compile("adam", lr=0.001)
    losshistory, train_state = model.train(0)
    return losshistory.loss_train[0]

initial_losses = get_initial_loss(model)
loss_weights = [5/i if i!=0 else 1 for i in initial_losses] 


# 把后面Ndef个观测点的权重*100
# print('loss_weight',loss_weights)

# for i in range(Ndef,Ndef * 2):
#     loss_weights = loss_weights[i] * 100


model.compile("adam", lr=0.0002, loss_weights=loss_weights)

losshistory, train_state = model.train(epochs=1000000, display_every=1000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
model.save('./RT_Adam')
model.compile("L-BFGS",loss_weights=loss_weights)

# losshistory, train_state = model.train()
# dde.saveplot(losshistory, train_state, issave=True, isplot=True)
# model.save('./RT_L-BFGS')





