import numpy as np
import matplotlib.pyplot as plt
import random as r
import pandas as pd


class rw:
    @staticmethod
    def read_w(address):# receive a string,return a dict
        df = pd.read_csv(address)
        return dict(zip(df['name'], df['a_bweights']))
    @staticmethod
    def read_data(address,kind):# receive 2 string ,return a list
        df = pd.read_csv(address)
        return list(df[kind])
    @staticmethod
    def write_w(file_name,w):# receive string,list
        abstr = [f'a{i}' for i in range(8)] + ['b']+['c']
        dict1 = {'name': abstr, 'a_bweights': w}
        df = pd.DataFrame(dict1)
        df.to_csv(file_name+'.csv')
        print('we_ok')
    @staticmethod
    def write_loss(loss,name):#receive bool/num ,string
        if loss==True:
            with open(name+'.csv','w')as f:
                f.write(',loss')
                f.close()
            print('set_ok')
        else:
            with open(name + '.csv', 'a') as f:
                f.write(f'\n,{loss}')
                f.close()
            print('ls_ok')

class run:
    @staticmethod
    def y(weights,x):#receive nparray ,return nparray
        a=x+weights[:-2]
        return np.sum(np.sin(weights[:-2]*np.array([a for i in range(len(weights[:-2]))])))+weights[-1]
    @staticmethod
    def loss(y,yr):#receive nparray ,return nparray
        return 0.5*((y-yr)**2)
    @staticmethod
    def dloss_dy(y,yr):#receive nparray ,return nparray
        return y-yr
    @staticmethod
    def dy_dx(weights,x):#receive nparray ,return nparray
        a = x + weights[:-2]
        dsin=np.sin(weights[:-2] * np.array([a for i in range(len(weights[:-2]))]))
        dcos=np.cos(weights[:-2] * np.array([a for i in range(len(weights[:-2]))]))
        d=[]
        for i in range(len(weights[:-2])):
            d.append(sum(dsin[0:i])+dcos[i]+sum(dsin[i:len(weights[:-2])]))
        return np.array(d[0])
    @staticmethod
    def dy_dc(weights, x):
        a = x + weights[:-2]
        return np.sum(np.sin(weights[:-2] * np.array([a for i in range(len(weights[:-2]))]))) + 1
    @staticmethod
    def dx_dw(weights, x):  # receive nparray,nparray ,return nparray
        return x
    @staticmethod
    def da_db(a):  # receive nparray,nparray ,return nparray
        return a+1

    @staticmethod
    def t_town(weights,d_weights,a):  # receive nparray,nparray,num ,return nparray
        return weights-a*d_weights



#prepare data and weights
w=np.array(list(rw.read_w('weights2.csv').values()))
x=np.array(rw.read_data('train2.csv','Close')[1:53])
a=0.00001
bath_size=2
train_times=20
loss_bath=[]
loss_g=[]
loss_v=[]
d=[]
#一阶差分
x1=x#np.diff(x)
#forward
for i in range(train_times):
    for j in range(1,len(x1)):
        y=run.y(w,x1[j])
        loss_bath.append(run.loss(x1[j],x1[j-1]))
        d.append(run.dloss_dy(x1[j],x1[j-1]))
        if j%bath_size==0:
            l=sum(loss_bath) / len(loss_bath)
            loss_g.append(l)
            loss_bath=[]

            dls_dy=sum(d) / len(d)
            d=[]
            dy_dc=run.dy_dc(w,x[j])
            dy_dx=run.dy_dx(w,x1[j])
            dx_dw=run.dx_dw(w,x1[j])
            dx_da=run.da_db(x1[j])

            dw=dls_dy*dy_dx*dx_dw
            dc=dls_dy*dy_dc
            db=sum(dls_dy*dy_dx*dx_da)/len(dls_dy*dy_dx*dx_da)

            d_weights=np.array(list(dw)+[db]+[dc])
            w=run.t_town(w,d_weights,a)

    print(f'第{i+1}轮,loss={sum(loss_g)/len(loss_g)}')
    loss_v.append(sum(loss_g)/len(loss_g))
#print(w)
plt.plot([i for i in range(len(loss_v))],loss_v,c='red')
plt.xlabel('times')
plt.ylabel('loss')
plt.show()


ax=x1[1]-x1[0]
yx=[ax]
for i in range(0,len(x1[1:])):
    yx.append(run.y(w,yx[i]))
yx.pop(0)
yx=np.array(yx)
yx=(yx - yx.min(axis=0))/(yx.max(axis=0)-yx.min(axis=0))
xr=(x1[1:] - x1[1:].min(axis=0))/(x1[1:].max(axis=0)-x1[1:].min(axis=0))
print(w)
rw.write_w('weights2',list(w))

plt.plot([i for i in range(len(x1[1:]))],xr,c='red')
plt.plot([i for i in range(len(x1[1:]))],yx,c='blue')
plt.xlabel('times')
plt.ylabel('value')
plt.show()

'''
loss=1.8142300000000173
[-0.07556085  3.85041692 -1.85848955 -2.78986369  1.67096115  3.90846428
 -1.29802067  2.34843269 -0.99387582 -1.86819104]
'''