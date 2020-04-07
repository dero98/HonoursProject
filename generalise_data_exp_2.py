import random
import numpy as np
data_x_pos=np.zeros((100,2))
data_x_neg=np.zeros((100,2))
index=0
for i in range(-4,5,2):
    t=0
    while(t<20):
        data_x_pos[index][0]=round(random.uniform(i*math.pi-(0.5*math.pi),i*math.pi+(0.5*math.pi)),2)
        if(t in [1,5,7,18]):
            data_x_neg[index][0]=round(random.uniform(i*math.pi-(0.5*math.pi),i*math.pi+(0.5*math.pi)),2)
        else:
            data_x_neg[index][0]=round(random.uniform((i+1)*math.pi-(0.5*math.pi),(i+1)*math.pi+(0.5*math.pi)),2)
        t=t+1
        index=index+1
index=0
for i in range(-5,5,2):
    t=0
    while(t<20):
        data_x_pos[index][1]=round(random.uniform(i*math.pi,i*math.pi+math.pi),2)
        if(t in [2,6,9,19]):
            data_x_neg[index][1]=round(random.uniform((i+1)*math.pi,(i+1)*math.pi+math.pi),2)
        else:
            data_x_neg[index][1]=round(random.uniform((i+1)*math.pi,(i+1)*math.pi+math.pi),2)
        t=t+1
        index=index+1
f= open("./data/exp_2_pos.db","w+")
for row in data_x_pos:
    for feat in row:
        f.write(str(feat)+" ")
    f.write("\n")
f.close()
f= open("./data/exp_2_neg.db","w+")
for row in data_x_neg:
    for feat in row:
        f.write(str(feat)+" ")
    f.write("\n")
f.close()
