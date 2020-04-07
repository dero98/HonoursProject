import z3
import smt_solver as ss

smt_sol = z3.Solver()
x1 = z3.Real('x1')
x2 = z3.Real('x2')
x3 = z3.Real('x3')
x4 = z3.Real('x4')
x5 = z3.Real ('x5')
smt_sol.add( x1**3>x3**2,x2>0,x2<4,x1>10,x4>x5+10,x5<-5,x3>x2**3)
data_x_pos = ss.sols_v2(smt_sol,[x1,x2,x3,x4,x5],[],[],0,[],50)


f= open("./data/exp_5_pos.db","w+")
for row in data_x_pos:
    for feat in row:
        f.write(str(feat)+" ")
    f.write("\n")
f.close()

negative_sol=[]
smt_sol_neg=z3.Solver()
smt_sol_neg.add(x1**3>x3**2,x2>=4,x1>10,x4>x5+10,x5<-5,x3>x2**3)
negative_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3,x4,x5],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add(x1**3>x3**2,x2<=0,x1>10,x4>x5+10,x5<-5,x3>x2**3)
negative_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3,x4,x5],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add(x1**3>x3**2,x2>0,x2<4,x1<=10,x4>x5+10,x5<-5,x3>x2**3)
negative_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3,x4,x5],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add(x1**3<=x3**2,x2>0,x2<4,x1<=10,x4>x5+10,x5<-5,x3>x2**3)
negative_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3,x4,x5],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add(x1**3>x3**2,x2>0,x2<4,x1>10,x4>x5+10,x5>-5,x3>x2**3)
negative_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3,x4,x5],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add(x1**3>x3**2,x2>0,x2<4,x1>10,x4<=x5+10,x5<-5,x3>x2**3)
negative_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3,x4,x5],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add(x1**3>x3**2,x2>0,x2<4,x1>10,x4>x5+10,x5<-5,x3<x2**3)
negative_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3,x4,x5],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add(x1**3<=x3**2,x2<=0,x1<=10,x4<=x5+10,x5>-5,x3<x2**3)
negative_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3,x4,x5],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add(x1**3<=x3**2,x2>=4,x1<=10,x4<=x5+10,x5>-5,x3<x2**3)
negative_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3,x4,x5],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add(x1**3<=x3**2,x2>=4,x1>10,x4>x5+10,x5<-5,x3>x2**3)
negative_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3,x4,x5],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add(x1**3<=x3**2,x2<=0,x1>10,x4>x5+10,x5<-5,x3>x2**3)
negative_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3,x4,x5],[],[],0,[],10))

f= open("./data/exp_5_neg.db","w+")
for neg_ex in negative_sol:
    for row in neg_ex:
        for feat in row:
            f.write(str(feat)+" ")
        f.write("\n")
f.close()
