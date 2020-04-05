import z3
import smt_solver as ss

smt_sol = z3.Solver()
x1 = z3.Real('x1')
x2 = z3.Real('x2')
x3 = z3.Real('x3')
smt_sol.add( x1>x2+4,x3>0,x3<9)
data_x_pos = ss.sols_v2(smt_sol,[x1,x2,x3],[],[],0,[],50)


f= open("./data/exp_1_pos.db","w+")
for row in data_x_pos:
    for feat in row:
        f.write(str(feat)+" ")
    f.write("\n")
f.close()

negatice_sol=[]
smt_sol_neg=z3.Solver()
smt_sol_neg.add( x1<=x2+4,x3>0,x3<9)
negatice_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add( x1>x2+4,x3<=0)
negatice_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add( x1>x2+4,x3>=9)
negatice_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add( x1<=x2+4,x3>=9)
negatice_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3],[],[],0,[],10))

smt_sol_neg=z3.Solver()
smt_sol_neg.add( x1<=x2+4,x3<=0)
negatice_sol.append(ss.sols_v2(smt_sol_neg,[x1,x2,x3],[],[],0,[],10))

f= open("./data/exp_1_neg.db","w+")
for neg_ex in negative_sol:
    for row in neg_ex:
        for feat in row:
            f.write(str(feat)+" ")
        f.write("\n")
f.close()
