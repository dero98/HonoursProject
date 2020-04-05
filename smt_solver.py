import z3
def sols_v2(smt_sol,x_list,lw_bounds_list,up_bounds_list,count,fnl_sol,max_count):
    if(count<max_count):

        if(smt_sol.check() == z3.CheckSatResult(z3.Z3_L_FALSE)):
            return fnl_sol
        else:
            sol=[]
            count = count + 1
            smt_sol.check()
            init_model = smt_sol.model()
            if(lw_bounds_list!=[]):
                for index,x_var, in enumerate (x_list):
                    lw_bound = lw_bounds_list[index]
                    smt_sol.add(x_var<lw_bound)
                return sols_v2(smt_sol,x_list,[],[],count,fnl_sol,max_count)
            elif(up_bounds_list!=[]):
                for index,x_var, in enumerate (x_list):
                    up_bound = up_bounds_list[index]
                    smt_sol.add(x_var>up_bound)
                return sols_v2(smt_sol,x_list,[],[],count,fnl_sol,max_count)
            else:
                for x_var in x_list:

                    x_val = convert_to_float(init_model[x_var].as_decimal(2))
                    sol.append(x_val)
                    up_bound = x_val*1.005
                    lw_bound = x_val*0.995
                    lw_bounds_list.append(lw_bound)
                    up_bounds_list.append(up_bound)
                fnl_sol.append(sol)
                smt_sol_2 = z3.Solver()
                smt_sol_2.add(smt_sol.assertions())
                l1 = sols_v2(smt_sol,x_list,lw_bounds_list,[],count,fnl_sol,max_count)
                l2 = sols_v2(smt_sol_2,x_list,[],up_bounds_list,count,fnl_sol,max_count)
                fnl_l =list(set(map(tuple,l1+l2)))
                return fnl_l

    else:
        return  fnl_sol
        
def convert_to_float(str_num):
    try:
        float_num = float(str_num)
        return float_num
    except ValueError:
        return convert_to_float(str_num[:-1])
