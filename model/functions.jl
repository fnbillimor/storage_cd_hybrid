function stor_dispatch(pm_dict,g,uncert)

    DI_pd = 48
    DI_ph = 2

    stor_g = pm_dict[:"stor"]
    c_i = stor_g.inv_cost[g]
    c_f = stor_g.c_f[g]
    c_deg = stor_g.c_deg[g]
    econ_life = stor_g.econ_life[g]
    tax_life = stor_g.tax_life[g]
    pmax = stor_g.pmax[g]
    dur = stor_g.dur[g]
    emax = pmax*dur
    q_d = stor_g.q_d[g]
    q_c = stor_g.q_c[g]
    gen_avail_f = pm_dict[:"gen_avail_f"]

    # region selection
    price_choice = pm_dict[:"price_choice"]
    demand_choice = pm_dict[:"demand_choice"]

    result_dict = Dict{Any,Any}()
    model = Model()
    price_o = round.(gen_avail_f[!,price_choice],digits=0)
    fcas_o = round.(gen_avail_f[!,[:fcas1,:fcas2,:fcas3,:fcas4,:fcas5,:fcas6,:fcas7,:fcas8]],digits=0)

    Ntime = nrow(gen_avail_f)
    Nyear = gen_avail_f.scen_yr[Ntime]
    Nmonth = gen_avail_f.scen_month[Ntime]
    Nqtr = gen_avail_f.scen_qtr[Ntime]
    Nfcas = 8
    fcas_util = zeros(Nfcas)
    fcas_util[1:3].=0
    fcas_util[4]=0.2
    fcas_util[5:7].=0
    fcas_util[8]=-0.1
    degrad = 0.03
    Random.seed!(123)
    distr_n = Normal(0.0,uncert)
    price_dev = rand(distr_n, Ntime)
    fcas_dev = rand(distr_n,(Ntime,Nfcas))
    price = min.(max.(price_o .+ price_dev,-1000),15000)
    fcas = min.(max.(fcas_o .+ fcas_dev,0),15000)

    #mth_to_qtr = unique(gen_avail_f[:,[:scen_month,:scen_qtr]])
    #qtr_to_yr = unique(gen_avail_f[:,[:scen_qtr,:scen_yr]])

    @variable(model, p_d[t in 1:Ntime] >= 0)
    @variable(model, p_c[t in 1:Ntime] >= 0)
    @variable(model, fcas_d[n in 1:Nfcas,t in 1:Ntime] >= 0)
    @variable(model, soc[t in 1:Ntime] >= 0)

    # operational constraints
    @constraint(model,p_d_max[t in 1:Ntime],p_d[t] <= pmax)
    @constraint(model,p_c_max[t in 1:Ntime],p_c[t] <= pmax)
    @constraint(model,fcas_d_max[n in 1:Nfcas,t in 1:Ntime] , fcas_d[n,t] <= 2*pmax)
    @constraint(model,fcas_d_max_raise[n in 1:3,t in 1:Ntime] ,p_d[t]-p_c[t] + fcas_d[n,t] + fcas_d[4,t] <= pmax)
    @constraint(model,fcas_d_max_lower[n in 5:7,t in 1:Ntime] ,fcas_d[n,t] + fcas_d[8,t] <= p_d[t]-p_c[t] + pmax)

    @constraint(model,soc_max[t in 1:Ntime],soc[t] <= emax*(1-degrad)^(gen_avail_f.scen_yr[t]-1))
    @constraint(model,soc_eq[t in 2:Ntime],soc[t] == soc[t-1] + q_c*p_c[t]/ DI_ph  - (1/q_d)*p_d[t]/ DI_ph )

    @expression(model,spot_rev_di[t in 1:Ntime], (price[t])*p_d[t]  / DI_ph )
    @expression(model,spot_cost_di[t in 1:Ntime], (price[t])*p_c[t]  / DI_ph )
    @expression(model,fcas_rev_di[n in 1:Nfcas,t in 1:Ntime],fcas_d[n,t]*(fcas[t,n] + price[t]*fcas_util[n])/ DI_ph )

    @expression(model,c_deg_di[t in 1:Ntime], 0*c_deg*(p_d[t]+p_c[t])   / DI_ph )
    @expression(model,deg_di[t in 1:Ntime], (p_d[t]+p_c[t]) / DI_ph )

    @expression(model,net_rev_di[t in 1:Ntime],spot_rev_di[t]-spot_cost_di[t]+sum(fcas_rev_di[n,t] for n in 1:Nfcas))
    #@expression(model,net_prof_di[t in 1:Ntime],net_rev_di[t] - c_deg_di[t] )

    @expression(model,net_prof_di[t in 1:Ntime],net_rev_di[t] )

    @expression(model,deg_qtr[q in 1:Nqtr], sum(deg_di[t] for t in gen_avail_f[gen_avail_f.scen_qtr .== q,:].index) )

    # Degradation constraint
    @constraint(model,dec_con[q in 1:Nqtr],deg_qtr[q]<= emax*1*90 )

    @expression(model,sum_prof_di,sum(net_prof_di[t] for t in 1:Ntime))
    @objective(model,Max,sum_prof_di)

    solver_mip = Gurobi.Optimizer
    set_optimizer(model,solver_mip)
    #set_silent(model)
    #set_optimizer_attribute(model, "Method", 2)
    set_optimizer_attribute(model, "FeasibilityTol", 0.01)
    set_optimizer_attribute(model, "OptimalityTol", 0.01)
    #set_optimizer_attribute(model, "NonConvex", 2)
    #set_optimizer_attribute(model, "MIPGap", 0.01)
    #set_optimizer_attribute(model, "Presolve", 1)
    #set_optimizer_attribute(model, "NumericFocus", 3)
    @time optimize!(model)

    p_d_out = value.(model[:p_d])
    p_c_out = value.(model[:p_c])
    fcas_d_out = value.(model[:fcas_d])
    soc_out = value.(model[:soc])

    act_spot_rev_di = price_o .* p_d_out ./ DI_ph
    act_spot_cost_di = price_o .* p_c_out ./ DI_ph
    act_fcas_rev_di = zeros(Ntime,Nfcas)
    for n in 1:Nfcas,t in 1:Ntime
        act_fcas_rev_di[t,n] = fcas_d_out[n,t]*(fcas_o[t,n] + price_o[t]*fcas_util[n])/ DI_ph
    end
    act_c_deg_di = 0 .* c_deg .* (p_c_out.+ p_d_out) ./ DI_ph

    act_net_rev_di = zeros(Ntime)
    act_net_prof_di = zeros(Ntime)
    for t in 1:Ntime
        act_net_rev_di[t] = act_spot_rev_di[t]-act_spot_cost_di[t]+sum(act_fcas_rev_di[t,n] for n in 1:Nfcas)
        act_net_prof_di[t]= act_net_rev_di[t] - act_c_deg_di[t]
    end

    push!(result_dict, "act_spot_rev_di" => act_spot_rev_di)
    push!(result_dict, "act_spot_cost_di" => act_spot_cost_di)
    push!(result_dict, "act_fcas_rev_di" => act_fcas_rev_di)
    push!(result_dict, "act_c_deg_di" => act_c_deg_di)
    push!(result_dict, "act_net_rev_di" => act_net_rev_di)
    push!(result_dict, "act_net_prof_di" => act_net_prof_di)

    push!(result_dict, "p_d_out" => p_d_out)
    push!(result_dict, "p_c_out" => p_c_out)
    push!(result_dict, "fcas_d_out" => fcas_d_out)
    push!(result_dict, "soc_out" => soc_out)

    return result_dict
end

function rev_swap(pm_dict,g,disp_res,con_price_f,vol,fixed_fin,gearing_ratio)

    DI_pd = 48
    DI_ph = 2

    gen_avail_f = pm_dict[:"gen_avail_f"]

    p_c_out = disp_res["p_c_out"]
    p_d_out = disp_res["p_d_out"]
    soc_out = disp_res["soc_out"]
    fcas_d_out = disp_res["fcas_d_out"] # (Nfcas,Ntime)

    energy_rev_DI = disp_res["act_spot_rev_di"]
    energy_cost_DI = disp_res["act_spot_cost_di"]
    fcas_rev_DI = disp_res["act_fcas_rev_di"]
    sum_fcas_rev_DI = sum(Array(fcas_rev_DI),dims=2)
    c_deg = disp_res["act_c_deg_di"]

    gen_g = pm_dict[:"stor"][g,:]
    c_i = gen_g.inv_cost
    c_f = gen_g.c_f
    econ_life = gen_g.econ_life
    tax_life = gen_g.tax_life
    pmax = gen_g.pmax
    aux = gen_g.aux
    dur = gen_g.dur

    r_f = pm_dict[:"r_f"]
    k_d_margin = pm_dict[:"k_d_margin"]
    k_e_margin = pm_dict[:"k_e_margin"]
    tax_rate = pm_dict[:"tax_rate"]

    # debt sizing
    min_dscr = pm_dict[:"min_dscr"]
    ave_dscr = pm_dict[:"ave_dscr"]
    min_k_e_qtr = pm_dict[:"min_k_e_qtr"]
    min_k_e_yr = pm_dict[:"min_k_e_yr"]

    # region selection
    price_choice = pm_dict[:"price_choice"]
    demand_choice = pm_dict[:"demand_choice"]

    model = Model()

    price = round.(gen_avail_f[!,price_choice],digits=0)
    fcas = round.(gen_avail_f[!,[:fcas1,:fcas2,:fcas3,:fcas4,:fcas5,:fcas6,:fcas7,:fcas8]],digits=0)

    Ntime = nrow(gen_avail_f)
    Nyear = gen_avail_f.scen_yr[Ntime]
    Nmonth = gen_avail_f.scen_month[Ntime]

    c_f_yr = c_f .* pmax
    c_f_qtr = c_f_yr / 4
    c_f_mth = c_f_yr / 12

    mth_to_qtr = unique(gen_avail_f[:,[:scen_month,:scen_qtr]])
    qtr_to_yr = unique(gen_avail_f[:,[:scen_qtr,:scen_yr]])

    k_d = r_f + k_d_margin
    k_e = r_f + k_e_margin
    d_t = econ_life - 5
    e_t = econ_life

    k_d_q = k_d/4
    k_e_q = k_e/4
    d_t_q = d_t*4
    e_t_q = e_t*4

    k_d_qtr = k_d_q / (1- ((1+k_d_q)^(-d_t_q))  )
    k_e_qtr = k_e_q / (1- ((1+k_e_q)^(-e_t_q))  )

    k_e_yr = k_e / (1- ((1+k_e)^(-e_t))  )


    rev_di = energy_rev_DI .- energy_cost_DI .+ sum_fcas_rev_DI

    #net_rev_di = rev_di .- c_deg

    rev_qtr = zeros(Nqtr)
    c_deg_qtr = zeros(Nqtr)
    fcas_rev_qtr = zeros(Nqtr)

    for q in 1:Nqtr
        t = gen_avail_f[gen_avail_f.scen_qtr .== q,:].index
        rev_qtr[q] = sum(rev_di[t])
        c_deg_qtr[q] = 0*sum(c_deg[t])
        fcas_rev_qtr[q] = sum(sum_fcas_rev_DI[t])
    end

    solver_mip = Gurobi.Optimizer
    set_optimizer(model,solver_mip)

    if (con_price_f == 0)
        @variable(model, con_price >= 0)
    else
        con_price = con_price_f
    end

    @expression(model,contract_in_qtr[q in 1:Nqtr],vol*con_price )
    @expression(model,contract_out_qtr[q in 1:Nqtr],vol*rev_qtr[q] )
    @expression(model,contract_net_qtr[q in 1:Nqtr],contract_in_qtr[q] - contract_out_qtr[q] )

    @expression(model,net_rev_qtr[q in 1:Nqtr], rev_qtr[q] + contract_net_qtr[q] - c_deg_qtr[q] )
    @expression(model,ebitda_qtr[q in 1:Nqtr],  net_rev_qtr[q] - c_f_qtr )

    @variable(model, debt >= 0)
    @variable(model, equity >= 0)

    @expression(model,debt_service_qtr[m in 1:Nqtr],k_d_qtr*debt)
    @expression(model,int_qtr[m in 1:Nqtr], NPFinancial.ipmt.(Real(k_d_q), m, d_t_q, -1)*debt)
    @expression(model,ppmt_qtr[m in 1:Nqtr],debt_service_qtr[m] - int_qtr[m])
    #@expression(model,ppmt_qtr_test[m in 1:Nqtr],NPFinancial.ppmt.(Real(k_d_q), m, d_t_q,-1)* debt)
    @expression(model,debt_out_qtr[m in 1:Nqtr],debt - sum(ppmt_qtr[n] for n in 1:m))

    @expression(model,depn_qtr[m in 1:Nqtr],c_i*pmax/(tax_life*4))
    @expression(model,tax_qtr[q in 1:Nqtr],(ebitda_qtr[q]-depn_qtr[q])*tax_rate )

    @expression(model,cfads_qtr[q in 1:Nqtr],ebitda_qtr[q] - tax_qtr[q])

    @expression(model,cfads_ex_fcas_qtr[q in 1:Nqtr], ebitda_qtr[q] - tax_qtr[q] - fcas_rev_qtr[q] )

    @expression(model,eq_distrib_qtr[q in 1:Nqtr],cfads_qtr[q] - debt_service_qtr[q])
    @expression(model,eq_distrib_ex_fcas_qtr[q in 1:Nqtr],cfads_ex_fcas_qtr[q] - debt_service_qtr[q])
    @expression(model,eq_distrib_yr[y in 1:Nyear],sum(eq_distrib_qtr[q] for q in qtr_to_yr[qtr_to_yr.scen_yr .== y,:scen_qtr]))

    @constraint(model,total_cap, debt + equity  == c_i*pmax*1.0 )

    if fixed_fin == 1
        @constraint(model,debt_gearing, debt == c_i*pmax*1.0*gearing_ratio )
    elseif fixed_fin == 0
        @constraint(model,dscr_min[q in 1:Nqtr], cfads_qtr[q] >= min_dscr*debt_service_qtr[q] )
        @constraint(model,dscr_ave, sum(cfads_qtr[q] for q in 1:Nqtr) >= k_d_qtr*debt*ave_dscr*Nqtr )
    end

    if (con_price_f == 0)
        #@constraint(model,eq_dist_min_qtr[q in 1:Nqtr], eq_distrib_qtr[q]-(0*fcas_rev_qtr[q]) >= min_k_e_qtr*equity )
        @constraint(model,eq_dist_min_yr[y in 1:Nyear], eq_distrib_yr[y] >= min_k_e_yr*equity )
        @constraint(model,eq_dist_ave, sum(eq_distrib_qtr[q] for q in 1:Nqtr) >= k_e_qtr*equity*Nqtr )
        @objective(model,Min,con_price)
    else
        @variable(model,inv_equity>=0)
        @constraint(model,inv_equity_con, inv_equity*equity  == 1)
        @objective(model,Max, sum(eq_distrib_qtr[q] for q in 1:Nqtr)*inv_equity/Nyear)
        set_optimizer_attribute(model, "NonConvex", 2)
    end

    #set_silent(model)
    #set_optimizer_attribute(model, "NonConvex", 2)
    #set_optimizer_attribute(model, "Method", 2)
    #set_optimizer_attribute(model, "FeasibilityTol", 0.01)
    #set_optimizer_attribute(model, "OptimalityTol", 0.01)
    #set_optimizer_attribute(model, "MIPGap", 0.01)
    #set_optimizer_attribute(model, "Presolve", 1)
    #set_optimizer_attribute(model, "NumericFocus", 3)
    @time optimize!(model)

    return model
end

function rev_swap_yardstick(pm_dict,g,disp_res,disp_res_perfect,con_price_f,vol,fixed_fin,gearing_ratio)

    DI_pd = 48
    DI_ph = 2

    gen_avail_f = pm_dict[:"gen_avail_f"]

    p_c_out = disp_res["p_c_out"]
    p_d_out = disp_res["p_d_out"]
    soc_out = disp_res["soc_out"]
    fcas_d_out = disp_res["fcas_d_out"] # (Nfcas,Ntime)

    energy_rev_DI = disp_res["act_spot_rev_di"]
    energy_cost_DI = disp_res["act_spot_cost_di"]
    fcas_rev_DI = disp_res["act_fcas_rev_di"]
    sum_fcas_rev_DI = sum(Array(fcas_rev_DI),dims=2)
    c_deg = disp_res["act_c_deg_di"]

    # perfect foresight
    p_c_out_perfect = disp_res_perfect["p_c_out"]
    p_d_out_perfect = disp_res_perfect["p_d_out"]
    soc_out_perfect = disp_res_perfect["soc_out"]
    fcas_d_out_perfect = disp_res_perfect["fcas_d_out"] # (Nfcas,Ntime)

    energy_rev_DI_perfect = disp_res_perfect["act_spot_rev_di"]
    energy_cost_DI_perfect = disp_res_perfect["act_spot_cost_di"]
    fcas_rev_DI_perfect = disp_res_perfect["act_fcas_rev_di"]
    sum_fcas_rev_DI_perfect = sum(Array(fcas_rev_DI_perfect),dims=2)
    c_deg_perfect = disp_res_perfect["act_c_deg_di"]


    gen_g = pm_dict[:"stor"][g,:]
    c_i = gen_g.inv_cost
    c_f = gen_g.c_f
    econ_life = gen_g.econ_life
    tax_life = gen_g.tax_life
    pmax = gen_g.pmax
    aux = gen_g.aux
    dur = gen_g.dur

    r_f = pm_dict[:"r_f"]
    k_d_margin = pm_dict[:"k_d_margin"]
    k_e_margin = pm_dict[:"k_e_margin"]
    tax_rate = pm_dict[:"tax_rate"]

    # debt sizing
    min_dscr = pm_dict[:"min_dscr"]
    ave_dscr = pm_dict[:"ave_dscr"]
    min_k_e_qtr = pm_dict[:"min_k_e_qtr"]
    min_k_e_yr = pm_dict[:"min_k_e_yr"]

    # region selection
    price_choice = pm_dict[:"price_choice"]
    demand_choice = pm_dict[:"demand_choice"]

    model = Model()

    price = round.(gen_avail_f[!,price_choice],digits=0)
    fcas = round.(gen_avail_f[!,[:fcas1,:fcas2,:fcas3,:fcas4,:fcas5,:fcas6,:fcas7,:fcas8]],digits=0)

    Ntime = nrow(gen_avail_f)
    Nyear = gen_avail_f.scen_yr[Ntime]
    Nmonth = gen_avail_f.scen_month[Ntime]

    c_f_yr = c_f .* pmax
    c_f_qtr = c_f_yr / 4
    c_f_mth = c_f_yr / 12

    mth_to_qtr = unique(gen_avail_f[:,[:scen_month,:scen_qtr]])
    qtr_to_yr = unique(gen_avail_f[:,[:scen_qtr,:scen_yr]])

    k_d = r_f + k_d_margin
    k_e = r_f + k_e_margin
    d_t = econ_life - 5
    e_t = econ_life

    k_d_q = k_d/4
    k_e_q = k_e/4
    d_t_q = d_t*4
    e_t_q = e_t*4

    k_d_qtr = k_d_q / (1- ((1+k_d_q)^(-d_t_q))  )
    k_e_qtr = k_e_q / (1- ((1+k_e_q)^(-e_t_q))  )

    k_e_yr = k_e / (1- ((1+k_e)^(-e_t))  )


    rev_di = energy_rev_DI .- energy_cost_DI .+ sum_fcas_rev_DI
    rev_di_perfect = energy_rev_DI_perfect .- energy_cost_DI_perfect .+ sum_fcas_rev_DI_perfect

    #net_rev_di = rev_di .- c_deg
    rev_qtr = zeros(Nqtr)
    c_deg_qtr = zeros(Nqtr)
    fcas_rev_qtr = zeros(Nqtr)

    rev_qtr_perfect = zeros(Nqtr)
    c_deg_qtr_perfect = zeros(Nqtr)
    fcas_rev_qtr_perfect = zeros(Nqtr)

    for q in 1:Nqtr
        t = gen_avail_f[gen_avail_f.scen_qtr .== q,:].index
        rev_qtr[q] = sum(rev_di[t])
        c_deg_qtr[q] = sum(c_deg[t])
        fcas_rev_qtr[q] = sum(sum_fcas_rev_DI[t])

        rev_qtr_perfect[q] = sum(rev_di_perfect[t])
        c_deg_qtr_perfect[q] = sum(c_deg_perfect[t])
        fcas_rev_qtr_perfect[q] = sum(sum_fcas_rev_DI_perfect[t])

    end

    solver_mip = Gurobi.Optimizer
    set_optimizer(model,solver_mip)

    if (con_price_f == 0)
        @variable(model, con_price >= 0)
    else
        con_price = con_price_f
    end

    @expression(model,contract_in_qtr[q in 1:Nqtr],vol*con_price )
    @expression(model,contract_out_qtr[q in 1:Nqtr],vol*rev_qtr_perfect[q] )
    @expression(model,contract_net_qtr[q in 1:Nqtr],contract_in_qtr[q] - contract_out_qtr[q] )

    @expression(model,net_rev_qtr[q in 1:Nqtr], rev_qtr[q] + contract_net_qtr[q] - c_deg_qtr[q] )
    @expression(model,ebitda_qtr[q in 1:Nqtr],  net_rev_qtr[q] - c_f_qtr )

    @variable(model, debt >= 0)
    @variable(model, equity >= 0)

    @expression(model,debt_service_qtr[m in 1:Nqtr],k_d_qtr*debt)
    @expression(model,int_qtr[m in 1:Nqtr], NPFinancial.ipmt.(Real(k_d_q), m, d_t_q, -1)*debt)
    @expression(model,ppmt_qtr[m in 1:Nqtr],debt_service_qtr[m] - int_qtr[m])
    #@expression(model,ppmt_qtr_test[m in 1:Nqtr],NPFinancial.ppmt.(Real(k_d_q), m, d_t_q,-1)* debt)
    @expression(model,debt_out_qtr[m in 1:Nqtr],debt - sum(ppmt_qtr[n] for n in 1:m))

    @expression(model,depn_qtr[m in 1:Nqtr],c_i*pmax/(tax_life*4))
    @expression(model,tax_qtr[q in 1:Nqtr],(ebitda_qtr[q]-depn_qtr[q])*tax_rate )

    @expression(model,cfads_qtr[q in 1:Nqtr],ebitda_qtr[q] - tax_qtr[q])

    @expression(model,cfads_ex_fcas_qtr[q in 1:Nqtr], ebitda_qtr[q] - tax_qtr[q] - fcas_rev_qtr[q] )

    @expression(model,eq_distrib_qtr[q in 1:Nqtr],cfads_qtr[q] - debt_service_qtr[q])
    @expression(model,eq_distrib_ex_fcas_qtr[q in 1:Nqtr],cfads_ex_fcas_qtr[q] - debt_service_qtr[q])
    @expression(model,eq_distrib_yr[y in 1:Nyear],sum(eq_distrib_qtr[q] for q in qtr_to_yr[qtr_to_yr.scen_yr .== y,:scen_qtr]))

    @constraint(model,total_cap, debt + equity  == c_i*pmax*1.0 )

    if fixed_fin == 1
        @constraint(model,debt_gearing, debt == c_i*pmax*1.0*gearing_ratio )
    elseif fixed_fin == 0
        @constraint(model,dscr_min[q in 1:Nqtr], cfads_qtr[q] >= min_dscr*debt_service_qtr[q] )
        @constraint(model,dscr_ave, sum(cfads_qtr[q] for q in 1:Nqtr) >= k_d_qtr*debt*ave_dscr*Nqtr )
    end

    if (con_price_f == 0)
        #@constraint(model,eq_dist_min_qtr[q in 1:Nqtr], eq_distrib_qtr[q]-(0*fcas_rev_qtr[q]) >= min_k_e_qtr*equity )
        @constraint(model,eq_dist_min_yr[y in 1:Nyear], eq_distrib_yr[y] >= min_k_e_yr*equity )
        @constraint(model,eq_dist_ave, sum(eq_distrib_qtr[q] for q in 1:Nqtr) >= k_e_qtr*equity*Nqtr )
        @objective(model,Min,con_price)
    else
        @variable(model,inv_equity>=0)
        @constraint(model,inv_equity_con, inv_equity*equity  == 1)
        @objective(model,Max, sum(eq_distrib_qtr[q] for q in 1:Nqtr)*inv_equity/Nyear)
        set_optimizer_attribute(model, "NonConvex", 2)
    end

    #set_silent(model)
    #set_optimizer_attribute(model, "NonConvex", 2)
    #set_optimizer_attribute(model, "Method", 2)
    #set_optimizer_attribute(model, "FeasibilityTol", 0.01)
    #set_optimizer_attribute(model, "OptimalityTol", 0.01)
    #set_optimizer_attribute(model, "MIPGap", 0.01)
    #set_optimizer_attribute(model, "Presolve", 1)
    #set_optimizer_attribute(model, "NumericFocus", 3)
    @time optimize!(model)

    return model
end

function avail(pm_dict,g,disp_res,avail_thres,fixed_fin,gearing_ratio)

    DI_pd = 48
    DI_ph = 2

    gen_avail_f = pm_dict[:"gen_avail_f"]

    p_c_out = disp_res["p_c_out"]
    p_d_out = disp_res["p_d_out"]
    soc_out = disp_res["soc_out"]
    fcas_d_out = disp_res["fcas_d_out"] # (Nfcas,Ntime)

    energy_rev_DI = disp_res["act_spot_rev_di"]
    energy_cost_DI = disp_res["act_spot_cost_di"]
    fcas_rev_DI = disp_res["act_fcas_rev_di"]
    sum_fcas_rev_DI = sum(Array(fcas_rev_DI),dims=2)
    c_deg = disp_res["act_c_deg_di"]

    gen_g = pm_dict[:"stor"][g,:]
    c_i = gen_g.inv_cost
    c_f = gen_g.c_f
    econ_life = gen_g.econ_life
    tax_life = gen_g.tax_life
    pmax = gen_g.pmax
    aux = gen_g.aux
    dur = gen_g.dur

    r_f = pm_dict[:"r_f"]
    k_d_margin = pm_dict[:"k_d_margin"]
    k_e_margin = pm_dict[:"k_e_margin"]
    tax_rate = pm_dict[:"tax_rate"]

    # debt sizing
    min_dscr = pm_dict[:"min_dscr"]
    ave_dscr = pm_dict[:"ave_dscr"]
    min_k_e_qtr = pm_dict[:"min_k_e_qtr"]
    min_k_e_yr = pm_dict[:"min_k_e_yr"]

    # region selection
    price_choice = pm_dict[:"price_choice"]
    demand_choice = pm_dict[:"demand_choice"]

    model = Model()

    price = round.(gen_avail_f[!,price_choice],digits=0)
    fcas = round.(gen_avail_f[!,[:fcas1,:fcas2,:fcas3,:fcas4,:fcas5,:fcas6,:fcas7,:fcas8]],digits=0)

    Ntime = nrow(gen_avail_f)
    Nyear = gen_avail_f.scen_yr[Ntime]
    Nmonth = gen_avail_f.scen_month[Ntime]
    Nqtr = gen_avail_f.scen_qtr[Ntime]

    c_f_yr = c_f .* pmax
    c_f_qtr = c_f_yr / 4
    c_f_mth = c_f_yr / 12

    mth_to_qtr = unique(gen_avail_f[:,[:scen_month,:scen_qtr]])
    qtr_to_yr = unique(gen_avail_f[:,[:scen_qtr,:scen_yr]])

    k_d = r_f + k_d_margin
    k_e = r_f + k_e_margin
    d_t = econ_life - 5
    e_t = econ_life

    k_d_q = k_d/4
    k_e_q = k_e/4
    d_t_q = d_t*4
    e_t_q = e_t*4

    k_d_qtr = k_d_q / (1- ((1+k_d_q)^(-d_t_q))  )
    k_e_qtr = k_e_q / (1- ((1+k_e_q)^(-e_t_q))  )

    k_e_yr = k_e / (1- ((1+k_e)^(-e_t))  )


    rev_di = energy_rev_DI .- energy_cost_DI .+ sum_fcas_rev_DI

    #net_rev_di = rev_di .- c_deg

    rev_qtr = zeros(Nqtr)
    c_deg_qtr = zeros(Nqtr)
    fcas_rev_qtr = zeros(Nqtr)

    for q in 1:Nqtr
        t = gen_avail_f[gen_avail_f.scen_qtr .== q,:].index
        rev_qtr[q] = sum(rev_di[t])
        c_deg_qtr[q] = 0*sum(c_deg[t])
        fcas_rev_qtr[q] = sum(sum_fcas_rev_DI[t])
    end

    rev_qtr = avail_thres .+ rev_qtr
    net_rev_qtr = rev_qtr .- c_deg_qtr
    ebitda_qtr = net_rev_qtr .- c_f_qtr
    solver_mip = Gurobi.Optimizer
    set_optimizer(model,solver_mip)

    @variable(model, debt >= 0)
    @variable(model, equity >= 0)

    @expression(model,debt_service_qtr[m in 1:Nqtr],k_d_qtr*debt)
    @expression(model,int_qtr[m in 1:Nqtr], NPFinancial.ipmt.(Real(k_d_q), m, d_t_q, -1)*debt)
    @expression(model,ppmt_qtr[m in 1:Nqtr],debt_service_qtr[m] - int_qtr[m])
    #@expression(model,ppmt_qtr_test[m in 1:Nqtr],NPFinancial.ppmt.(Real(k_d_q), m, d_t_q,-1)* debt)
    @expression(model,debt_out_qtr[m in 1:Nqtr],debt - sum(ppmt_qtr[n] for n in 1:m))
    @expression(model,rev_qtr_out[q in 1:Nqtr],rev_qtr[q])
    @expression(model,depn_qtr[m in 1:Nqtr],c_i*pmax/(tax_life*4))
    @expression(model,tax_qtr[q in 1:Nqtr],(ebitda_qtr[q]-depn_qtr[q])*tax_rate )

    @expression(model,spot_qtr[q in 1:Nqtr],ebitda_qtr[q] - tax_qtr[q])
    @expression(model,cfads_qtr[q in 1:Nqtr],ebitda_qtr[q] - tax_qtr[q])

    @expression(model,cfads_ex_fcas_qtr[q in 1:Nqtr], ebitda_qtr[q] - tax_qtr[q] - fcas_rev_qtr[q] )

    @expression(model,eq_distrib_qtr[q in 1:Nqtr],cfads_qtr[q] - debt_service_qtr[q])
    @expression(model,eq_distrib_ex_fcas_qtr[q in 1:Nqtr],cfads_ex_fcas_qtr[q] - debt_service_qtr[q])
    @constraint(model,total_cap, debt + equity  == c_i*pmax*1.0 )

    if fixed_fin == 1
        @constraint(model,debt_gearing, debt == c_i*pmax*1.0*gearing_ratio )
    else
        @constraint(model,dscr_min[q in 1:Nqtr], cfads_qtr[q] >= min_dscr*debt_service_qtr[q] )
        @constraint(model,dscr_ave, sum(cfads_qtr[q] for q in 1:Nqtr) >= k_d_qtr*debt*ave_dscr*Nqtr )
    end

    @variable(model,inv_equity>=0)
    @constraint(model,inv_equity_con, inv_equity*equity  == 1)
    @objective(model,Max, sum(eq_distrib_qtr[q] for q in 1:Nqtr)*inv_equity/Nyear)
    set_optimizer_attribute(model, "NonConvex", 2)

    #set_silent(model)
    #set_optimizer_attribute(model, "NonConvex", 2)
    #set_optimizer_attribute(model, "Method", 2)
    #set_optimizer_attribute(model, "FeasibilityTol", 0.01)
    #set_optimizer_attribute(model, "OptimalityTol", 0.01)
    #set_optimizer_attribute(model, "MIPGap", 0.01)
    #set_optimizer_attribute(model, "Presolve", 1)
    #set_optimizer_attribute(model, "NumericFocus", 3)
    @time optimize!(model)

    return model
end

function floor_con(pm_dict,g,disp_res,floor_thres,fixed_fin,gearing_ratio)

    DI_pd = 48
    DI_ph = 2

    gen_avail_f = pm_dict[:"gen_avail_f"]

    p_c_out = disp_res["p_c_out"]
    p_d_out = disp_res["p_d_out"]
    soc_out = disp_res["soc_out"]
    fcas_d_out = disp_res["fcas_d_out"] # (Nfcas,Ntime)

    energy_rev_DI = disp_res["act_spot_rev_di"]
    energy_cost_DI = disp_res["act_spot_cost_di"]
    fcas_rev_DI = disp_res["act_fcas_rev_di"]
    sum_fcas_rev_DI = sum(Array(fcas_rev_DI),dims=2)
    c_deg = disp_res["act_c_deg_di"]

    gen_g = pm_dict[:"stor"][g,:]
    c_i = gen_g.inv_cost
    c_f = gen_g.c_f
    econ_life = gen_g.econ_life
    tax_life = gen_g.tax_life
    pmax = gen_g.pmax
    aux = gen_g.aux
    dur = gen_g.dur

    r_f = pm_dict[:"r_f"]
    k_d_margin = pm_dict[:"k_d_margin"]
    k_e_margin = pm_dict[:"k_e_margin"]
    tax_rate = pm_dict[:"tax_rate"]

    # debt sizing
    min_dscr = pm_dict[:"min_dscr"]
    ave_dscr = pm_dict[:"ave_dscr"]
    min_k_e_qtr = pm_dict[:"min_k_e_qtr"]
    min_k_e_yr = pm_dict[:"min_k_e_yr"]

    # region selection
    price_choice = pm_dict[:"price_choice"]
    demand_choice = pm_dict[:"demand_choice"]

    model = Model()

    price = round.(gen_avail_f[!,price_choice],digits=0)
    fcas = round.(gen_avail_f[!,[:fcas1,:fcas2,:fcas3,:fcas4,:fcas5,:fcas6,:fcas7,:fcas8]],digits=0)

    Ntime = nrow(gen_avail_f)
    Nyear = gen_avail_f.scen_yr[Ntime]
    Nmonth = gen_avail_f.scen_month[Ntime]
    Nqtr = gen_avail_f.scen_qtr[Ntime]

    c_f_yr = c_f .* pmax
    c_f_qtr = c_f_yr / 4
    c_f_mth = c_f_yr / 12

    mth_to_qtr = unique(gen_avail_f[:,[:scen_month,:scen_qtr]])
    qtr_to_yr = unique(gen_avail_f[:,[:scen_qtr,:scen_yr]])

    k_d = r_f + k_d_margin
    k_e = r_f + k_e_margin
    d_t = econ_life - 5
    e_t = econ_life

    k_d_q = k_d/4
    k_e_q = k_e/4
    d_t_q = d_t*4
    e_t_q = e_t*4

    k_d_qtr = k_d_q / (1- ((1+k_d_q)^(-d_t_q))  )
    k_e_qtr = k_e_q / (1- ((1+k_e_q)^(-e_t_q))  )

    k_e_yr = k_e / (1- ((1+k_e)^(-e_t))  )


    rev_di = energy_rev_DI .- energy_cost_DI .+ sum_fcas_rev_DI

    #net_rev_di = rev_di .- c_deg

    rev_qtr = zeros(Nqtr)
    c_deg_qtr = zeros(Nqtr)
    fcas_rev_qtr = zeros(Nqtr)

    for q in 1:Nqtr
        t = gen_avail_f[gen_avail_f.scen_qtr .== q,:].index
        rev_qtr[q] = sum(rev_di[t])
        c_deg_qtr[q] = sum(c_deg[t])
        fcas_rev_qtr[q] = sum(sum_fcas_rev_DI[t])
    end

    rev_qtr = max.(floor_thres,rev_qtr)
    net_rev_qtr = rev_qtr .- c_deg_qtr
    ebitda_qtr = net_rev_qtr .- c_f_qtr
    solver_mip = Gurobi.Optimizer
    set_optimizer(model,solver_mip)

    @variable(model, debt >= 0)
    @variable(model, equity >= 0)

    @expression(model,debt_service_qtr[m in 1:Nqtr],k_d_qtr*debt)
    @expression(model,int_qtr[m in 1:Nqtr], NPFinancial.ipmt.(Real(k_d_q), m, d_t_q, -1)*debt)
    @expression(model,ppmt_qtr[m in 1:Nqtr],debt_service_qtr[m] - int_qtr[m])
    #@expression(model,ppmt_qtr_test[m in 1:Nqtr],NPFinancial.ppmt.(Real(k_d_q), m, d_t_q,-1)* debt)
    @expression(model,debt_out_qtr[m in 1:Nqtr],debt - sum(ppmt_qtr[n] for n in 1:m))

    @expression(model,depn_qtr[m in 1:Nqtr],c_i*pmax/(tax_life*4))
    @expression(model,tax_qtr[q in 1:Nqtr],(ebitda_qtr[q]-depn_qtr[q])*tax_rate )

    @expression(model,cfads_qtr[q in 1:Nqtr],ebitda_qtr[q] - tax_qtr[q])

    @expression(model,cfads_ex_fcas_qtr[q in 1:Nqtr], ebitda_qtr[q] - tax_qtr[q] - fcas_rev_qtr[q] )

    @expression(model,eq_distrib_qtr[q in 1:Nqtr],cfads_qtr[q] - debt_service_qtr[q])
    @expression(model,eq_distrib_ex_fcas_qtr[q in 1:Nqtr],cfads_ex_fcas_qtr[q] - debt_service_qtr[q])
    @constraint(model,total_cap, debt + equity  == c_i*pmax*1.0 )

    if fixed_fin == 1
        @constraint(model,debt_gearing, debt == c_i*pmax*1.0*gearing_ratio )
    elseif fixed_fin == 0
        @constraint(model,dscr_min[q in 1:Nqtr], cfads_qtr[q] >= min_dscr*debt_service_qtr[q] )
        @constraint(model,dscr_ave, sum(cfads_qtr[q] for q in 1:Nqtr) >= k_d_qtr*debt*ave_dscr*Nqtr )
    end

    @variable(model,inv_equity>=0)
    @constraint(model,inv_equity_con, inv_equity*equity  == 1)
    @objective(model,Max, sum(eq_distrib_qtr[q] for q in 1:Nqtr)*inv_equity/Nyear)
    set_optimizer_attribute(model, "NonConvex", 2)

    #set_silent(model)
    #set_optimizer_attribute(model, "NonConvex", 2)
    #set_optimizer_attribute(model, "Method", 2)
    #set_optimizer_attribute(model, "FeasibilityTol", 0.01)
    #set_optimizer_attribute(model, "OptimalityTol", 0.01)
    #set_optimizer_attribute(model, "MIPGap", 0.01)
    #set_optimizer_attribute(model, "Presolve", 1)
    #set_optimizer_attribute(model, "NumericFocus", 3)
    @time optimize!(model)

    return model
end

function collar(pm_dict,g,disp_res,floor_thres,cap_thres,cap_share_proj,fixed_fin,gearing_ratio)

    DI_pd = 48
    DI_ph = 2

    gen_avail_f = pm_dict[:"gen_avail_f"]

    p_c_out = disp_res["p_c_out"]
    p_d_out = disp_res["p_d_out"]
    soc_out = disp_res["soc_out"]
    fcas_d_out = disp_res["fcas_d_out"] # (Nfcas,Ntime)

    energy_rev_DI = disp_res["act_spot_rev_di"]
    energy_cost_DI = disp_res["act_spot_cost_di"]
    fcas_rev_DI = disp_res["act_fcas_rev_di"]
    sum_fcas_rev_DI = sum(Array(fcas_rev_DI),dims=2)
    c_deg = disp_res["act_c_deg_di"]

    gen_g = pm_dict[:"stor"][g,:]
    c_i = gen_g.inv_cost
    c_f = gen_g.c_f
    econ_life = gen_g.econ_life
    tax_life = gen_g.tax_life
    pmax = gen_g.pmax
    aux = gen_g.aux
    dur = gen_g.dur

    r_f = pm_dict[:"r_f"]
    k_d_margin = pm_dict[:"k_d_margin"]
    k_e_margin = pm_dict[:"k_e_margin"]
    tax_rate = pm_dict[:"tax_rate"]

    # debt sizing
    min_dscr = pm_dict[:"min_dscr"]
    ave_dscr = pm_dict[:"ave_dscr"]
    min_k_e_qtr = pm_dict[:"min_k_e_qtr"]
    min_k_e_yr = pm_dict[:"min_k_e_yr"]

    # region selection
    price_choice = pm_dict[:"price_choice"]
    demand_choice = pm_dict[:"demand_choice"]

    model = Model()

    price = round.(gen_avail_f[!,price_choice],digits=0)
    fcas = round.(gen_avail_f[!,[:fcas1,:fcas2,:fcas3,:fcas4,:fcas5,:fcas6,:fcas7,:fcas8]],digits=0)

    Ntime = nrow(gen_avail_f)
    Nyear = gen_avail_f.scen_yr[Ntime]
    Nmonth = gen_avail_f.scen_month[Ntime]
    Nqtr = gen_avail_f.scen_qtr[Ntime]

    c_f_yr = c_f .* pmax
    c_f_qtr = c_f_yr / 4
    c_f_mth = c_f_yr / 12

    mth_to_qtr = unique(gen_avail_f[:,[:scen_month,:scen_qtr]])
    qtr_to_yr = unique(gen_avail_f[:,[:scen_qtr,:scen_yr]])

    k_d = r_f + k_d_margin
    k_e = r_f + k_e_margin
    d_t = econ_life - 5
    e_t = econ_life

    k_d_q = k_d/4
    k_e_q = k_e/4
    d_t_q = d_t*4
    e_t_q = e_t*4

    k_d_qtr = k_d_q / (1- ((1+k_d_q)^(-d_t_q))  )
    k_e_qtr = k_e_q / (1- ((1+k_e_q)^(-e_t_q))  )

    k_e_yr = k_e / (1- ((1+k_e)^(-e_t))  )


    rev_di = energy_rev_DI .- energy_cost_DI .+ sum_fcas_rev_DI

    #net_rev_di = rev_di .- c_deg

    rev_qtr = zeros(Nqtr)
    c_deg_qtr = zeros(Nqtr)
    fcas_rev_qtr = zeros(Nqtr)

    for q in 1:Nqtr
        t = gen_avail_f[gen_avail_f.scen_qtr .== q,:].index
        rev_qtr[q] = sum(rev_di[t])
        c_deg_qtr[q] = sum(c_deg[t])
        fcas_rev_qtr[q] = sum(sum_fcas_rev_DI[t])
    end

    rev_qtr = max.(floor_thres,rev_qtr)
    excess_qtr = max.(rev_qtr .- cap_thres,0)
    excess_share_proj = cap_share_proj .* excess_qtr
    rev_qtr = min.(cap_thres,rev_qtr)
    rev_qtr = rev_qtr .+ excess_share_proj
    net_rev_qtr = rev_qtr .- (0 .* c_deg_qtr)
    ebitda_qtr = net_rev_qtr .- c_f_qtr
    solver_mip = Gurobi.Optimizer
    set_optimizer(model,solver_mip)

    @variable(model, debt >= 0)
    @variable(model, equity >= 0)

    @expression(model,debt_service_qtr[m in 1:Nqtr],k_d_qtr*debt)
    @expression(model,int_qtr[m in 1:Nqtr], NPFinancial.ipmt.(Real(k_d_q), m, d_t_q, -1)*debt)
    @expression(model,ppmt_qtr[m in 1:Nqtr],debt_service_qtr[m] - int_qtr[m])
    #@expression(model,ppmt_qtr_test[m in 1:Nqtr],NPFinancial.ppmt.(Real(k_d_q), m, d_t_q,-1)* debt)
    @expression(model,debt_out_qtr[m in 1:Nqtr],debt - sum(ppmt_qtr[n] for n in 1:m))

    @expression(model,depn_qtr[m in 1:Nqtr],c_i*pmax/(tax_life*4))
    @expression(model,tax_qtr[q in 1:Nqtr],(ebitda_qtr[q]-depn_qtr[q])*tax_rate )

    @expression(model,rev_qtr_out[q in 1:Nqtr],rev_qtr[q])
    @expression(model,cfads_qtr[q in 1:Nqtr],ebitda_qtr[q] - tax_qtr[q])

    @expression(model,cfads_ex_fcas_qtr[q in 1:Nqtr], ebitda_qtr[q] - tax_qtr[q] - fcas_rev_qtr[q] )

    @expression(model,eq_distrib_qtr[q in 1:Nqtr],cfads_qtr[q] - debt_service_qtr[q])
    @expression(model,eq_distrib_ex_fcas_qtr[q in 1:Nqtr],cfads_ex_fcas_qtr[q] - debt_service_qtr[q])
    @constraint(model,total_cap, debt + equity  == c_i*pmax*1.0 )

    if fixed_fin == 1
        @constraint(model,debt_gearing, debt == c_i*pmax*1.0*gearing_ratio )
    else
        @constraint(model,dscr_min[q in 1:Nqtr], cfads_qtr[q] >= min_dscr*debt_service_qtr[q] )
        @constraint(model,dscr_ave, sum(cfads_qtr[q] for q in 1:Nqtr) >= k_d_qtr*debt*ave_dscr*Nqtr )
    end

    @variable(model,inv_equity>=0)
    @constraint(model,inv_equity_con, inv_equity*equity  == 1)
    @objective(model,Max, sum(eq_distrib_qtr[q] for q in 1:Nqtr)*inv_equity/Nyear)
    set_optimizer_attribute(model, "NonConvex", 2)

    #set_silent(model)
    #set_optimizer_attribute(model, "NonConvex", 2)
    #set_optimizer_attribute(model, "Method", 2)
    #set_optimizer_attribute(model, "FeasibilityTol", 0.01)
    #set_optimizer_attribute(model, "OptimalityTol", 0.01)
    #set_optimizer_attribute(model, "MIPGap", 0.01)
    #set_optimizer_attribute(model, "Presolve", 1)
    #set_optimizer_attribute(model, "NumericFocus", 3)
    @time optimize!(model)

    return model
end
