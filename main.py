import os
import copy
import math
import numpy as np
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

# ====================================================
# CBC Portable Solver Patch
# ====================================================
# GANTI PATH INI SESUAI LOKASI CBC.EXE PADA SISTEM BOS
CBC_PATH = r"D:\Solver\cbc\bin\cbc.exe"

def load_cbc_solver():
    """Load CBC portable - WORKING VERSION"""
    if not os.path.exists(CBC_PATH):
        raise FileNotFoundError(f"CBC not found at: {CBC_PATH}")
    
    solver = SolverFactory("cbc", executable=CBC_PATH)
    
    print(f">>> ✓ CBC Solver loaded successfully")
    print(f">>> Version: 2.10.12 (from test)")
    print(f">>> Location: {CBC_PATH}")
    
    return solver

# ====================================================
# USER SETTINGS
# ====================================================
DATA_FOLDER = "data"    # folder containing CSV files
RESULT_FOLDER = "result"  # folder to write outputs
MAX_ITER = 10
TOL_H = 1e-3
Y_LEVELS = [0.25, 0.5, 0.75, 1.0]
BIGM_FACTOR = 1.2
SOLVER_NAME = "cbc"

# Ensure result folder exists
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ====================================================
# LOAD CSV data
# ====================================================
def load_data(folder):
    def load_csv(name):
        path = os.path.join(folder, name + ".csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        return pd.read_csv(path)

    lahan_df        = load_csv("lahan")
    mills_df        = load_csv("penggilingan")
    warehouses_df   = load_csv("gudang")
    distributors_df = load_csv("distributor")
    markets_df      = load_csv("pasar")
    transport_df    = load_csv("transport")

    input_df = None
    path_input = os.path.join(folder, "input_produksi.csv")
    if os.path.exists(path_input):
        input_df = pd.read_csv(path_input)

    return {
        "lahan": lahan_df,
        "mills": mills_df,
        "warehouses": warehouses_df,
        "distributors": distributors_df,
        "markets": markets_df,
        "transport": transport_df,
        "input": input_df
    }

# ====================================================
# PREPARE PARAMETERS
# ====================================================
def prepare_params(d):
    """
    Read CSV dataframes and prepare parameter dictionaries for the model.
    Also precompute allowed arc sets for (L->G, G->W, W->D, D->M) from transport.csv.
    """
    # SETS
    L = list(d["lahan"]["lahan"].astype(str))
    G = list(d["mills"]["mill"].astype(str))
    W = list(d["warehouses"]["warehouse"].astype(str))
    D = list(d["distributors"]["distributor"].astype(str))
    M = list(d["markets"]["market"].astype(str))
    T = [1]   # single period
    I = []

    # input types (ignore biaya_ columns)
    if d["input"] is not None:
        cols = list(d["input"].columns)
        if "lahan" in cols:
            I = [c for c in cols if c not in ["lahan"] and not c.startswith("biaya")]
    else:
        I = ["pupuk", "benih", "pestisida"]

    S_levels = Y_LEVELS

    # -------------------------------------------------
    # PARAMETERS FROM CSVs
    # -------------------------------------------------
    # LAHAN
    lahan_df = d["lahan"]
    # required columns: lahan, luas_ha
    A = {row["lahan"]: float(row["luas_ha"]) for _, row in lahan_df.iterrows()}

    # baseline_prod: try multiple possible column names
    if "produktivitas_ton" in lahan_df.columns:
        baseline_prod = {row["lahan"]: float(row["produktivitas_ton"]) for _, row in lahan_df.iterrows()}
    elif "produktifitas_ton" in lahan_df.columns:
        baseline_prod = {row["lahan"]: float(row["produktifitas_ton"]) for _, row in lahan_df.iterrows()}
    elif "produktivitas" in lahan_df.columns:
        baseline_prod = {row["lahan"]: float(row["produktivitas"]) for _, row in lahan_df.iterrows()}
    else:
        baseline_prod = {row["lahan"]: 0.0 for _, row in lahan_df.iterrows()}

    # risiko and kualitas_awal (fallbacks if missing)
    risiko_lahan = {row["lahan"]: float(row["risiko"]) if "risiko" in lahan_df.columns else 0.0 for _, row in lahan_df.iterrows()}
    kualitas_awal = {row["lahan"]: float(row["kualitas_awal"]) if "kualitas_awal" in lahan_df.columns else 0.8 for _, row in lahan_df.iterrows()}

    # MILLS
    mills_df = d["mills"]
    Qcap_g = {row["mill"]: float(row["kapasitas_ton"]) for _, row in mills_df.iterrows()}
    Cmill = {row["mill"]: float(row["biaya_proses"]) for _, row in mills_df.iterrows()}
    Fix_mill = {row["mill"]: float(row["biaya_tetap"]) if "biaya_tetap" in mills_df.columns else 0.0 for _, row in mills_df.iterrows()}
    conv = {row["mill"]: float(row["efisiensi"]) if "efisiensi" in mills_df.columns else 1.0 for _, row in mills_df.iterrows()}

    # WAREHOUSES
    wh_df = d["warehouses"]
    Qcap_w = {row["warehouse"]: float(row["kapasitas_ton"]) for _, row in wh_df.iterrows()}
    Cstore = {row["warehouse"]: float(row["biaya_simpan"]) for _, row in wh_df.iterrows()}
    Fix_ware = {row["warehouse"]: float(row["biaya_tetap"]) if "biaya_tetap" in wh_df.columns else 0.0 for _, row in wh_df.iterrows()}

    # DISTRIBUTORS
    dist_df = d["distributors"]
    Cdist = {row["distributor"]: float(row["biaya_handling"]) for _, row in dist_df.iterrows()}

    # MARKETS
    markets_df = d["markets"]
    Demand = {(row["market"], 1): float(row["permintaan_ton"]) for _, row in markets_df.iterrows()}
    Price_market = {(row["market"], 1): float(row["harga_jual"]) for _, row in markets_df.iterrows()}
    pref_quality = {(row["market"], 1): float(row["preferensi_kualitas"]) if "preferensi_kualitas" in markets_df.columns else 0.0 for _, row in markets_df.iterrows()}

    # TRANSPORT: compute per-pair unit cost (distance*per_km + fixed)
    trans = {}
    for _, row in d["transport"].iterrows():
        a = str(row["origin"]); b = str(row["dest"])
        cost = 0.0
        if "jarak_km" in row.index and "biaya_per_km" in row.index:
            try:
                cost += float(row["jarak_km"]) * float(row["biaya_per_km"])
            except:
                pass
        if "biaya_tetap" in row.index:
            try:
                cost += float(row["biaya_tetap"])
            except:
                pass
        trans[(a, b)] = float(cost)

    # INPUT COSTS & XMAX from input_produksi.csv
    Cinput = {}
    xmax = {}
    if d["input"] is not None:
        df = d["input"]
        for _, row in df.iterrows():
            for col in df.columns:
                if col == "lahan":
                    continue
                if col.startswith("biaya_"):
                    key = col.replace("biaya_", "")
                    try:
                        Cinput[key] = float(row[col])
                    except:
                        pass
                else:
                    try:
                        val = float(row[col])
                        xmax[col] = max(xmax.get(col, 0.0), val)
                    except:
                        pass
    # fallbacks
    if not Cinput:
        Cinput = {"pupuk": 1200.0, "benih": 2500.0, "pestisida": 3000.0}
    if not xmax:
        xmax = {"pupuk": 500.0, "benih": 200.0, "pestisida": 100.0}

    # OTHER PARAMETERS
    H0 = 5000.0
    delta = 200.0
    eta = 0.1
    HG = {1: 4800.0}

    alpha = {it: 0.01 for it in I}
    beta = {"intensive": 0.5, "conventional": 0.0}
    lambda_s = {lvl: 0.5 for lvl in S_levels}

    labor_cost_per_ton = 100.0
    water_cost_per_unit = 50.0

    # -------------------------------------------------
    # WATER PARAMETERS (defaults)
    # -------------------------------------------------
    rain = {1: 120.0}
    evap = {1: 80.0}
    Wreq = {l: 50.0 * A[l] for l in L}  # default requirement proportional to area
    theta = 0.05

    # -------------------------------------------------
    # PRECOMPUTE ALLOWED ARCS FROM transport.csv
    # -------------------------------------------------
    allowed_LG = set((o, d) for (o, d) in trans.keys() if (o in L and d in G))
    allowed_GW = set((o, d) for (o, d) in trans.keys() if (o in G and d in W))
    allowed_WD = set((o, d) for (o, d) in trans.keys() if (o in W and d in D))
    allowed_DM = set((o, d) for (o, d) in trans.keys() if (o in D and d in M))

    # assemble params dict
    params = dict(
        L=L, G=G, W=W, D=D, M=M, T=T, I=I, S_levels=S_levels,
        A=A,
        baseline_prod=baseline_prod,
        risiko_lahan=risiko_lahan,
        kualitas_awal=kualitas_awal,
        Qcap_g=Qcap_g, Cmill=Cmill, Fix_mill=Fix_mill, conv=conv,
        Qcap_w=Qcap_w, Cstore=Cstore, Fix_ware=Fix_ware,
        Cdist=Cdist, Demand=Demand, Price_market=Price_market,
        pref_quality=pref_quality,
        trans=trans, Cinput=Cinput, xmax=xmax,
        H0=H0, delta=delta, eta=eta, HG=HG,
        alpha=alpha, beta=beta, lambda_s=lambda_s,
        labor_cost_per_ton=labor_cost_per_ton, water_cost_per_unit=water_cost_per_unit,
        # water params
        rain=rain, evap=evap, Wreq=Wreq, theta=theta,
        # allowed arcs
        allowed_LG=allowed_LG, allowed_GW=allowed_GW, allowed_WD=allowed_WD, allowed_DM=allowed_DM
    )

    return params


def build_pyomo_model(params, H_param):
    model = ConcreteModel()

    # sets
    model.L = Set(initialize=params["L"])
    model.G = Set(initialize=params["G"])
    model.W = Set(initialize=params["W"])
    model.D = Set(initialize=params["D"])
    model.M = Set(initialize=params["M"])
    model.T = Set(initialize=params["T"])
    model.I = Set(initialize=params["I"])
    model.K = RangeSet(0, len(params["S_levels"]) - 1)

    # params mapped
    model.A = Param(model.L, initialize=params["A"])
    model.baseline = Param(model.L, initialize=params["baseline_prod"])
    model.risk_l = Param(model.L, initialize=params["risiko_lahan"])
    model.k0 = Param(model.L, initialize=params["kualitas_awal"])

    model.Qcap_g = Param(model.G, initialize=params["Qcap_g"])
    model.Qcap_w = Param(model.W, initialize=params["Qcap_w"])

    # allowed arcs (as python sets inside params)
    allowed_LG = params.get("allowed_LG", set())
    allowed_GW = params.get("allowed_GW", set())
    allowed_WD = params.get("allowed_WD", set())
    allowed_DM = params.get("allowed_DM", set())

    # variables
    model.x = Var(model.I, model.L, model.T, domain=NonNegativeReals)          # inputs
    model.z_intensive = Var(model.L, model.T, domain=Binary)                  # pattern choice

    # discretization b_k and linearized s
    model.b = Var(model.L, model.T, model.K, domain=Binary)
    model.s = Var(model.L, model.T, model.K, domain=NonNegativeReals)

    model.Q = Var(model.L, model.T, domain=NonNegativeReals)                  # output sold from lahan

    # flows
    model.q_LG = Var(model.L, model.G, model.T, domain=NonNegativeReals)
    model.q_GW = Var(model.G, model.W, model.T, domain=NonNegativeReals)
    model.q_WD = Var(model.W, model.D, model.T, domain=NonNegativeReals)
    model.q_DM = Var(model.D, model.M, model.T, domain=NonNegativeReals)

    # facility activation
    model.Yg = Var(model.G, domain=Binary)
    model.Yw = Var(model.W, domain=Binary)

    # water variables
    model.irrig = Var(model.L, model.T, domain=NonNegativeReals)
    model.Water = Var(model.L, model.T, domain=NonNegativeReals)
    model.Gamma = Var(model.L, model.T, domain=NonNegativeReals)

    # production expression
    alpha = params["alpha"]
    beta = params["beta"]
    theta = params.get("theta", 0.0)

    def prod_expr(model, l, t):
        EF = sum(alpha[it] * model.x[it, l, t] for it in model.I)
        pattern = beta.get("intensive", 0.0) * model.z_intensive[l, t]
        risiko = model.risk_l[l]
        # include water-stress penalty - theta * Gamma
        return model.baseline[l] + EF + pattern - risiko - theta * model.Gamma[l, t]
    model.Prod = Expression(model.L, model.T, rule=prod_expr)

    # HP = Prod * area
    def HP_rule(model, l, t):
        return model.Prod[l, t] * model.A[l]
    model.HP = Expression(model.L, model.T, rule=HP_rule)

    # big-M for linearization
    HP_max = {}
    for l in params["L"]:
        HP_max[l] = params["A"][l] * (params["baseline_prod"].get(l, 0.0) + 2.0)
    bigM = {l: BIGM_FACTOR * HP_max[l] for l in params["L"]}

    # s = HP * b linearization constraints
    def s_up1(model, l, t, k):
        return model.s[l, t, k] <= model.HP[l, t]
    model.s_up1 = Constraint(model.L, model.T, model.K, rule=s_up1)

    def s_up2(model, l, t, k):
        return model.s[l, t, k] <= bigM[l] * model.b[l, t, k]
    model.s_up2 = Constraint(model.L, model.T, model.K, rule=s_up2)

    def s_low(model, l, t, k):
        return model.s[l, t, k] >= model.HP[l, t] - bigM[l] * (1 - model.b[l, t, k])
    model.s_low = Constraint(model.L, model.T, model.K, rule=s_low)

    def s_nonneg(model, l, t, k):
        return model.s[l, t, k] >= 0
    model.s_nonneg = Constraint(model.L, model.T, model.K, rule=s_nonneg)

    # exactly one level
    def one_level(model, l, t):
        return sum(model.b[l, t, k] for k in model.K) == 1
    model.one_level = Constraint(model.L, model.T, rule=one_level)

    # Q definition
    y_lev = params["S_levels"]
    def Q_def(model, l, t):
        return model.Q[l, t] == sum(y_lev[k] * model.s[l, t, k] for k in model.K)
    model.Q_def = Constraint(model.L, model.T, rule=Q_def)

    # flow balance: supply L -> G equals Q
    def supply_rule(model, l, t):
        return sum(model.q_LG[l, g, t] for g in model.G) == model.Q[l, t]
    model.supply = Constraint(model.L, model.T, rule=supply_rule)

    # mill capacity
    def mill_cap(model, g, t):
        return sum(model.q_LG[l, g, t] for l in model.L) <= model.Qcap_g[g] * model.Yg[g]
    model.mill_cap = Constraint(model.G, model.T, rule=mill_cap)

    # mill -> warehouse flow (conversion)
    def mill_out(model, g, t):
        return sum(model.q_GW[g, w, t] for w in model.W) == sum(model.q_LG[l, g, t] for l in model.L) * params["conv"].get(g, 1.0)
    model.mill_out = Constraint(model.G, model.T, rule=mill_out)

    # warehouse capacity
    def wh_cap(model, w, t):
        return sum(model.q_GW[g, w, t] for g in model.G) <= model.Qcap_w[w] * model.Yw[w]
    model.wh_cap = Constraint(model.W, model.T, rule=wh_cap)

    # warehouse outflow
    def wh_out(model, w, t):
        return sum(model.q_WD[w, d, t] for d in model.D) == sum(model.q_GW[g, w, t] for g in model.G)
    model.wh_out = Constraint(model.W, model.T, rule=wh_out)

    # distributor outflow
    def d_out(model, d, t):
        return sum(model.q_DM[d, m, t] for m in model.M) == sum(model.q_WD[w, d, t] for w in model.W)
    model.d_out = Constraint(model.D, model.T, rule=d_out)

    # demand constraint
    def demand_rule(model, m, t):
        return sum(model.q_DM[d, m, t] for d in model.D) <= params["Demand"].get((m, t), 1e9)
    model.demand = Constraint(model.M, model.T, rule=demand_rule)

    # --------------------------
    # WATER BALANCE & DEFICIT
    # Water = rain - evap + irrigation
    # Gamma >= Wreq - Water
    # Gamma >= 0
    # --------------------------
    def water_balance(model, l, t):
        return model.Water[l, t] == params["rain"].get(t, 0.0) - params["evap"].get(t, 0.0) + model.irrig[l, t]
    model.water_balance = Constraint(model.L, model.T, rule=water_balance)

    def gamma_low(model, l, t):
        return model.Gamma[l, t] >= params["Wreq"][l] - model.Water[l, t]
    model.gamma_low = Constraint(model.L, model.T, rule=gamma_low)

    def gamma_nonneg(model, l, t):
        return model.Gamma[l, t] >= 0
    model.gamma_nonneg = Constraint(model.L, model.T, rule=gamma_nonneg)

    # irrigation upper bound (prevent unbounded irrigation)
    def irrig_limit(model, l, t):
        return model.irrig[l, t] <= 1000.0
    model.irrig_limit = Constraint(model.L, model.T, rule=irrig_limit)

    # --------------------------
    # COST EXPRESSIONS
    # --------------------------
    def input_cost(model):
        return sum(params["Cinput"].get(i, 1000.0) * model.x[i, l, t] for i in model.I for l in model.L for t in model.T)
    model.InputCost = Expression(rule=input_cost)

    def mill_cost(model):
        return sum(params["Cmill"].get(g, 0.0) * sum(model.q_GW[g, w, t] for w in model.W) for g in model.G for t in model.T)
    model.MillCost = Expression(rule=mill_cost)

    def store_cost(model):
        return sum(params["Cstore"].get(w, 0.0) * sum(model.q_GW[g, w, t] for g in model.G) for w in model.W for t in model.T)
    model.StoreCost = Expression(rule=store_cost)

    def transport_cost(model):
        tot = 0.0
        for (a, b), c in params["trans"].items():
            for t in model.T:
                # only sum cost for arcs that exist in params["trans"]
                if a in model.L and b in model.G:
                    tot += c * model.q_LG[a, b, t]
                if a in model.G and b in model.W:
                    tot += c * model.q_GW[a, b, t]
                if a in model.W and b in model.D:
                    tot += c * model.q_WD[a, b, t]
                if a in model.D and b in model.M:
                    tot += c * model.q_DM[a, b, t]
        return tot
    model.TransCost = Expression(rule=transport_cost)

    def fixed_cost(model):
        return sum(params.get("Fix_mill", {}).get(g, 0.0) * model.Yg[g] for g in model.G) + \
               sum(params.get("Fix_ware", {}).get(w, 0.0) * model.Yw[w] for w in model.W)
    model.FixedCost = Expression(rule=fixed_cost)

    def other_cost(model):
        irrigation_cost = sum(params.get("water_cost_per_unit", 50.0) * model.irrig[l, t] for l in model.L for t in model.T)
        labor_cost = sum(params.get("labor_cost_per_ton", 100.0) * model.HP[l, t] for l in model.L for t in model.T)
        return irrigation_cost + labor_cost
    model.OtherCost = Expression(rule=other_cost)

    # --------------------------
    # OBJECTIVE: maximize profit
    # --------------------------
    def obj_rule(model):
        total_rev = 0.0
        for t in model.T:
            Ht = H_param.get(t, params.get("H0", 0.0))
            total_rev += sum(Ht * model.Q[l, t] for l in model.L)
        market_rev = sum(params["Price_market"].get((m, t), 0.0) * model.q_DM[d, m, t] for d in model.D for m in model.M for t in model.T)
        total_cost = model.InputCost + model.MillCost + model.StoreCost + model.TransCost + model.FixedCost + model.OtherCost
        return total_rev + market_rev - total_cost
    model.obj = Objective(rule=obj_rule, sense=maximize)

    # input upper bounds
    def xmax_rule(model, i, l, t):
        return model.x[i, l, t] <= params["xmax"].get(i, 1e6)
    model.xmax_con = Constraint(model.I, model.L, model.T, rule=xmax_rule)

    # -------------------------------------------------------
    # PATCH: restrict flows to allowed arcs only (from params)
    # -------------------------------------------------------
    # L->G
    def restrict_LG_arcs(m, l, g, t):
        if (l, g) not in allowed_LG:
            return m.q_LG[l, g, t] == 0
        return Constraint.Skip
    model.restrict_LG = Constraint(model.L, model.G, model.T, rule=restrict_LG_arcs)

    # G->W
    def restrict_GW_arcs(m, g, w, t):
        if (g, w) not in allowed_GW:
            return m.q_GW[g, w, t] == 0
        return Constraint.Skip
    model.restrict_GW = Constraint(model.G, model.W, model.T, rule=restrict_GW_arcs)

    # W->D
    def restrict_WD_arcs(m, w, d, t):
        if (w, d) not in allowed_WD:
            return m.q_WD[w, d, t] == 0
        return Constraint.Skip
    model.restrict_WD = Constraint(model.W, model.D, model.T, rule=restrict_WD_arcs)

    # D->M
    def restrict_DM_arcs(m, d, mkt, t):
        if (d, mkt) not in allowed_DM:
            return m.q_DM[d, mkt, t] == 0
        return Constraint.Skip
    model.restrict_DM = Constraint(model.D, model.M, model.T, rule=restrict_DM_arcs)

    return model


def iterative_solve(params, write_iteration_log=True):
    """
    Iteratively solve the model updating H_t each iteration using:
      H_t = H0 + delta * K_avg + eta * HG_t
    Returns last model, params, final H_param, and iteration log list.
    If write_iteration_log=True, also writes iteration log to RESULT_FOLDER/iteration_log.csv
    """
    # initial H_t guess
    H_param = {t: params.get("H0", 0.0) + params.get("eta", 0.0) * params.get("HG", {}).get(t, 0.0) for t in params["T"]}

    solver = load_cbc_solver()

    iter_log = []
    last_obj = None

    for it in range(1, MAX_ITER + 1):
        print(f"\n--- Iteration {it} : H_t = {H_param}")

        # build model with current H_param
        model = build_pyomo_model(params, H_param)

        # solve
        res = solver.solve(model, tee=False)
        status = (res.solver.status, res.solver.termination_condition)
        print("Solver status:", res.solver.status, "termination:", res.solver.termination_condition)

        # If solver not optimal, warn and break (safe fallback)
        if (res.solver.status != SolverStatus.ok) or (res.solver.termination_condition != TerminationCondition.optimal):
            print("WARNING: solver did not return optimal solution. Stopping iterative updates.")
            # still attempt to extract available solution if any
            # but break the loop to avoid misleading updates
            break

        # compute K_avg based on kualitas_awal and chosen y-levels and Q
        K_avg = {}
        per_lahan_info = {}
        for t in params["T"]:
            total_Q = 0.0
            weighted_K = 0.0
            for l in params["L"]:
                # determine chosen k (b variables)
                chosen_k = None
                for k in range(len(params["S_levels"])):
                    try:
                        val = model.b[l, t, k].value
                    except Exception:
                        val = None
                    if val is None:
                        continue
                    if val > 0.5:
                        chosen_k = k
                        break
                if chosen_k is None:
                    # fallback: take k with highest b value
                    chosen_k = max(range(len(params["S_levels"])), key=lambda kk: (getattr(model.b[l,t,kk],'value',0.0) or 0.0))
                y = params["S_levels"][chosen_k]
                # kualitas = kualitas_awal * y
                kval = params["kualitas_awal"][l] * y
                Qval = getattr(model.Q[l, t], "value", 0.0) or 0.0
                total_Q += Qval
                weighted_K += kval * Qval
                # store per-lahan for diagnostics
                per_lahan_info.setdefault(l, {})[t] = {
                    "chosen_k": chosen_k,
                    "y": y,
                    "kualitas": kval,
                    "Q": Qval,
                    "HP": getattr(model.HP[l,t], "value", None),
                    "ProdExpr": getattr(model.Prod[l,t], "expr", None)
                }
            K_avg[t] = (weighted_K / total_Q) if total_Q > 0 else (sum(params.get("lambda_s", {}).values()) * np.mean(params["S_levels"]))
        # update H_t
        new_H = {}
        for t in params["T"]:
            new_H[t] = params.get("H0", 0.0) + params.get("delta", 0.0) * K_avg[t] + params.get("eta", 0.0) * params.get("HG", {}).get(t, 0.0)

        max_change = max(abs(new_H[t] - H_param[t]) for t in params["T"])
        cur_obj = value(model.obj)
        print(f"Updated H_t: {new_H} | max_change: {max_change:.6f} | objective: {cur_obj:.2f}")

        # append to iter_log
        iter_log.append({
            "iter": it,
            "H_param": H_param.copy(),
            "H_new": new_H.copy(),
            "max_change": max_change,
            "objective": cur_obj,
            "K_avg": K_avg.copy(),
            "per_lahan": per_lahan_info
        })

        # update for next iteration
        H_param = new_H

        # optional stopping by objective small change
        if last_obj is not None and abs(last_obj - cur_obj) < TOL_H:
            print("Objective change below tolerance; stopping.")
            break
        last_obj = cur_obj

        if max_change < TOL_H:
            print("H_t converged by threshold.")
            return model, params, H_param, iter_log

    print("Reached max iterations or stopped early.")
    return model, params, H_param, iter_log

# ============================
# WRITE RESULTS + MAIN()
# ============================

def save_results(model, params, H_final, iter_log):
    """
    Save outputs to /result folder:
      - parameter summary
      - iteration log
      - production & quality table
      - full supply chain flows
      - cost & profit summary
    """
    os.makedirs(RESULT_FOLDER, exist_ok=True)

    # ======================================
    # 1. SAVE PARAMETER SUMMARY
    # ======================================
    param_table = pd.DataFrame({
        "H0": [params["H0"]],
        "delta": [params["delta"]],
        "eta": [params["eta"]],
        "GlobalPrice(HG_1)": [params["HG"].get(1, None)],
        "Final_Converged_H_t": [H_final[1]],
        "Total Lahan": [len(params["L"])],
        "Total Penggilingan": [len(params["G"])],
        "Total Gudang": [len(params["W"])],
        "Total Distributor": [len(params["D"])],
        "Total Pasar": [len(params["M"])]
    })

    param_table.to_csv(f"{RESULT_FOLDER}/parameter_summary.csv", index=False)

    # ======================================
    # 2. SAVE ITERATION LOG
    # ======================================
    iter_rows = []
    for entry in iter_log:
        iter_rows.append({
            "iter": entry["iter"],
            "H_old": entry["H_param"][1],
            "H_new": entry["H_new"][1],
            "max_change": entry["max_change"],
            "objective": entry["objective"],
            "K_avg": entry["K_avg"][1]
        })
    pd.DataFrame(iter_rows).to_csv(f"{RESULT_FOLDER}/iteration_log.csv", index=False)

    # ======================================
    # 3. PRODUCTION & QUALITY TABLE
    # ======================================
    prod_rows = []
    for l in params["L"]:

        # expression -> use value()
        HP = value(model.HP[l, 1])
        Q = value(model.Q[l, 1])
        prod = value(model.Prod[l, 1])

        # cari level kualitas
        k_sel = None
        for k in range(len(params["S_levels"])):
            if value(model.b[l,1,k]) > 0.5:
                k_sel = k
                break
        if k_sel is None:
            k_sel = 0

        y = params["S_levels"][k_sel]
        kval = params["kualitas_awal"][l] * y

        prod_rows.append({
            "Lahan": l,
            "Luas_ha": params["A"][l],
            "Baseline_Prod": params["baseline_prod"][l],
            "Prod_Final": prod,
            "HP_ton": HP,
            "Quality_Level": y,
            "Final_Quality": kval,
            "Q_ton": Q
        })

    pd.DataFrame(prod_rows).to_csv(f"{RESULT_FOLDER}/production_quality.csv", index=False)


    # ======================================
    # 4. FLOW TABLES
    # ======================================
    flow_rows = []

    for l in params["L"]:
        for g in params["G"]:
            q = model.q_LG[l, g, 1].value
            if q > 0:
                flow_rows.append({"from": l, "to": g, "flow_ton": q})

    for g in params["G"]:
        for w in params["W"]:
            q = model.q_GW[g, w, 1].value
            if q > 0:
                flow_rows.append({"from": g, "to": w, "flow_ton": q})

    for w in params["W"]:
        for d in params["D"]:
            q = model.q_WD[w, d, 1].value
            if q > 0:
                flow_rows.append({"from": w, "to": d, "flow_ton": q})

    for d in params["D"]:
        for m in params["M"]:
            q = model.q_DM[d, m, 1].value
            if q > 0:
                flow_rows.append({"from": d, "to": m, "flow_ton": q})

    pd.DataFrame(flow_rows).to_csv(f"{RESULT_FOLDER}/supply_chain_flows.csv", index=False)

    # ======================================
    # 5. PROFIT SUMMARY
    # ======================================
    profit_rows = [{
        "Total_Profit": value(model.obj),
        "InputCost": value(model.InputCost),
        "MillCost": value(model.MillCost),
        "StoreCost": value(model.StoreCost),
        "TransportCost": value(model.TransCost),
        "FixedCost": value(model.FixedCost),
        "LaborCost": value(model.OtherCost)
    }]

    pd.DataFrame(profit_rows).to_csv(f"{RESULT_FOLDER}/profit_summary.csv", index=False)

    print("\n>>> All result tables have been saved to /result folder.\n")

# =====================================================
# MAIN FUNCTION
# =====================================================

def main():
    print("Loading data...")
    data = load_data(DATA_FOLDER)

    print("Preparing parameters...")
    params = prepare_params(data)

    print("Start iterative solve...")
    model, params, H_final, iter_log = iterative_solve(params)

    print("\n=== FINAL H_t ===")
    print(H_final)

    save_results(model, params, H_final, iter_log)

    print("\nSample L->G flows:")
    for l in params["L"]:
        for g in params["G"]:
            val = model.q_LG[l, g, 1].value
            print(f"{l} -> {g} : {val}")

if __name__ == "__main__":
    main()
