import numpy as np
from scipy import optimize


def vars_to_h_and_u(vars):
    K = int(len(vars) / 4)
    h = np.hstack([vars[:K][:,np.newaxis],vars[K:2*K][:,np.newaxis]])
    u = np.hstack([vars[2*K:3*K][:,np.newaxis],vars[3*K:4*K][:,np.newaxis]])
    return h, u


def portfolio_value(h0,
                 vars,
                 r_hat,
                 v_hat,
                 gamma,
                 rho):
    """
    Function purpose
    -------------------------------------
    Solves for a terminal portfolio value


    Parameters
    ----------
    h0 : float
        Starting value of asset in portfolio
    invested : float [0,1]
        Percent invested
    r_vec : list of floats
        Forecasted asset returns
    var_vec : list of floats
        Forecasted asset variances
    p_vec : list of floats
        Percentage of current portfolio invested
    com_rate : float
        Commission rate of executing a trade
    risk_para : float (>= 0)
        Risk aversion parameter
    trade_para : float (>=0)
        Trading aversion parameter
    """

    # Initialize portfolio values
    unit_vec = np.ones(h0.shape)
    h, u = vars_to_h_and_u(vars)

    risk_av = gamma*(h[:, 0] * v_hat * h[:, 0]) / np.where((h@unit_vec).squeeze()<=0,1,(h@unit_vec).squeeze())  # Risk aversion
    trade_av = np.abs(u)@rho  # Trade aversion

    return (unit_vec.T@h[-1])[0] - risk_av.sum() - trade_av.sum()


def find_optimal_trade(h0, r_vec, var_vec, com_rate, risk_para, trade_para):

    K = r_vec.shape[0] - 1
    ret = np.hstack([r_vec[:-1,:], np.zeros((K,1))])
    foo = lambda x: -portfolio_value_v2(h0, x, ret, var_vec[:-1], risk_para, trade_para)

    # Constraints on portfolio allocation (keep percentage allocated between 0 and 1)
    def self_financing(vars):
        h, u = vars_to_h_and_u(vars)
        return -(u @ np.ones((2, 1)) + np.abs(u) @ com_rate).squeeze()


    def post_trade(vars):
        h, u = vars_to_h_and_u(vars)
        h_t = h0
        diff = []
        for t in range(K-1):
            h_plus = (h_t + u[t, :][:, np.newaxis]) * (1 + ret[t, :][:, np.newaxis])
            diff.append(h_plus - h[t+1])
            h_t = h_plus
        return np.array(diff).flatten()

    cons = (
        {'type': 'ineq', 'fun': self_financing}, # self-financing constraint
        {'type': 'eq', 'fun': post_trade}  # portfolio value constraint
    )

    x0 = np.hstack([np.array([h0[0]]*K),np.array([h0[1]]*K),np.zeros((K,2))])
    res = optimize.minimize(foo, x0, method='SLSQP', constraints=cons) # , bounds=[(0,np.infty)]*2*K
    h_star, u_star = vars_to_h_and_u(res.x)
    print(h_star)
    print(u_star)

    return u_star[0][:, np.newaxis]


def port_value(h0,
                 invested,
                 r_vec,
                 var_vec,
                 p_vec,
                 com_rate,
                 risk_para,
                 trade_para):
    """
    Function purpose
    -------------------------------------
    Solves for a terminal portfolio value


    Parameters
    ----------
    h0 : float
        Starting value of asset in portfolio
    invested : float [0,1]
        Percent invested
    r_vec : list of floats
        Forecasted asset returns
    var_vec : list of floats
        Forecasted asset variances
    p_vec : list of floats
        Percentage of current portfolio invested
    com_rate : float
        Commission rate of executing a trade
    risk_para : float (>= 0)
        Risk aversion parameter
    trade_para : float (>=0)
        Trading aversion parameter
    """

    # Initialize portfolio values
    V = h0  # Total portfolio value
    asset_bal = V * invested  # Amount invested in asset
    cash_bal = V - asset_bal  # Amount invested in cash

    # Compute terminal portfolio value
    for i, r in enumerate(r_vec[:-1]):
        if i == 0:
            amount_chg = V * (p_vec[i] - invested)
        else:
            amount_chg = V * (p_vec[i] - p_vec[i - 1])

        asset_bal = (asset_bal + amount_chg) * (1 + r)  # Update invested amount, grow by next period return
        cash_bal -= amount_chg + np.abs(amount_chg) * com_rate  # Update cash balance (less cost of trade)
        V = asset_bal + cash_bal  # Update portfolio balance
        risk_av = risk_para * (p_vec[i] * var_vec[i] * p_vec[i]) / V  # Risk aversion
        trade_av = trade_para * np.abs(p_vec[i] - p_vec[i - 1])  # Trade aversion
        V -= risk_av + trade_av

    if isinstance(V,np.ndarray):
        return V[0]
    else:
        return V


def find_optimal_holdings(h0,
                 invested,
                 r_vec,
                 var_vec,
                 com_rate,
                 risk_para,
                 trade_para,
                 seed=None):
    # Constraints on portfolio allocation (keep percentage allocated between 0 and 1)

    K = r_vec.shape[0] - 1
    cons = ({'type': 'ineq', 'fun': lambda x: x - 1e-6},  # x[i] >= 0
            {'type': 'ineq', 'fun': lambda x: (1 - 1e-6) - x})  # x[i] <= 1
    bounds = [(0,1)]*K

    foo = lambda x: -port_value(h0,invested,r_vec.squeeze(),var_vec.squeeze(),x,com_rate,risk_para,trade_para)

    if seed is None:
        seed = np.random.uniform(size=K)

    prob = optimize.minimize(foo,
                             seed,
                             method='L-BFGS-B',
                             #constraints=cons,
                             bounds=bounds) #options={'maxiter':1000}

    return prob.x
