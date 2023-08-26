"""
    This file has been created for initializing the constants dictionary of the NPV formula.
    
    The NPV formula is given by:

    \[
    NPV = \\sum_{t=1}^{T} \\frac{ (Q_o * r_o) - (Q_w * r_wp) + (Q_g * r_gp) - OPEX }{(1 + d)^t} - CAPEX
    \]
    
    where:
    - t: time (annular)
    - $Q_o$: Total Oil Field Flow Rate (production)
    - $Q_w$: Totla Water Field Flow Rate (production)
    - $Q_g$: TOtal Gas Field Flow Rate (production)


    constants to define (dict):
        ro: oil price ($/bbl)
        rgp: gas price ($/bbl)
        rwp: water production cost ($/bbl)
        d: annual discount rate (0<d<1)
        opex: operational expenditure ($)
        capex: capital expenditure ($)

"""
constants = {
    'ro': 100,
    'rgp': 0.4,
    'rwp': 5,
    'd': 0.1,
    'opex': 5000000,
    'capex': 70000000,
}
