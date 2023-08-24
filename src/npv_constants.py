"""
This file has been created for initializing the constants dictionary of the NPV formula.

    constants (dict):
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
