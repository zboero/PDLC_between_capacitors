import importlib
import types
import sys
import numpy as np

# Ensure module can be imported even if numba is missing (handled in module)
ef = importlib.import_module('E_field_parallel')

# configure a small grid
ef.Lx = 4
ef.Ly = 4
ef.Ly2 = ef.Ly // 2
ef.dx = 1.0
ef.dy_cell = 1.0
ef.dy_glass = 1.0

# simple potentials and material properties
ef.V_ITO = 0.0
ef.V_fingA = 1.0
ef.V_fingB = -1.0
ef.eps_polym = 1.0
ef.eps_glass = 1.0

# electrode geometry
ef.Lb2La = 1.0
ef.la = int(ef.Lx / (1 + ef.Lb2La))
ef.lb = int(ef.la * ef.Lb2La)
ef.la2 = int(ef.la / 2)

Phi0 = np.zeros((ef.Ly + 1, ef.Lx + 1))
Phi, l2norm, ite = ef.relaxation(Phi0, omega=1.5, l2_target=1e-4, maxiter=100)

def test_l2norm_decreases_and_boundaries():
    assert l2norm < 1e-4, f"l2norm did not decrease enough: {l2norm}"
    assert np.all(Phi[-1, :] == ef.V_ITO)
    assert np.all(Phi[ef.Ly2, :ef.la2] == ef.V_fingA)
    assert np.all(Phi[ef.Ly2, (ef.la2 + ef.lb):] == ef.V_fingB)
