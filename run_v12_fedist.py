#!/usr/bin/env python3
"""靶1 实证 v12：Fe 2/12 非相邻替换位（v10 相邻位发散，换最远对）H2O bridge 弛豫"""
import json, sys
sys.path.insert(0, '/home/vortex/涡肉身壳/papers/scf_v08')
import numpy as np
from ase.io import read
from ase.build import molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from gpaw import GPAW
from gpaw.mixer import Mixer
from dftd3.ase import DFTD3
from ase.calculators.mixing import SumCalculator

def make_calc(name):
    g = GPAW(mode='pw', xc='PBE', h=0.25, txt=f'/tmp/target1/v12_{name}.txt',
             maxiter=150, mixer=Mixer(0.1, 5),
             spinpol=True, setups={'Cr': ':d,4.0', 'Fe': ':d,4.0'},
             occupations={'name': 'fermi-dirac', 'width': 0.05, 'fixmagmom': True},
             convergence={'energy': 2e-3, 'eigenstates': 1e-3, 'density': 1e-2},
             parallel={'gpu': True})
    return SumCalculator([g, DFTD3(method='PBE', damping='d3bj')])

def build_slab():
    slab = read('/tmp/target1/Cr2O3_slab0.cif')
    mm = [3.0 if s.symbol == 'Cr' and i % 2 == 0 else -3.0 if s.symbol == 'Cr' else 0.0
          for i, s in enumerate(slab)]
    top_z = max(s.scaled_position[2] for s in slab)
    surf_cr = [i for i, s in enumerate(slab) if s.symbol == 'Cr' and abs(s.scaled_position[2] - top_z) < 0.15]
    # 非相邻位：表面 Cr 中距离最远的一对
    dmax, pair = -1, (surf_cr[0], surf_cr[1])
    for a in range(len(surf_cr)):
        for b in range(a + 1, len(surf_cr)):
            d = np.linalg.norm(slab[surf_cr[a]].position - slab[surf_cr[b]].position)
            if d > dmax: dmax, pair = d, (surf_cr[a], surf_cr[b])
    idxs = list(pair)
    for idx in idxs:
        slab[idx].symbol = 'Fe'
        mm[idx] = 4.0 if mm[idx] > 0 else -4.0
    slab.set_initial_magnetic_moments(mm)
    # bridge：最近两 Fe（含替换位）中点
    fe_pos = slab[idxs[0]].position
    others = [slab[i].position for i in surf_cr if i not in idxs]
    dmin, c1 = 1e9, None
    for c in others:
        d = np.linalg.norm(c - fe_pos)
        if d < dmin: dmin, c1 = d, c
    bridge = (fe_pos + c1) / 2
    return slab, bridge

E_gas = -10.05149
slab, bridge = build_slab()
calc_b = make_calc('bare'); slab.calc = calc_b
E_bare = float(slab.get_potential_energy())
print(f"v12 裸 Fe2/12 非相邻: {E_bare:.5f}", flush=True)
slab2, bridge2 = build_slab()
h2o = molecule('H2O'); h2o.rotate(90, 'x')
top_z2 = max(s.scaled_position[2] for s in slab2) * slab2.cell[2][2]
h2o.positions += [bridge2[0], bridge2[1], top_z2 + 2.3]
ads = slab2 + h2o
ads.set_constraint(FixAtoms(indices=range(30)))
calc_a = make_calc('ads'); ads.calc = calc_a
opt = BFGS(ads, logfile='/tmp/target1/v12_ads_relax.log')
opt.run(fmax=0.15, steps=40)
E_ads = float(ads.get_potential_energy())
e_ads = E_ads - E_bare - E_gas
print(f"v12 H2O@bridge: {E_ads:.5f} | E_ads={e_ads:.4f} eV | {opt.nsteps} 步", flush=True)
json.dump({'E_bare': E_bare, 'E_ads_total': E_ads, 'E_ads_eV': round(e_ads, 4),
           'nsteps': opt.nsteps, 'fe_pair': [int(x) for x in idxs]},
          open('/tmp/target1/results_v12.json', 'w'), indent=1)
print("DONE v12", flush=True)
