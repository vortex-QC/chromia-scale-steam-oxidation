#!/usr/bin/env python3
"""靶1 实证 v9：Fe 浓度梯度（2/12、3/12 表面 Cr→Fe）×（裸+吸附）"""
import json, time
import sys; sys.path.insert(0, '/home/vortex/涡肉身壳/papers/scf_v08')
import numpy as np
from ase.io import read
from ase.build import molecule
from ase.constraints import FixAtoms
from gpaw import GPAW
from gpaw.mixer import Mixer
from dftd3.ase import DFTD3
from ase.calculators.mixing import SumCalculator

def make_calc(name):
    g = GPAW(mode='pw', xc='PBE', h=0.25, txt=f'/tmp/target1/v9_{name}.txt',
             maxiter=150, mixer=Mixer(0.1, 5),
             spinpol=True, setups={'Cr': ':d,4.0', 'Fe': ':d,4.0'},
             occupations={'name': 'fermi-dirac', 'width': 0.05, 'fixmagmom': True},
             convergence={'energy': 2e-3, 'eigenstates': 1e-3, 'density': 1e-2},
             parallel={'gpu': True})
    return SumCalculator([g, DFTD3(method='PBE', damping='d3bj')])

def build_fe_slab(n_fe):
    slab = read('/tmp/target1/Cr2O3_slab0.cif')
    mm = [3.0 if s.symbol=='Cr' and i%2==0 else -3.0 if s.symbol=='Cr' else 0.0
          for i, s in enumerate(slab)]
    top_z = max(s.scaled_position[2] for s in slab)
    surf_cr = [i for i, s in enumerate(slab) if s.symbol=='Cr' and abs(s.scaled_position[2]-top_z) < 0.15]
    idxs = surf_cr[:n_fe]
    for idx in idxs:
        slab[idx].symbol = 'Fe'
        mm[idx] = 4.0 if mm[idx] > 0 else -4.0
    slab.set_initial_magnetic_moments(mm)
    # Fe 邻位 bridge：第一个 Fe 与最近表面 Cr 中点
    fe_pos = slab[idxs[0]].position
    others = [slab[i].position for i in surf_cr if i not in idxs]
    if not others:
        others = [s.position for s in slab if s.symbol in ('Cr','Fe') and np.linalg.norm(s.position - fe_pos) > 0.1]
    dmin, c1 = 1e9, None
    for c in others:
        d = np.linalg.norm(c - fe_pos)
        if d < dmin: dmin, c1 = d, c
    bridge = (fe_pos + c1)/2
    return slab, bridge

E_gas = -10.05149
res = {}
for n_fe in [2, 3]:
    slab, bridge = build_fe_slab(n_fe)
    tag = f'fe{n_fe}'
    calc_b = make_calc(f'{tag}_bare'); slab.calc = calc_b
    E_bare = float(slab.get_potential_energy())
    print(f"{tag} 裸: {E_bare:.5f}", flush=True)
    slab2, _ = build_fe_slab(n_fe)
    h2o = molecule('H2O'); h2o.rotate(90, 'x')
    top_z2 = max(s.scaled_position[2] for s in slab2) * slab2.cell[2][2]
    h2o.positions += [bridge[0], bridge[1], top_z2 + 2.3]
    ads = slab2 + h2o
    ads.set_constraint(FixAtoms(indices=range(30)))
    calc_a = make_calc(f'{tag}_ads'); ads.calc = calc_a
    E_ads = float(ads.get_potential_energy())
    print(f"{tag} H2O@bridge: {E_ads:.5f}", flush=True)
    res[tag] = {'E_bare': E_bare, 'E_ads_total': E_ads,
                'E_ads_eV': round(E_ads - E_bare - E_gas, 4)}
json.dump(res, open('/tmp/target1/results_v9.json','w'), indent=1)
print("DONE v9", flush=True)
