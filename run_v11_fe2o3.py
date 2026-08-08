#!/usr/bin/env python3
"""靶1 实证 v11：Fe2O3(0001) 相 H2O bridge 位弛豫（纯 Fe2O3 表面）"""
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
    g = GPAW(mode='pw', xc='PBE', h=0.25, txt=f'/tmp/target1/v11_{name}.txt',
             maxiter=200, mixer=Mixer(0.05, 5),
             spinpol=True, setups={'Fe': ':d,4.0'},
             occupations={'name': 'fermi-dirac', 'width': 0.05, 'fixmagmom': True},
             convergence={'energy': 2e-3, 'eigenstates': 1e-3, 'density': 1e-2},
             parallel={'gpu': True})
    return SumCalculator([g, DFTD3(method='PBE', damping='d3bj')])

def build_slab():
    slab = read('/tmp/target1/Fe2O3_slab0.cif')
    # Fe2O3 反铁磁（刚玉同构）：交替 ±4.0
    # a-Fe2O3 磁序：层内铁磁+层间反铁磁（按 z 层分组，相邻层反号）
    zs = sorted(set(round(s.position[2], 1) for s in slab if s.symbol == 'Fe'))
    layer_of = {z: k for k, z in enumerate(zs)}
    mm = [5.0 if s.symbol == 'Fe' and layer_of[round(s.position[2], 1)] % 2 == 0
          else -5.0 if s.symbol == 'Fe' else 0.0 for s in slab]
    slab.set_initial_magnetic_moments(mm)
    # 表面 Fe 层
    top_z = max(s.scaled_position[2] for s in slab)
    surf_fe = [i for i, s in enumerate(slab) if s.symbol == 'Fe' and abs(s.scaled_position[2] - top_z) < 0.15]
    # bridge 位：最近两个表面 Fe 中点
    p0 = slab[surf_fe[0]].position
    dmin, c1 = 1e9, None
    for i in surf_fe[1:]:
        d = np.linalg.norm(slab[i].position - p0)
        if d < dmin: dmin, c1 = d, slab[i].position
    bridge = (p0 + c1) / 2
    return slab, bridge, surf_fe

E_gas = -10.05149
slab, bridge, _ = build_slab()
# 1) 裸 slab（D3）
calc_b = make_calc('bare'); slab.calc = calc_b
E_bare = float(slab.get_potential_energy())
print(f"v11 裸 Fe2O3: {E_bare:.5f}", flush=True)
# 2) H2O@bridge 弛豫（冻结 slab 30 原子）
slab2, bridge2, _ = build_slab()
h2o = molecule('H2O'); h2o.rotate(90, 'x')
top_z2 = max(s.scaled_position[2] for s in slab2) * slab2.cell[2][2]
h2o.positions += [bridge2[0], bridge2[1], top_z2 + 2.3]
ads = slab2 + h2o
ads.set_constraint(FixAtoms(indices=range(30)))
calc_a = make_calc('ads'); ads.calc = calc_a
opt = BFGS(ads, logfile='/tmp/target1/v11_ads_relax.log')
opt.run(fmax=0.15, steps=40)
E_ads = float(ads.get_potential_energy())
e_ads = E_ads - E_bare - E_gas
print(f"v11 H2O@bridge: {E_ads:.5f} | E_ads={e_ads:.4f} eV | {opt.nsteps} 步", flush=True)
json.dump({'E_bare': E_bare, 'E_ads_total': E_ads, 'E_ads_eV': round(e_ads, 4),
           'nsteps': opt.nsteps}, open('/tmp/target1/results_v11.json', 'w'), indent=1)
print("DONE v11", flush=True)
