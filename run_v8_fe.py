#!/usr/bin/env python3
"""靶1 实证 v8：Fe 掺杂元素效应——Cr2O3(0001) 表面 Cr→Fe 替换，bridge 位 H2O 吸附对比"""
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
    g = GPAW(mode='pw', xc='PBE', h=0.25, txt=f'/tmp/target1/v8_{name}.txt',
             maxiter=150, mixer=Mixer(0.1, 5),
             spinpol=True, setups={'Cr': ':d,4.0', 'Fe': ':d,4.0'},
             occupations={'name': 'fermi-dirac', 'width': 0.05, 'fixmagmom': True},
             convergence={'energy': 2e-3, 'eigenstates': 1e-3, 'density': 1e-2},
             parallel={'gpu': True})
    return SumCalculator([g, DFTD3(method='PBE', damping='d3bj')])

def mm_for(slab):
    return [3.0 if s.symbol=='Cr' and i%2==0 else -3.0 if s.symbol=='Cr' else 0.0
            for i, s in enumerate(slab)]

# Fe 掺杂 slab：表面层第一个 Cr → Fe（磁矩 +4，随原 Cr 层符号）
slab = read('/tmp/target1/Cr2O3_slab0.cif')
mm = mm_for(slab)
top_z = max(s.scaled_position[2] for s in slab)
surf_cr_idx = [i for i, s in enumerate(slab) if s.symbol=='Cr' and abs(s.scaled_position[2]-top_z) < 0.15]
idx_fe = surf_cr_idx[0]
slab[idx_fe].symbol = 'Fe'
mm[idx_fe] = 4.0 if mm[idx_fe] > 0 else -4.0
slab.set_initial_magnetic_moments(mm)
print(f"Fe 掺杂位: 原子{idx_fe} (原 AFM 符号 {'+' if mm[idx_fe]>0 else '-'})", flush=True)

# 参考：Fe 掺杂裸 slab
calc_b = make_calc('fe_bare'); slab.calc = calc_b
E_bare_fe = float(slab.get_potential_energy())
print(f"Fe-slab 裸+D3: {E_bare_fe:.5f}", flush=True)

# Fe 邻位 bridge：Fe 与最近表面 Cr 中点
fe_pos = slab[idx_fe].position
surf_cr2 = [slab[i].position for i in surf_cr_idx if i != idx_fe]
dmin, c1 = 1e9, None
for c in surf_cr2:
    d = np.linalg.norm(c - fe_pos)
    if d < dmin: dmin, c1 = d, c
bridge_fe = (fe_pos + c1)/2

# H2O@bridge(Fe 邻位) 单点（冻结 slab）
slab2 = read('/tmp/target1/Cr2O3_slab0.cif')
mm2 = mm_for(slab2)
slab2[idx_fe].symbol = 'Fe'
mm2[idx_fe] = 4.0 if mm2[idx_fe] > 0 else -4.0
slab2.set_initial_magnetic_moments(mm2)
h2o = molecule('H2O'); h2o.rotate(90, 'x')
top_z2 = max(s.scaled_position[2] for s in slab2) * slab2.cell[2][2]
h2o.positions += [bridge_fe[0], bridge_fe[1], top_z2 + 2.3]
ads = slab2 + h2o
ads.set_constraint(FixAtoms(indices=range(30)))
calc_a = make_calc('fe_ads'); ads.calc = calc_a
E_ads_fe = float(ads.get_potential_energy())
print(f"Fe-slab H2O@bridge: {E_ads_fe:.5f}", flush=True)

E_gas = -10.05149  # v7 气相+D3 缓存
res = {'E_bare_fe': E_bare_fe, 'E_ads_fe_total': E_ads_fe, 'E_h2o_gas_d3': E_gas,
       'E_ads_fe_bridge_eV': round(E_ads_fe - E_bare_fe - E_gas, 4),
       'ref_pure_bridge_eV': -0.2942}  # v7 纯 Cr2O3 bridge 位
json.dump(res, open('/tmp/target1/results_v8.json','w'), indent=1)
print("DONE v8", flush=True)
