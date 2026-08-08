#!/usr/bin/env python3
"""靶1 实证 v7：位点扫描（top-Cr/top-O/bridge）×H2O 分子 + D3 色散修正"""
import json, time
import sys; sys.path.insert(0, '/home/vortex/涡肉身壳/papers/scf_v08')
import numpy as np
from ase.io import read
from ase.build import molecule
from ase import Atoms
from ase.constraints import FixAtoms
from gpaw import GPAW
from gpaw.mixer import Mixer
from dftd3.ase import DFTD3
from ase.calculators.mixing import SumCalculator

def make_calc(name):
    g = GPAW(mode='pw', xc='PBE', h=0.25, txt=f'/tmp/target1/v7_{name}.txt',
             maxiter=150, mixer=Mixer(0.1, 5),
             spinpol=True, setups={'Cr': ':d,4.0'},
             occupations={'name': 'fermi-dirac', 'width': 0.05, 'fixmagmom': True},
             convergence={'energy': 2e-3, 'eigenstates': 1e-3, 'density': 1e-2},
             parallel={'gpu': True})
    return SumCalculator([g, DFTD3(method='PBE', damping='d3bj')])

def mm_for(slab):
    return [3.0 if s.symbol=='Cr' and i%2==0 else -3.0 if s.symbol=='Cr' else 0.0
            for i, s in enumerate(slab)]

def site_xy(slab):
    """表面位点：top-Cr（最高 Cr 的 xy）、top-O（次层 O 投影）、bridge（两最近表面 Cr 中点）"""
    top_z = max(s.scaled_position[2] for s in slab)
    surf_cr = [s.position for s in slab if s.symbol=='Cr' and abs(s.scaled_position[2]-top_z) < 0.15]
    c0 = np.array(surf_cr[0])
    # bridge: 最近的另一个表面 Cr
    dmin, c1 = 1e9, None
    for c in surf_cr[1:]:
        d = np.linalg.norm(np.array(c)-c0)
        if d < dmin: dmin, c1 = d, np.array(c)
    bridge = (c0 + c1)/2
    # top-O: 找一个 O 的 xy 投影
    o0 = next(s.position for s in slab if s.symbol=='O')
    return {'top_Cr': c0, 'top_O': o0, 'bridge': bridge}

slab = read('/tmp/target1/Cr2O3_slab0.cif')
slab.set_initial_magnetic_moments(mm_for(slab))
sites = site_xy(slab)
top_z = max(s.scaled_position[2] for s in slab) * slab.cell[2][2]

# 参考：裸 slab + H2O 气相（PBE+D3）
calc_b = make_calc('bare_d3'); slab.calc = calc_b
E_bare = float(slab.get_potential_energy())
print(f"裸 slab+D3: {E_bare:.5f}", flush=True)
h2o_gas = molecule('H2O'); h2o_gas.center(vacuum=8.0)
calc_g = make_calc('gas_d3'); h2o_gas.calc = calc_g
E_gas = float(h2o_gas.get_potential_energy())
print(f"H2O 气相+D3: {E_gas:.5f}", flush=True)

# 3 位点 H2O 单点（冻结 slab）
results = {}
for sname, sxy in sites.items():
    slab_i = read('/tmp/target1/Cr2O3_slab0.cif')
    slab_i.set_initial_magnetic_moments(mm_for(slab_i))
    h2o = molecule('H2O'); h2o.rotate(90, 'x')
    h2o.positions += [sxy[0], sxy[1], top_z + 2.3]
    ads = slab_i + h2o
    ads.set_constraint(FixAtoms(indices=range(30)))
    calc_i = make_calc(f'site_{sname}')
    ads.calc = calc_i
    try:
        e = float(ads.get_potential_energy())
        results[sname] = {'E': e, 'E_ads': round(e - E_bare - E_gas, 4)}
        print(f"{sname}: E={e:.5f} E_ads={results[sname]['E_ads']:.4f} eV", flush=True)
    except Exception as ex:
        results[sname] = {'error': str(ex)[:80]}
        print(f"{sname}: 失败 {str(ex)[:60]}", flush=True)

res = {'E_bare': E_bare, 'E_gas': E_gas, 'sites': results}
json.dump(res, open('/tmp/target1/results_v7.json','w'), indent=1)
print("DONE v7", flush=True)
