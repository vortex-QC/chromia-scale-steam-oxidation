#!/usr/bin/env python3
"""靶1 实证 v10：bridge 位 H2O 弛豫三体系（纯 Cr2O3 / Fe 1/12 / Fe 2/12）"""
import json, time
import sys; sys.path.insert(0, '/home/vortex/涡肉身壳/papers/scf_v08')
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
    g = GPAW(mode='pw', xc='PBE', h=0.25, txt=f'/tmp/target1/v10_{name}.txt',
             maxiter=150, mixer=Mixer(0.1, 5),
             spinpol=True, setups={'Cr': ':d,4.0', 'Fe': ':d,4.0'},
             occupations={'name': 'fermi-dirac', 'width': 0.05, 'fixmagmom': True},
             convergence={'energy': 2e-3, 'eigenstates': 1e-3, 'density': 1e-2},
             parallel={'gpu': True})
    return SumCalculator([g, DFTD3(method='PBE', damping='d3bj')])

def build_slab(n_fe):
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
    fe_pos = slab[idxs[0]].position if idxs else slab[surf_cr[0]].position
    others = [slab[i].position for i in surf_cr if i not in idxs]
    if not others:
        others = [s.position for s in slab if s.symbol=='Cr' and np.linalg.norm(s.position - fe_pos) > 0.1]
    dmin, c1 = 1e9, None
    for c in others:
        d = np.linalg.norm(c - fe_pos)
        if d < dmin: dmin, c1 = d, c
    bridge = (fe_pos + c1)/2
    return slab, bridge

def relax(slab, bridge, name):
    h2o = molecule('H2O'); h2o.rotate(90, 'x')
    top_z = max(s.scaled_position[2] for s in slab) * slab.cell[2][2]
    h2o.positions += [bridge[0], bridge[1], top_z + 2.3]
    ads = slab + h2o
    ads.set_constraint(FixAtoms(indices=range(30)))
    calc = make_calc(name); ads.calc = calc
    try:
        opt = BFGS(ads, logfile=f'/tmp/target1/v10_{name}_relax.log')
        opt.run(fmax=0.15, steps=40)
        return float(ads.get_potential_energy()), opt.nsteps, True
    except Exception as ex:
        return float(ads.get_potential_energy()), 0, False

E_gas = -10.05149
res = {}
for n_fe, tag in [(0, 'pure'), (1, 'fe1'), (2, 'fe2')]:
    slab, bridge = build_slab(n_fe)
    E_ads, nsteps, ok = relax(slab, bridge, tag)
    E_bare = {'pure': -173.99032, 'fe1': -171.15024, 'fe2': -170.11152}[tag]  # 缓存裸 slab+D3
    e_ads = E_ads - E_bare - E_gas
    res[tag] = {'E_ads_total': E_ads, 'E_ads_eV': round(e_ads, 4), 'nsteps': nsteps, 'ok': ok}
    print(f"{tag}: E_ads={e_ads:.4f} eV | {nsteps} 步 | ok={ok}", flush=True)
json.dump(res, open('/tmp/target1/results_v10.json','w'), indent=1)
print("DONE v10", flush=True)
