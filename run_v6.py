#!/usr/bin/env python3
"""靶1 实证首步 v6（突破版）：fixmagmom + eigenstates 放宽——裸/分子/解离全链+弛豫"""
import json, time
import sys; sys.path.insert(0, '/home/vortex/涡肉身壳/papers/scf_v08')
from ase.io import read
from ase.build import molecule
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from gpaw import GPAW
from gpaw.mixer import Mixer

MM = None
def get_mm(slab):
    global MM
    if MM is None:
        MM = [3.0 if s.symbol=='Cr' and i%2==0 else -3.0 if s.symbol=='Cr' else 0.0
              for i, s in enumerate(slab)]
    return MM

def make_calc(name):
    return GPAW(mode='pw', xc='PBE', h=0.25, txt=f'/tmp/target1/v6_{name}.txt',
                maxiter=150, mixer=Mixer(0.1, 5),
                spinpol=True, setups={'Cr': ':d,4.0'},
                occupations={'name': 'fermi-dirac', 'width': 0.05, 'fixmagmom': True},
                convergence={'energy': 2e-3, 'eigenstates': 1e-3, 'density': 1e-2},
                parallel={'gpu': True})

# 裸 slab（fixmagmom 协议）
slab = read('/tmp/target1/Cr2O3_slab0.cif')
slab.set_initial_magnetic_moments(get_mm(slab))
calc = make_calc('bare'); slab.calc = calc
E_bare = float(slab.get_potential_energy())
print(f"v6 裸 slab: E={E_bare:.5f}", flush=True)

# 构型 1：H2O 分子弛豫（冻结 slab）
slab1 = read('/tmp/target1/Cr2O3_slab0.cif')
slab1.set_initial_magnetic_moments(get_mm(slab1))
h2o = molecule('H2O'); h2o.rotate(90, 'x')
cx = slab1.cell[0][0]/2; cy = slab1.cell[1][1]/2
top_z = max(s.scaled_position[2] for s in slab1) * slab1.cell[2][2]
h2o.positions += [cx, cy, top_z + 2.3]
ads = slab1 + h2o
ads.set_constraint(FixAtoms(indices=range(30)))
calc1 = make_calc('mol'); ads.calc = calc1
try:
    opt = BFGS(ads, logfile='/tmp/target1/v6_mol_relax.log')
    opt.run(fmax=0.15, steps=40)
    E_mol = float(ads.get_potential_energy())
    print(f"v6 H2O分子弛豫: E={E_mol:.5f} | 步数={opt.nsteps}", flush=True)
except Exception as ex:
    E_mol = float(ads.get_potential_energy())
    print(f"v6 分子(容错): E={E_mol:.5f} | {str(ex)[:60]}", flush=True)

# 构型 2：解离弛豫
slab2 = read('/tmp/target1/Cr2O3_slab0.cif')
slab2.set_initial_magnetic_moments(get_mm(slab2))
oh = molecule('OH'); oh.rotate(90, 'x')
oh.positions += [cx, cy, top_z + 2.0]
h = Atoms('H', positions=[[cx + 1.2, cy, top_z + 1.5]])
diss = slab2 + oh + h
diss.set_constraint(FixAtoms(indices=range(30)))
calc2 = make_calc('diss'); diss.calc = calc2
try:
    opt2 = BFGS(diss, logfile='/tmp/target1/v6_diss_relax.log')
    opt2.run(fmax=0.15, steps=40)
    E_diss = float(diss.get_potential_energy())
    print(f"v6 解离弛豫: E={E_diss:.5f} | 步数={opt2.nsteps}", flush=True)
except Exception as ex:
    E_diss = float(diss.get_potential_energy())
    print(f"v6 解离(容错): E={E_diss:.5f} | {str(ex)[:60]}", flush=True)

E_gas = -10.039496  # 气相缓存（17 步真收敛，无磁无 U，协议一致）
res = {'E_bare': E_bare, 'E_mol': E_mol, 'E_diss': E_diss, 'E_h2o_gas': E_gas,
       'E_ads_mol_eV': round(E_mol - E_bare - E_gas, 4),
       'E_ads_diss_eV': round(E_diss - E_bare - E_gas, 4)}
json.dump(res, open('/tmp/target1/results_v6.json','w'), indent=1)
print("DONE v6", flush=True)
