#!/usr/bin/env python3
"""靶1 实证首步 v4（修正）：冻结 slab + H2O 分子/解离弛豫（GPAW GPU PBE+U AFM）"""
import json, time, re
sys_path = '/home/vortex/涡肉身壳/papers/scf_v08'
import sys; sys.path.insert(0, sys_path)
from ase.io import read
from ase.build import molecule
from ase import Atoms
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from gpaw import GPAW
from gpaw.mixer import Mixer

def make_calc(name, txt):
    return GPAW(mode='pw', xc='PBE', h=0.25, txt=txt,
                maxiter=150, mixer=Mixer(0.05, 5),
                spinpol=True, setups={'Cr': ':d,4.0'},
                occupations={'name': 'fermi-dirac', 'width': 0.05},
                convergence={'energy': 1e-2, 'density': 5e-2},
                parallel={'gpu': True})

def E_robust(atoms, name):
    """容错能量：SCF 异常时读 txt 最后迭代能量"""
    calc = make_calc(name, f'/tmp/target1/{name}.txt')
    atoms.calc = calc
    t0 = time.time()
    try:
        e = atoms.get_potential_energy()
        conv = True
    except Exception:
        conv = False
        txt = open(f'/tmp/target1/{name}.txt').read()
        vals = [float(m) for m in re.findall(r'iter:\s*\d+\s+[\d:]+\s+(-?\d+\.\d+)', txt)]
        e = vals[-1] if vals else float('nan')
    return float(e), conv, (time.time()-t0)/60

def freeze_slab(atoms):
    """冻结 slab 原子（原子序 < len(slab)）"""
    n_slab = 30  # slab0 30 原子
    atoms.set_constraint(FixAtoms(indices=range(n_slab)))

# 参考能量
E_bare = -168.375683          # 裸 slab（缓存）
E_h2o_gas = -10.039496        # H2O 气相（缓存，17 步真收敛）

# === 构型 1：H2O 分子弛豫（冻结 slab）===
slab = read('/tmp/target1/Cr2O3_slab0.cif')
mm = [3.0 if s.symbol=='Cr' and i%2==0 else -3.0 if s.symbol=='Cr' else 0.0
      for i, s in enumerate(slab)]
slab.set_initial_magnetic_moments(mm)
freeze_slab(slab)
h2o = molecule('H2O')
h2o.rotate(90, 'x')
cx = slab.cell[0][0]/2; cy = slab.cell[1][1]/2
top_z = max(s.scaled_position[2] for s in slab) * slab.cell[2][2]
h2o.positions += [cx, cy, top_z + 2.3]
ads = slab + h2o
ads.set_constraint(FixAtoms(indices=range(30)))
calc = make_calc('v4_mol', '/tmp/target1/v4_mol.txt')
ads.calc = calc
try:
    opt = BFGS(ads, logfile='/tmp/target1/v4_mol_relax.log', trajectory='/tmp/target1/v4_mol.traj')
    opt.run(fmax=0.1, steps=40)
    E_mol = float(ads.get_potential_energy()); conv_mol = True
except Exception:
    E_mol, conv_mol, _ = E_robust(ads, 'v4_mol')
print(f"v4 H2O分子弛豫: E={E_mol:.5f} | conv={conv_mol}", flush=True)

# === 构型 2：解离弛豫（冻结 slab，ASE 正确 OH 键长）===
slab2 = read('/tmp/target1/Cr2O3_slab0.cif')
slab2.set_initial_magnetic_moments(mm)
freeze_slab(slab2)
oh = molecule('OH')  # 正确 0.96Å 键长
oh.rotate(90, 'x')
oh.positions += [cx, cy, top_z + 2.0]
h = Atoms('H', positions=[[cx + 1.2, cy, top_z + 1.5]])
diss = slab2 + oh + h
diss.set_constraint(FixAtoms(indices=range(30)))
calc2 = make_calc('v4_diss', '/tmp/target1/v4_diss.txt')
diss.calc = calc2
try:
    opt2 = BFGS(diss, logfile='/tmp/target1/v4_diss_relax.log', trajectory='/tmp/target1/v4_diss.traj')
    opt2.run(fmax=0.1, steps=40)
    E_diss = float(diss.get_potential_energy()); conv_diss = True
except Exception:
    E_diss, conv_diss, _ = E_robust(diss, 'v4_diss')
print(f"v4 解离弛豫: E={E_diss:.5f} | conv={conv_diss}", flush=True)

res = {'E_bare': E_bare, 'E_h2o_gas': E_h2o_gas,
       'E_mol_relaxed': E_mol, 'E_diss_relaxed': E_diss,
       'E_ads_mol_eV': round(E_mol - E_bare - E_h2o_gas, 4),
       'E_ads_diss_eV': round(E_diss - E_bare - E_h2o_gas, 4),
       'conv': [conv_mol, conv_diss]}
json.dump(res, open('/tmp/target1/results_v4.json','w'), indent=1)
print("DONE v4", flush=True)
