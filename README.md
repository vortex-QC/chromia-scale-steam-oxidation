# Alloying-element effects on chromia-scale water activation (steam oxidation)

DFT benchmark of the Fe concentration gradient in water activation on Cr₂O₃(0001) surfaces — the first elementary step of steam oxidation of 9-12%Cr turbine steels.

**Preprint**: DOI 10.5281/zenodo.21847151 (EN) / 10.5281/zenodo.[CN] (CN)

## Summary

- **Gap**: OQMD/Materials Project cover bulk oxide thermodynamics but have zero coverage of steam-oxidation layers (off-stoichiometric alloys / steam-surface / scale-alloy interface); literature DFT is limited to isolated single systems.
- **Baseline**: H₂O adsorption on Cr₂O₃(0001) — molecular adsorption favored (bridge site E_ads = -0.29 eV) over dissociation (near zero).
- **Fe concentration gradient** (relaxed, bridge site, PBE+U+D3(BJ), GPAW GPU):

| System | E_ads (eV) | Δ vs. pure |
|---|---|---|
| Pure Cr₂O₃ | -0.198 | baseline |
| Fe 1/12 substitutional | -0.369 | +0.171 |
| Fe 2/12 non-adjacent | -0.483 | +0.285 |
| Fe₂O₃(0001) phase | -0.232 | +0.034 |

**Key finding**: enhancement is strongest in the *doped transition state* (+0.17 to +0.29 eV, isolated substitutional Fe in the chromia lattice) and mild in the complete hematite phase (+0.03 eV) — assigning the engineering observation of poor protection at Fe-enriched zones to the early enrichment regime.

## Structure

```
scripts/   run_v4.py ... run_v12_fedist.py   (full calculation chain, v4→v12)
data/      results_v4/6/7/8/9/10/11/12.json  (energies, E_ads, relaxation steps)
data/      Cr2O3_bulk.cif, Cr2O3_slab0.cif, Fe2O3_slab0.cif
```

## Environment

- GPAW 25.7 (GPU/CuPy build), Python 3.10, ASE 3.29, DFTD3 via `dftd3.ase`
- GPU environment: `source gpu_env.sh` (LD_LIBRARY_PATH for CUDA 12.1 + nvidia pip libs; `GPAW_NEW=1`)

## Reproduction notes (critical)

1. **Spin-polarized convergence (general solution)**: oxide surfaces with AFM order require (i) `fixmagmom` fixed magnetic moments, (ii) explicit `eigenstates: 1e-3` relaxation (default 4e-8 never converges), (iii) correct magnetic order — **Cr₂O₃: ±3.0 μB alternating; α-Fe₂O₃: layer-wise FM + interlayer AFM, ±5.0 μB** (Fe³⁺ high-spin d⁵). A naive i%2 alternation assigns anti-parallel moments within the same layer and diverges (v11 history: 4 failed attempts).
2. **Energy protocol**: all E_ads use the *same* relaxed protocol; single-point E_ads are NOT transferable (v8 single-point "Fe weakens adsorption" was falsified by relaxation — the relaxed result is Fe *enhances* by 0.17 eV).
3. **Fe 2/12**: only non-adjacent substitution converges; adjacent Fe-Fe configurations diverge (unstable geometry).
4. **H₂O gas reference**: E_gas = -10.05149 eV (large-cell single molecule, 17 steps, fully converged) — do not reuse other protocols' gas energy.

## Results cross-check

| System | script | E_bare (eV) | E_ads,total (eV) | E_ads (eV) | steps |
|---|---|---|---|---|---|
| pure Cr₂O₃ | run_v10_relax.py | -173.99032 | — | -0.198 | 40 |
| Fe 1/12 | run_v10_relax.py | -171.15024 | — | -0.369 | 21 |
| Fe 2/12 non-adj | run_v12_fedist.py | -170.12852 | -180.66247 | -0.483 | 40 |
| Fe₂O₃ phase | run_v11_fe2o3.py | -144.64388 | -154.92685 | -0.232 | 40 |

---
*Qin Chao | ORCID 0009-0006-2000-5644 | 2026-08-08*
