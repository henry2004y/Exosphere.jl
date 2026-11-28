# Exosphere.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://henry2004y.github.io/Exosphere.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://henry2004y.github.io/Exosphere.jl/dev)
[![CI](https://github.com/henry2004y/Exosphere.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/henry2004y/Exosphere.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/henry2004y/Exosphere.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/henry2004y/Exosphere.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Standalone, zero-allocation planetary neutral exosphere models, `PlasmaPopulations.jl`-style neutral species and VDFs, EUV shadow / optical-depth / Chapman / ray attenuation, conservative one-way and two-way charge–neutral interaction kernels, structured neutral field containers, and adaptive dual-cell box quadrature for space and planetary plasma simulations.

## Key Features

- **Neutral Species, VDFs & Populations (`PlasmaPopulations.jl` Style)**: `isbits` `NeutralSpecies` (with chemical/nuclide formula parsing for atoms and planetary molecules), `AbstractNeutralVDF` (`ColdNeutral`, `Maxwellian`, `BiMaxwellianNeutral` with `pdf`, `mean`, `cov`, and zero-allocation `rand`), and composable `NeutralPopulation`s.
- **1D Radial, 3D Asymmetric & Time-Dependent Profiles**: Allocation-free, `ForwardDiff`-compatible `AbstractNeutralProfile` implementations—`ExponentialProfile`, `PowerLawProfile`, `ChamberlainProfile`, callable 1D/2D/3D `ExosphereProfile`, `DayNightProfile`, `PlumeProfile`, `CompositeProfile`, `TabulatedRadialProfile`, and `TimeDependentProfile`—paired with composable kinematics (`ZeroVelocity`, `ConstantVelocity`, `CorotationVelocity`, `RadialOutflowVelocity`) and temperature models (`ConstantTemperature`, `DayNightTemperature`).
- **Multi-Species GPU-Ready Exosphere & EUV Attenuation**: Combine `NeutralPopulation` tuples into an `isbits`, GPU-ready `ExosphereModel` (or use legacy `NeutralComponentCode` vectors via `build_exosphere_model`) with `NoAttenuation`, cylindrical shadowing (`ShadowCylinder`), curved-atmosphere `OpticalDepth` / Smith & Smith (1972) `chapman_function`, or 3D/dynamic line-of-sight `RayOpticalDepth`.
- **Zero-Overhead One-Way & Self-Consistent Two-Way Coupling**: Compile-time coupling traits (`OneWayCoupling` vs. `TwoWayCoupling`), conservative Eulerian exchange channels (`Photoionization`, `ElectronImpactIonization`, `ResonantChargeExchange`, `NonResonantChargeExchange`, `ElasticCollision`, `Recombination`), kinetic/Hybrid-PIC collision & ENA helpers, and structured `GriddedNeutralField` (`CartesianNeutralGrid`, `SphericalNeutralGrid`, `NeutralSourceBuffer` with conservative CIC deposition, `NeutralParticleBuffer`).
- **Adaptive Octree Dual-Cell Quadrature**: `integrate_exosphere_box!` adaptively subdivides 2D or 3D cells near sharp scale heights and inner planetary boundaries ($r < r_{\text{cut}}$) with analytic $\sinh(b)/b$ finite-cell corrections.
- **Optional Package Extensions**: Zero-cost weak-dependency extensions for `Unitful.jl` (dimensional `UnitScale`, `UnitfulProfile`, `strip_units`), `Adapt.jl` (GPU array migration), and `PlasmaPopulations.jl` (`Particle`/`Population`/`VDF` bridges).

## Installation

```julia
using Pkg
Pkg.add("Exosphere")
```

Or for local development:

```julia
using Pkg
Pkg.develop(path = "/path/to/Exosphere.jl")
```

## Quick Start

### 1. GPU-Ready Static One-Way Exosphere & Conservative Coupling

```julia
using Exosphere
using StaticArrays

sp_o = NeutralSpecies(; m = 16.0)
sp_h = NeutralSpecies(; m = 1.0)
pop_o = NeutralPopulation(
    sp_o,
    ExponentialProfile(100.0, 1.0, 0.2);
    velocity = CorotationVelocity((0.0, 0.0, 0.01)),
    temperature = ConstantTemperature(0.01),
)
pop_h = NeutralPopulation(
    sp_h,
    PowerLawProfile(10.0, 1.0, 2.0),
    Maxwellian(sp_h; T = 0.02),
)
od = OpticalDepth(; cross_sections = (0.05, 0.01), chapman = true)
exo = ExosphereModel(pop_o, pop_h; planet_radius = 1.0, optical_depth = od)
@assert isbits(exo)

x = SVector(1.5, 0.0, 0.0)
ion = PlasmaState(5.0, SVector(-2.0, 0.0, 0.0), 0.1, 16.0)
channels = (Photoionization(1.0e-6), ResonantChargeExchange(2.0e-3))

# One-way coupling: neutral feedback deposition compiles away to zero instructions
plasma_src = couple_point!(exo, x, channels, ion; species = 1)
```

### 2. Two-Way Dynamic Gridded Coupling (Same Solver Loop)

```julia
grid = CartesianNeutralGrid((-2.0, -2.0, -2.0), (0.5, 0.5, 0.5), (9, 9, 9))
field = GriddedNeutralField(grid, 2; masses = (16.0, 1.0), two_way = true)
sample_to_grid!(field, exo)

reset_sources!(field)
plasma_src = couple_point!(field, x, channels, ion; species = 1)
# External neutral solver reads neutral_sources(field) (dn, dmom, dE)
```

### 3. Legacy `NeutralComponentCode` & Adaptive Box Quadrature

```julia
o_comp = NeutralComponentCode(:exponential, 100.0, 0.2, 0.0, 0.01, "O")
h_comp = NeutralComponentCode(:powerlaw, 10.0, 0.0, 2.0, 0.01, "H")
exo_legacy = build_exosphere_model(;
    planet_radius = 1.0,
    components = [o_comp, h_comp],
    optical_depth = OpticalDepth(; cross_sections = [0.05, 0.01], chapman = true),
    inner_radius = 1.0,
)

sumN = zeros(n_components(exo_legacy))
sumNA = zeros(n_components(exo_legacy))
vol = integrate_exosphere_box!(sumN, sumNA, exo_legacy, x, SVector(0.2, 0.2, 0.2), 0, 1.0)
```

## Examples

- [`examples/planetary_exosphere_euv.jl`](examples/planetary_exosphere_euv.jl): Generates a 3-panel visualization of a 3-component Mars-like exosphere (`O`, `H`, `He`) with Chapman EUV optical-depth attenuation and volumetric photoionization production rates (requires `CairoMakie`):

```bash
julia --project=/path/to/env-with-CairoMakie examples/planetary_exosphere_euv.jl
```

## Testing & Benchmarking

Run the full unit, conservation, AD, extension, `Aqua.jl`, and `ExplicitImports.jl` test suite:

```bash
julia --project=Exosphere.jl -e 'using Pkg; Pkg.test()'
```

Benchmarks are defined in [`benchmark/benchmarks.jl`](benchmark/benchmarks.jl) using `BenchmarkTools.jl` and tracked across revisions with [`AirspeedVelocity.jl`](https://github.com/MilesCranmer/AirspeedVelocity.jl):

```bash
cd Exosphere.jl && benchpkg Exosphere --rev=main,dirty
```

## License

`Exosphere.jl` is licensed under the [MIT License](LICENSE).
