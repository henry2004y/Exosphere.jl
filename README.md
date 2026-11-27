# Exosphere.jl

[![CI](https://github.com/henry2004y/Exosphere.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/henry2004y/Exosphere.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/henry2004y/Exosphere.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/henry2004y/Exosphere.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Standalone, zero-allocation planetary neutral exosphere profiles, EUV shadow and optical-depth/Chapman attenuation, ion–neutral source rates, and adaptive dual-cell box quadrature for space and planetary plasma simulations.

## Key Features

- **Allocation-Free Neutral Density Profiles**: Concrete `AbstractNeutralProfile` implementations—`ExponentialProfile`, `PowerLawProfile`, `ChamberlainProfile`, and the callable 1D/2D/3D `ExosphereProfile` (`exosphere_profile`)—supporting planetary exospheres and cometary $1/r^2$ Haser-limit comae with zero heap allocations.
- **Multi-Component Exosphere & EUV Attenuation**: Combine multiple `NeutralComponentCode` species (`:exponential`, `:powerlaw`, `:chamberlain`) in `ExosphereModel` (`build_exosphere_model`) with either geometric cylindrical shadowing (`ShadowCylinder`, `in_shadow`, `shadow_factor`) or curved-atmosphere `OpticalDepth` / Smith & Smith (1972) `chapman_function` EUV attenuation (`photo_attenuation`).
- **Pointwise Ion–Neutral Source Rates**: Lightweight, `@inline` helpers for local volumetric photoionization (`photoionization_rate`) and resonant charge-exchange (`charge_exchange_rate`) rates.
- **Adaptive Octree Dual-Cell Quadrature**: `integrate_exosphere_box!` adaptively subdivides 2D or 3D cells near sharp scale heights and inner planetary boundaries ($r < r_{\text{cut}}$) to integrate neutral column and EUV-attenuated production densities.

## Installation

`Exosphere.jl` is registered in [LingXiaoRegistry](https://github.com/henry2004y/LingXiaoRegistry). You can install it directly from GitHub or a local checkout:

```julia
using Pkg
Pkg.add(url = "https://github.com/henry2004y/Exosphere.jl.git")
```

Or for local development:

```julia
using Pkg
Pkg.develop(path = "/path/to/Exosphere.jl")
```

## Quick Start

```julia
using Exosphere
using StaticArrays

# 1. Single-component callable 1D/2D/3D radial profile
ep = exosphere_profile(; n0 = 1.0e4, r0 = 1.0, H = 0.25)
n_local = ep(1.5, 0.2, 0.0)

# 2. Multi-component exosphere with Chapman EUV optical-depth attenuation
o_comp = NeutralComponentCode(:exponential, 100.0, 0.2, 0.0, 0.01, "O")
h_comp = NeutralComponentCode(:powerlaw, 10.0, 0.0, 2.0, 0.01, "H")
od = OpticalDepth(;
    cross_sections = [0.05, 0.01],
    solar_dir = (1.0, 0.0, 0.0),
    chapman = true,
)
exo = build_exosphere_model(;
    planet_radius = 1.0,
    components = [o_comp, h_comp],
    optical_depth = od,
    inner_radius = 1.0,
)

# 3. Evaluate neutral density, EUV attenuation, and source rates
x = SVector(1.5, 0.0, 0.0)
r = 1.5
n_tot = exo_density(exo, r)
atten = photo_attenuation(exo, x, r)
q_photo = photoionization_rate(1.0e-6, n_tot, atten)
q_cx = charge_exchange_rate(2.0e-19, 5.0, n_tot, 4.0e5)

# 4. Adaptive box quadrature outside r_cut = 1.0
sumN = zeros(n_components(exo))
sumNA = zeros(n_components(exo))
vol = integrate_exosphere_box!(sumN, sumNA, exo, x, SVector(0.2, 0.2, 0.2), 0, 1.0)
```

## Public API

- **Neutral profiles**: `AbstractNeutralProfile`, `ExponentialProfile`, `PowerLawProfile`, `ChamberlainProfile`, `ExosphereProfile`, `exosphere_profile`, `neutral_density`, `local_scale_height`.
- **Multi-component exosphere & attenuation**: `NeutralComponentCode`, `ShadowCylinder`, `OpticalDepth`, `ExosphereModel`, `build_exosphere_model`, `in_shadow`, `shadow_factor`, `n_components`, `exo_component_density`, `exo_density`, `exo_scale_height`, `chapman_function`, `photo_attenuation`.
- **Ion–neutral source rates**: `photoionization_rate`, `charge_exchange_rate`.
- **Adaptive quadrature**: `integrate_exosphere_box!`.

## Testing & Benchmarking

Run the unit and `Aqua.jl` quality test suite:

```bash
julia --project=Exosphere.jl -e 'using Pkg; Pkg.test()'
```

Benchmarks are defined in [`benchmark/benchmarks.jl`](benchmark/benchmarks.jl) using `BenchmarkTools.jl` and tracked across revisions with [`AirspeedVelocity.jl`](https://github.com/MilesCranmer/AirspeedVelocity.jl):

```bash
# Compare the current working tree against main using the AirspeedVelocity CLI
cd Exosphere.jl && benchpkg Exosphere --rev=main,dirty
```

Or from Julia:

```julia
using AirspeedVelocity

AirspeedVelocity.benchmark("Exosphere", ["main", "dirty"]; path = "/path/to/Exosphere.jl")
```

## License

`Exosphere.jl` is licensed under the [MIT License](LICENSE).
