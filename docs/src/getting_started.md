# Getting Started

## Installation

Install `Exosphere.jl` from the Julia package manager:

```julia
using Pkg
Pkg.add("Exosphere")
```

For local development against a checkout:

```julia
using Pkg
Pkg.develop(path = "/path/to/Exosphere.jl")
```

## Coordinate & Code-Unit Conventions

`Exosphere.jl` operates in consistent simulation code units (for example, lengths in
planetary radii $R_p$, densities in $\text{cm}^{-3}$ or normalized units, and cross
sections $\sigma_c$ in the matching inverse column-density units so that optical depth
$\tau = n_c \sigma_c H_c$ is dimensionless):

- **Planet-centered relative vectors**: Functions such as [`in_shadow`](@ref),
  [`photo_attenuation`](@ref), and [`integrate_exosphere_box!`](@ref) expect position
  vectors $\mathbf{x}$ relative to the planet center (`exo.center`), while callable
  [`ExosphereProfile`](@ref) objects subtract `p.center` automatically.
- **Solar direction (`solar_dir`)**: Normalized internally to a unit vector pointing
  **toward the Sun** (`+x` by default). Points with $\mathbf{x} \cdot \hat{\mathbf{s}} > 0$
  are on the dayside; points with $\mathbf{x} \cdot \hat{\mathbf{s}} < 0$ are on the
  nightside / downstream wake.
- **Surface clamping**: Single-component radial profiles clamp $n(r) = n_0$ for $r \le r_0$,
  whereas [`exo_component_density`](@ref) and [`exo_density`](@ref) return `0.0` inside
  `exo.Rp`.

## Walkthrough

The example below constructs a two-component exosphere (`O` exponential + `H` power-law)
with curved-atmosphere Chapman EUV attenuation, evaluates local densities and ion–neutral
source rates, and integrates the neutral column over an exterior 3D box:

```jldoctest
using Exosphere
using StaticArrays

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

x = SVector(1.5, 0.0, 0.0)
r = 1.5
n_tot = exo_density(exo, r)
atten = photo_attenuation(exo, x, r)
q_photo = photoionization_rate(1.0e-6, n_tot, atten)
q_cx = charge_exchange_rate(2.0e-19, 5.0, n_tot, 4.0e5)

sumN = zeros(n_components(exo))
sumNA = zeros(n_components(exo))
vol = integrate_exosphere_box!(sumN, sumNA, exo, x, SVector(0.2, 0.2, 0.2), 0, 1.0)

vol ≈ 0.008 &&
    n_tot > 0.0 &&
    0.0 < atten <= 1.0 &&
    q_photo > 0.0 &&
    q_cx > 0.0 &&
    sumNA[1] <= sumN[1]

# output
true
```

