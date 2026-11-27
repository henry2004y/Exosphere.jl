```@raw html
---
layout: home

hero:
  name: "Exosphere.jl"
  text: "Neutral Exospheres & EUV Attenuation"
  tagline: "Zero-allocation planetary neutral exosphere profiles, Chapman/optical-depth EUV attenuation, ion–neutral source rates, and adaptive dual-cell box quadrature."
  actions:
    - theme: brand
      text: Getting Started
      link: /getting_started
    - theme: alt
      text: Profiles & Attenuation
      link: /physics
    - theme: alt
      text: API Reference
      link: /api

features:
  - title: Allocation-Free Neutral Profiles
    details: Concrete ExponentialProfile, PowerLawProfile, ChamberlainProfile, and callable 1D/2D/3D ExosphereProfile supporting planetary exospheres and cometary 1/r² Haser comae with 0 B heap allocations.
  - title: EUV Shadow & Chapman Attenuation
    details: Multi-component ExosphereModel with geometric cylindrical shadowing (ShadowCylinder) or curved-atmosphere OpticalDepth and Smith & Smith (1972) Chapman grazing-incidence transmission.
  - title: Ion–Neutral Source Rates
    details: Inline, type-stable Float64/Float32 helpers for local volumetric photoionization and resonant charge-exchange collision rates with zero allocations.
  - title: Adaptive Octree Box Quadrature
    details: Recursive integrate_exosphere_box! refines 2D and 3D cells near sharp scale heights and inner planetary boundaries (r < rcut) with analytic sinh(b)/b finite-cell corrections.
---
```

# Overview

**Exosphere.jl** is a standalone, pure-Julia package providing planetary neutral exosphere
models, EUV photo-attenuation, ion–neutral source rates, and adaptive cell-averaged
quadrature for space and planetary plasma simulations.

## Quick Example

````@example index_quick
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

# 4. Adaptive box quadrature outside rcut = 1.0
sumN = zeros(n_components(exo))
sumNA = zeros(n_components(exo))
vol = integrate_exosphere_box!(sumN, sumNA, exo, x, SVector(0.2, 0.2, 0.2), 0, 1.0)

println("n_local = ", round(n_local; sigdigits = 4))
println("n_tot   = ", round(n_tot; sigdigits = 4), ", atten = ", round(atten; sigdigits = 4))
println("q_photo = ", round(q_photo; sigdigits = 4), ", q_cx = ", round(q_cx; sigdigits = 4))
println("vol     = ", vol, ", sumN = ", round.(sumN; sigdigits = 4), ", sumNA = ", round.(sumNA; sigdigits = 4))
````

