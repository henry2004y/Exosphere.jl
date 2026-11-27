# Sources, Quadrature & Performance

## Ion–Neutral Source Rates & `Float32` Usage

`Exosphere.jl` provides `@inline`, zero-allocation source-rate kernels for hybrid-PIC and
kinetic/fluid plasma solvers:

- [`photoionization_rate`](@ref)`(nu_photo, neutral_n, atten = 1.0)` computes
  $q_{\text{photo}} = \nu_{\text{photo}} \, n_n \, A$.
- [`charge_exchange_rate`](@ref)`(sigma_cx, ion_n, neutral_n, v_rel)` computes
  $R_{\text{cx}} = \sigma_{\text{cx}} \, n_i \, n_n \, v_{\text{rel}}$.

Single-component profiles (`ExponentialProfile`, `PowerLawProfile`, `ChamberlainProfile`)
also accept `FT = Float32` for low-memory and GPU-kernel evaluations:

````@example sources_f32
using Exosphere

exp32 = ExponentialProfile(10.0f0, 1.0f0, 0.5f0; FT = Float32)
n32 = neutral_density(exp32, 1.5f0)
q32 = photoionization_rate(1.0f-6, n32, 0.8f0)
cx32 = charge_exchange_rate(2.0f-19, 5.0f0, n32, 4.0f5)

println("n32  = ", n32, " (", typeof(n32), ")")
println("q32  = ", q32, " (", typeof(q32), ")")
println("cx32 = ", cx32, " (", typeof(cx32), ")")
````

## Adaptive Octree Dual-Cell Box Quadrature

When computing cell-integrated neutral column densities `sumN` and EUV-attenuated
production weights `sumNA` on Cartesian or AMR grids near a planetary obstacle,
[`integrate_exosphere_box!`](@ref) recursively refines each axis-aligned box `(c, h)`
around `exo.center` up to depth `8`:

1. **Interior cull**: Boxes fully inside the inner sphere ($r + \text{half\_diag} \le r_{\text{cut}}$)
   immediately return `0.0`.
2. **Adaptive splitting**: Boxes are subdivided into 8 octants (or 4 quadrants when one
   axis has $h_d \le 0$ for 2D simulations) whenever $h_{\max} > 2 H_{\text{steep}}$ in the
   exterior, or $h_{\max} > 0.5 H_{\text{steep}}$ if the box straddles $r = r_{\text{cut}}$.
3. **Sub-cell boundary volume fraction**: Straddling leaf cells estimate the exterior
   fraction using a SIMD-vectorized $4 \times 4 \times 4$ (3D) or $4 \times 4$ (2D)
   midpoint lattice against $r_{\text{cut}}^2$.
4. **Finite-cell $\sinh(b)/b$ correction**: Non-straddling leaf cells multiply each
   component's density by $\prod_{d} \sinh(b_d)/b_d$ with
   $b_d = \frac{h_d |c_d|}{2 r H_c}$, preserving exact exponential cell averages across
   coarse cells without extra quadrature points.

````@example box_quadrature
using Exosphere
using StaticArrays

o_comp = NeutralComponentCode(:exponential, 100.0, 0.2, 0.0, 0.01, "O")
h_comp = NeutralComponentCode(:powerlaw, 10.0, 0.0, 2.0, 0.01, "H")
od = OpticalDepth(; cross_sections = [0.05, 0.01], solar_dir = (1.0, 0.0, 0.0), chapman = true)
exo = build_exosphere_model(; planet_radius = 1.0, components = [o_comp, h_comp], optical_depth = od)

sumN = zeros(n_components(exo))
sumNA = zeros(n_components(exo))

# Exterior 3D box
vol_3d = integrate_exosphere_box!(sumN, sumNA, exo, SVector(1.5, 0.0, 0.0), SVector(0.2, 0.2, 0.2), 0, 1.0)
println("3D exterior vol = ", round(vol_3d; sigdigits = 4),
    ", sumN = ", round.(sumN; sigdigits = 4), ", sumNA = ", round.(sumNA; sigdigits = 4))

# Boundary-straddling 2D cell (h[3] = 0.0)
fill!(sumN, 0.0)
fill!(sumNA, 0.0)
area_2d = integrate_exosphere_box!(sumN, sumNA, exo, SVector(1.05, 0.1, 0.0), SVector(0.4, 0.4, 0.0), 0, 1.0)
println("2D straddle area = ", round(area_2d; sigdigits = 4),
    ", sumN = ", round.(sumN; sigdigits = 4), ", sumNA = ", round.(sumNA; sigdigits = 4))
````

## Planetary Exosphere EUV Example

The standalone script
[`examples/planetary_exosphere_euv.jl`](https://github.com/henry2004y/Exosphere.jl/blob/main/examples/planetary_exosphere_euv.jl)
visualizes a 3-component Mars-like exosphere (`O` exponential, `H` power-law, `He`
Chamberlain) with Chapman EUV attenuation (`solar_dir = (+1, 0, 0)`):

![Planetary Exosphere & Chapman EUV Attenuation](assets/planetary_exosphere_euv.png)

To run the example locally and save the figure:

```bash
julia --project=/path/to/env-with-CairoMakie examples/planetary_exosphere_euv.jl
```

## Performance Benchmarks

The [`benchmark/benchmarks.jl`](https://github.com/henry2004y/Exosphere.jl/blob/main/benchmark/benchmarks.jl)
suite tracks profile evaluations, shadow/optical-depth/Chapman attenuation, batched
photoionization, and 2D/3D adaptive box quadrature using `BenchmarkTools.jl` and
`AirspeedVelocity.jl`:

```bash
cd Exosphere.jl && benchpkg Exosphere --rev=main,dirty
```

