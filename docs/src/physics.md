# Profiles & EUV Attenuation

## Single-Component Radial Profiles

All single-component radial profiles subtype [`AbstractNeutralProfile`](@ref) and share the
allocation-free [`neutral_density`](@ref) and [`local_scale_height`](@ref) interface
(`H(r) = -1 / (d\ln n / dr)`):

- [`ExponentialProfile`](@ref): Hydrostatic isothermal envelope with constant scale height $H$:
  ```math
  n(r) = n_0 \exp\!\left(-\frac{\max(r - r_0, 0)}{H}\right), \qquad H_{\text{loc}}(r) = H.
  ```
  Setting `H = Inf` produces a constant density $n_0$ (`local_scale_height == 0`).
- [`PowerLawProfile`](@ref): Radial power-law envelope with exponent $k \ge 0$ (e.g., $k = 2$
  for radial outflow / Haser limit):
  ```math
  n(r) = n_0 \left(\frac{r_0}{\max(r, r_0)}\right)^k, \qquad H_{\text{loc}}(r) = \frac{r}{k}.
  ```
  Small integer powers $k \in \{0, 1, 2, 3, 4\}$ use multiplication instead of `^`.
- [`ChamberlainProfile`](@ref): Classical exospheric hydrostatic distribution with scale
  parameter $H$:
  ```math
  n(r) = n_0 \exp\!\left(-H\left(\frac{1}{r_0} - \frac{1}{\max(r, r_0)}\right)\right), \qquad
  H_{\text{loc}}(r) = \frac{r^2}{H}.
  ```
- [`ExosphereProfile`](@ref) (constructed via [`exosphere_profile`](@ref)): Combined
  power-law and exponential envelope callable directly in 1D `(x,)`, 2D `(x, y)`, and
  3D `(x, y, z)` coordinates around `center`:
  ```math
  n(\mathbf{x}) = n_0 \left(\frac{r_0}{\max(r, r_0)}\right)^{\text{power}}
  \exp\!\left(-\frac{\max(r - r_0, 0)}{H}\right).
  ```

````@example profiles_demo
using Exosphere

exp_p = ExponentialProfile(10.0, 1.0, 0.5)
pl_p = PowerLawProfile(10.0, 1.0, 2.0)
ch_p = ChamberlainProfile(10.0, 1.0, 2.0)
haser = exosphere_profile(; n0 = 9.0, r0 = 1.0, H = Inf, power = 2.0)

println("Exponential: n(1.5) = ", round(neutral_density(exp_p, 1.5); sigdigits = 4),
    ", H = ", local_scale_height(exp_p, 1.5))
println("PowerLaw:    n(2.0) = ", neutral_density(pl_p, 2.0),
    ", H = ", local_scale_height(pl_p, 2.0))
println("Chamberlain: n(2.0) = ", round(neutral_density(ch_p, 2.0); sigdigits = 4),
    ", H = ", local_scale_height(ch_p, 2.0))
println("Haser 3D:    n(3,0,0) = ", haser(3.0, 0.0, 0.0))
````

## Multi-Component Exosphere Models

[`NeutralComponentCode`](@ref) packs a species' profile selector (`:exponential`,
`:powerlaw`, or `:chamberlain`), surface density $n_0$ at $R_p$, scale height $H$, power
exponent $k$, neutral temperature $T_0$, and label. [`build_exosphere_model`](@ref) combines
a list of components into an [`ExosphereModel`](@ref) queried via [`n_components`](@ref),
[`exo_component_density`](@ref), [`exo_density`](@ref), and [`exo_scale_height`](@ref).

## Cylindrical Shadow vs. Optical Depth & Chapman Attenuation

[`ExosphereModel`](@ref) supports either geometric shadowing or radiative EUV absorption
(mutually exclusive):

- **Geometric Shadow Cylinder ([`ShadowCylinder`](@ref))**: Points behind the planet
  ($\mathbf{x} \cdot \hat{\mathbf{s}} < 0$, downstream distance $\le \text{half\_height}$)
  within cylinder radius $R$ have [`in_shadow`](@ref)` == true` and
  [`shadow_factor`](@ref)` == 0.0`; all other points have attenuation `1.0`.
- **Plain Optical Depth ([`OpticalDepth`](@ref) with `chapman = false`)**: Computes vertical
  optical depth $\tau_v(r) = \sum_c n_c(r)\sigma_c H_c(r)$ on the dayside ($\tau = \max(\tau_v, \tau_{\text{floor}}) / \max(\cos\chi, \mu_{\text{floor}})$)
  and grazing-ray optical depth $\tau_g(p) = \sum_c n_c(p)\sigma_c \sqrt{2\pi p H_c(p)}$ at
  impact parameter $p = \sqrt{r^2 - (\mathbf{x}\cdot\hat{\mathbf{s}})^2} > R_p$ on the
  nightside, returning $\max(\exp(-\tau), q_{\min})$ (and $q_{\min}$ when $p \le R_p$ or
  $\tau \ge \tau_{\text{cutoff}}$).
- **Smith & Smith (1972) Chapman Function ([`OpticalDepth`](@ref) with `chapman = true`)**:
  Uses [`chapman_function`](@ref)`(Xp, cos_sza)` with $X_p = r / H_c$ for the dominant
  absorbing component (`chapman_component`, auto-selected by maximum $n_c(R_p)\sigma_c H_c$
  when unspecified) so that optical depth $\tau = \tau_v \operatorname{Ch}(X_p, \cos\chi)$
  transitions smoothly across the terminator ($\cos\chi \approx 0$) into the nightside
  shadow.

````@example attenuation_demo
using Exosphere
using StaticArrays

sc = ShadowCylinder(; solar_dir = (1.0, 0.0, 0.0), radius = 1.0)
o_comp = NeutralComponentCode(:exponential, 100.0, 0.2, 0.0, 0.01, "O")

od_plain = OpticalDepth(; cross_sections = [0.05], solar_dir = (1.0, 0.0, 0.0), chapman = false)
od_chap = OpticalDepth(; cross_sections = [0.05], solar_dir = (1.0, 0.0, 0.0), chapman = true)
exo_plain = build_exosphere_model(; planet_radius = 1.0, components = [o_comp], optical_depth = od_plain)
exo_chap = build_exosphere_model(; planet_radius = 1.0, components = [o_comp], optical_depth = od_chap)

x_day = SVector(1.5, 0.0, 0.0)
x_term = SVector(0.0, 1.5, 0.0)
x_night = SVector(-1.5, 0.5, 0.0)

for (label, x) in (("Dayside   ", x_day), ("Terminator", x_term), ("Nightside ", x_night))
    r = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
    println(
        label,
        " | shadow = ", shadow_factor(sc, x),
        " | plain = ", round(photo_attenuation(exo_plain, x, r); sigdigits = 4),
        " | chapman = ", round(photo_attenuation(exo_chap, x, r); sigdigits = 4),
    )
end
````

