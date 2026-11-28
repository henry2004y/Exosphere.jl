# Two-Way Charge–Neutral Coupling & Extensibility

`Exosphere.jl` separates **neutral representation** ([`AbstractNeutralModel`](@ref)),
**coupling directionality** ([`OneWayCoupling`](@ref) vs. [`TwoWayCoupling`](@ref)),
and **charge–neutral interaction physics** ([`AbstractInteractionChannel`](@ref)) so the
same plasma solver loop runs with zero feedback overhead on a static GPU-ready exosphere
and upgrades seamlessly to a self-consistent two-way neutral grid or DSMC backend.

## `PlasmaPopulations.jl`-Style Neutral Species, VDFs & Populations

Neutral species and local velocity distribution functions follow the same composable style
as [`PlasmaPopulations.jl`](https://github.com/JuliaPlasma/PlasmaPopulations.jl):

- [`NeutralSpecies`](@ref) stores particle mass `m` (`isbits`) and parses chemical/nuclide
  formulas (`NeutralSpecies("O")`, `NeutralSpecies("CO2")`, `NeutralSpecies("{18}O")`).
- [`AbstractNeutralVDF`](@ref) ([`ColdNeutral`](@ref), [`Maxwellian`](@ref),
  [`BiMaxwellianNeutral`](@ref)) provides `pdf`, `Statistics.mean`, `Statistics.cov`, and
  zero-allocation `Random.rand`.
- [`NeutralPopulation`](@ref) binds a `NeutralSpecies`, an [`AbstractNeutralProfile`](@ref)
  (1D radial, [`DayNightProfile`](@ref), [`PlumeProfile`](@ref), [`CompositeProfile`](@ref),
  [`TabulatedRadialProfile`](@ref), or [`TimeDependentProfile`](@ref)), and either spatial
  kinematics/temperature models or a constant VDF.
- [`ExosphereModel`](@ref)`(pop1, pop2, ...; ...)` constructs a tuple-backed, `isbits`,
  GPU-ready static multi-species exosphere.

````@example coupling_static
using Exosphere
using StaticArrays

sp_o = NeutralSpecies(; m = 16.0)
sp_h = NeutralSpecies(; m = 1.0)

pop_o = NeutralPopulation(
    sp_o,
    DayNightProfile(ExponentialProfile(100.0, 1.0, 0.2); asymmetry = 0.4);
    velocity = CorotationVelocity((0.0, 0.0, 0.01)),
    temperature = ConstantTemperature(0.01),
)
pop_h = NeutralPopulation(
    sp_h,
    PowerLawProfile(10.0, 1.0, 2.0),
    Maxwellian(sp_h; T = 0.02),
)

exo = ExosphereModel(
    pop_o, pop_h;
    planet_radius = 1.0,
    attenuation = RayOpticalDepth(; cross_sections = (0.05, 0.01), s_max = 6.0),
)

println("isbits(exo)      = ", isbits(exo))
println("coupling_mode    = ", coupling_mode(exo))
println("dayside O state  = ", neutral_state(exo, SVector(1.4, 0.0, 0.0); species = 1))
println("nightside O dens = ", round(neutral_density(exo, SVector(-1.4, 0.0, 0.0); species = 1); sigdigits = 4))
````

## Conservative Eulerian Coupling (One-Way vs. Two-Way MHD)

Every [`AbstractInteractionChannel`](@ref) ([`Photoionization`](@ref),
[`ElectronImpactIonization`](@ref), [`ResonantChargeExchange`](@ref),
[`NonResonantChargeExchange`](@ref), [`ElasticCollision`](@ref), [`Recombination`](@ref))
evaluates exact conservation-law volumetric sources via [`compute_exchange`](@ref).

[`couple_point!`](@ref) dispatches on [`coupling_mode`](@ref):
- For [`OneWayCoupling`](@ref) models (`ExosphereModel`, read-only tabulated
  `GriddedNeutralField`), [`deposit_neutral_source!`](@ref) compiles away to a zero-cost
  no-op.
- For [`TwoWayCoupling`](@ref) models (`GriddedNeutralField(; two_way = true)`),
  [`couple_point!`](@ref) deposits the exact counter-balancing neutral moment source onto
  [`NeutralSourceBuffer`](@ref) (direct `@inbounds` cell accumulation for matched
  `CartesianIndex` grids, or conservative adjoint CIC deposition for arbitrary 3D
  positions) so an external neutral fluid solver can advance `(n, u, T)`.

````@example coupling_twoway
using Exosphere
using StaticArrays

sp_o = NeutralSpecies(; m = 16.0)
pop_o = NeutralPopulation(
    sp_o,
    ExponentialProfile(100.0, 1.0, 0.5);
    temperature = ConstantTemperature(0.01),
)
exo_static = ExosphereModel(pop_o; planet_radius = 1.0)

grid = CartesianNeutralGrid((-2.0, -2.0, -2.0), (0.5, 0.5, 0.5), (9, 9, 9))
field = GriddedNeutralField(grid, 1; masses = (16.0,), two_way = true)
sample_to_grid!(field, exo_static)

ion = PlasmaState(5.0, SVector(-2.0, 0.3, 0.0), 0.1, 16.0)
channels = (Photoionization(1.0e-5), ResonantChargeExchange(2.0e-3))

reset_sources!(field)
x = SVector(1.37, -0.22, 0.19)
plasma_src = couple_point!(field, x, channels, ion; species = 1)
sbuf = neutral_sources(field)

println("Plasma dn  = ", round(plasma_src.dn; sigdigits = 5))
println("Neutral Σdn = ", round(sum(sbuf.dn); sigdigits = 5))
println("Momentum conservation residual = ", maximum(abs.(plasma_src.dmom + sum(sbuf.dmom))))
````

## Kinetic / Hybrid-PIC & DSMC Coupling

Hybrid-PIC and kinetic solvers sample neutral velocities directly via
`rand(rng, local_vdf(model, x; species))`, compute exact Poisson collision probabilities
with [`collision_probability`](@ref) and [`particle_charge_exchange_rate`](@ref), and
record emitted energetic neutral atoms (ENAs) via [`charge_exchange_collision`](@ref) and
[`deposit_neutral_particle!`](@ref) (a compile-time no-op in one-way mode; stored in a
pre-allocated [`NeutralParticleBuffer`](@ref) in two-way mode).

````@example coupling_pic
using Exosphere
using Random: MersenneTwister
using StaticArrays

rng = MersenneTwister(7)
grid = CartesianNeutralGrid((-2.0, -2.0, -2.0), (0.5, 0.5, 0.5), (9, 9, 9))
field = GriddedNeutralField(grid, 1; masses = (16.0,), two_way = true, particle_capacity = 32)
fill!(field.n, 50.0)
fill!(field.T, 0.02)

x_p = SVector(1.2, 0.1, 0.0)
v_p = SVector(-3.0, 0.4, 0.0)
v_n = rand(rng, local_vdf(field, x_p; species = 1))
ev = charge_exchange_collision(x_p, v_p, 16.0, 1.0, v_n, 1)
deposit_neutral_particle!(field, ev.ena)

println("New ion vel  = ", round.(ev.ion_vel; sigdigits = 4))
println("Emitted ENA  = ", ev.ena)
println("Buffered ENAs = ", length(active_particles(neutral_particles(field))))
````

## Optional Package Extensions

`Exosphere.jl` keeps core runtime dependencies minimal and provides three automatic
package extensions loaded only when the corresponding package is imported:

- **`Unitful.jl`** (`ExosphereUnitfulExt`): physical-unit constructors for
  `ExponentialProfile`, `PowerLawProfile`, `ChamberlainProfile`, dimensional conversions
  via [`UnitScale`](@ref), [`to_code_units`](@ref), [`from_code_units`](@ref), and
  [`strip_units`](@ref).
- **`Adapt.jl`** (`ExosphereAdaptExt`): moves [`GriddedNeutralField`](@ref),
  [`NeutralSourceBuffer`](@ref), [`SphericalNeutralGrid`](@ref), and
  [`TabulatedRadialProfile`](@ref) between CPU and GPU array backends.
- **`PlasmaPopulations.jl`** (`ExospherePlasmaPopulationsExt`): converts neutral
  `PlasmaPopulations.Particle` (`q == 0`) to [`NeutralSpecies`](@ref), converts
  `PlasmaPopulations.Population` to [`PlasmaState`](@ref) (SI or code units via
  [`UnitScale`](@ref)), and converts `Cold` / `BiMaxwellian` VDFs directly into
  [`ColdNeutral`](@ref) / [`Maxwellian`](@ref) / [`BiMaxwellianNeutral`](@ref).
