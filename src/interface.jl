# ==============================================================================
# Core Interfaces, Coupling Traits, State Containers & Code-Unit Scaling
# ==============================================================================

# CODATA 2018 SI constants (matching PlasmaPopulations.jl)
"""
    U_SI

Unified atomic mass unit in kilograms (`1.6605390666e-27` kg, CODATA 2018).
"""
const U_SI = 1.6605390666e-27

"""
    MP_SI

Proton mass in kilograms (`1.67262192369e-27` kg, CODATA 2018).
"""
const MP_SI = 1.67262192369e-27

"""
    ME_SI

Electron mass in kilograms (`9.1093837015e-31` kg, CODATA 2018).
"""
const ME_SI = 9.1093837015e-31

"""
    KB_SI

Boltzmann constant in J/K (`1.380649e-23` J/K, exact SI).
"""
const KB_SI = 1.380649e-23

"""
    E_SI

Elementary charge in Coulombs (`1.602176634e-19` C, exact SI).
"""
const E_SI = 1.602176634e-19

# ------------------------------------------------------------------------------
# Unit Adapter Hooks (no-op for bare numbers; extended in ExosphereUnitfulExt)
# ------------------------------------------------------------------------------

"""
    si_mass(x)

Convert `x` to kilograms when `x` carries physical units (via `ExosphereUnitfulExt`),
returning bare numbers unchanged.
"""
@inline si_mass(x::Real) = x

"""
    si_length(x)

Convert `x` to meters when `x` carries physical units (via `ExosphereUnitfulExt`),
returning bare numbers unchanged.
"""
@inline si_length(x::Real) = x

"""
    si_number_density(x)

Convert `x` to `m^-3` when `x` carries physical units (via `ExosphereUnitfulExt`),
returning bare numbers unchanged.
"""
@inline si_number_density(x::Real) = x

"""
    si_velocity(x)

Convert `x` to `m/s` when `x` carries physical units (via `ExosphereUnitfulExt`),
returning bare numbers unchanged.
"""
@inline si_velocity(x::Real) = x

"""
    si_temperature(x)

Convert `x` to Kelvin when `x` carries temperature units (via `ExosphereUnitfulExt`),
returning bare numbers unchanged.
"""
@inline si_temperature(x::Real) = x

"""
    si_energy(x)

Convert `x` to Joules when `x` carries energy or temperature units (via
`ExosphereUnitfulExt`), returning bare numbers unchanged.
"""
@inline si_energy(x::Real) = x

"""
    si_cross_section(x)

Convert `x` to `m^2` when `x` carries area units (via `ExosphereUnitfulExt`),
returning bare numbers unchanged.
"""
@inline si_cross_section(x::Real) = x

"""
    si_frequency(x)

Convert `x` to `s^-1` when `x` carries frequency units (via `ExosphereUnitfulExt`),
returning bare numbers unchanged.
"""
@inline si_frequency(x::Real) = x

# ------------------------------------------------------------------------------
# Abstract Type Hierarchy
# ------------------------------------------------------------------------------

"""
    AbstractNeutralModel

Abstract supertype for single- or multi-species neutral exosphere/atmosphere models
(static analytic [`ExosphereModel`](@ref), tabulated or two-way mutable
[`GriddedNeutralField`](@ref), or external neutral solver adapters).
"""
abstract type AbstractNeutralModel end

"""
    AbstractNeutralProfile

Abstract supertype for scalar neutral number density profiles (1D radial, 3D asymmetric,
composite, tabulated, or time-dependent).
"""
abstract type AbstractNeutralProfile end

"""
    AbstractNeutralVDF

Abstract supertype for neutral particle velocity distribution functions normalized to
``\\int f(\\mathbf{v})\\,d^3v = 1``.
"""
abstract type AbstractNeutralVDF end

"""
    AbstractVelocityModel

Abstract supertype for spatial/temporal neutral bulk velocity models
``\\mathbf{u}_n(\\mathbf{x}, t)``.
"""
abstract type AbstractVelocityModel end

"""
    AbstractTemperatureModel

Abstract supertype for spatial/temporal neutral temperature models
``T_n(\\mathbf{x}, t)``.
"""
abstract type AbstractTemperatureModel end

"""
    AbstractAttenuation

Abstract supertype for solar EUV radiative attenuation models returning a transmission
factor in ``[0, 1]``.
"""
abstract type AbstractAttenuation end

"""
    AbstractInteractionChannel

Abstract supertype for charge–neutral reaction channels (photoionization, electron-impact
ionization, charge exchange, elastic collisions, recombination).
"""
abstract type AbstractInteractionChannel end

"""
    AbstractNeutralGrid

Abstract supertype for structured neutral spatial grids ([`CartesianNeutralGrid`](@ref),
[`SphericalNeutralGrid`](@ref)).
"""
abstract type AbstractNeutralGrid end

# ------------------------------------------------------------------------------
# Coupling Mode Traits
# ------------------------------------------------------------------------------

"""
    AbstractCouplingMode

Trait supertype distinguishing read-only one-way neutral backgrounds
([`OneWayCoupling`](@ref)) from mutable two-way coupled neutral models
([`TwoWayCoupling`](@ref)).
"""
abstract type AbstractCouplingMode end

"""
    OneWayCoupling <: AbstractCouplingMode

Compile-time trait for read-only/static neutral models where plasma-to-neutral feedback
deposition compiles away to a zero-cost no-op.
"""
struct OneWayCoupling <: AbstractCouplingMode end

"""
    TwoWayCoupling <: AbstractCouplingMode

Compile-time trait for dynamic neutral models that accumulate plasma-to-neutral mass,
momentum, energy, or particle feedback.
"""
struct TwoWayCoupling <: AbstractCouplingMode end

"""
    coupling_mode(model::AbstractNeutralModel) -> AbstractCouplingMode

Return [`OneWayCoupling`](@ref)`()` (default) or [`TwoWayCoupling`](@ref)`()` for `model`.
"""
@inline coupling_mode(::AbstractNeutralModel) = OneWayCoupling()

# ------------------------------------------------------------------------------
# Local Macroscopic States, Conservative Moment Sources & Kinetic Particles
# ------------------------------------------------------------------------------

"""
    NeutralState(n, u, T)

Local macroscopic state of one neutral species: number density `n`, bulk velocity
`u::SVector{3}`, and temperature `T` (in energy units when `kB = 1`, or Kelvin when
paired with physical `kB`).
"""
struct NeutralState{TN <: Number, TV <: Number, TT <: Number}
    n::TN
    u::SVector{3, TV}
    T::TT
end

@inline NeutralState(n::Number, u::NTuple{3, Number}, T::Number) =
    NeutralState(n, SVector{3}(u[1], u[2], u[3]), T)

@inline function NeutralState(n::Number, u::AbstractVector{<:Number}, T::Number)
    length(u) == 3 || throw(ArgumentError("NeutralState: velocity u must have length 3"))
    return NeutralState(n, SVector{3}(u[1], u[2], u[3]), T)
end

"""
    PlasmaState(n, u, T, m)

Local macroscopic state of one charged plasma species: number density `n`, bulk velocity
`u::SVector{3}`, temperature `T`, and particle mass `m`.
"""
struct PlasmaState{TN <: Number, TV <: Number, TT <: Number, TM <: Number}
    n::TN
    u::SVector{3, TV}
    T::TT
    m::TM
end

@inline PlasmaState(n::Number, u::NTuple{3, Number}, T::Number, m::Number) =
    PlasmaState(n, SVector{3}(u[1], u[2], u[3]), T, m)

@inline function PlasmaState(n::Number, u::AbstractVector{<:Number}, T::Number, m::Number)
    length(u) == 3 || throw(ArgumentError("PlasmaState: velocity u must have length 3"))
    return PlasmaState(n, SVector{3}(u[1], u[2], u[3]), T, m)
end

"""
    MomentSource(dn, dmom, dE)

Volumetric conservation-law source rates (or integrated deltas) for one species:
- `dn`: number density rate ``\\partial n / \\partial t``
- `dmom::SVector{3}`: momentum density rate ``\\partial (m n \\mathbf{u}) / \\partial t``
- `dE`: total energy density rate ``\\partial E / \\partial t``, where
  ``E = \\frac{1}{2} m n |\\mathbf{u}|^2 + \\frac{3}{2} n k_B T``.
"""
struct MomentSource{TN <: Number, TMom <: Number, TE <: Number}
    dn::TN
    dmom::SVector{3, TMom}
    dE::TE
end

@inline MomentSource(dn::Number, dmom::NTuple{3, Number}, dE::Number) =
    MomentSource(dn, SVector{3}(dmom[1], dmom[2], dmom[3]), dE)

@inline Base.:+(a::MomentSource, b::MomentSource) =
    MomentSource(a.dn + b.dn, a.dmom + b.dmom, a.dE + b.dE)

@inline Base.:-(a::MomentSource, b::MomentSource) =
    MomentSource(a.dn - b.dn, a.dmom - b.dmom, a.dE - b.dE)

@inline Base.:-(a::MomentSource) = MomentSource(-a.dn, -a.dmom, -a.dE)

@inline Base.:*(s::Number, a::MomentSource) = MomentSource(s * a.dn, s * a.dmom, s * a.dE)
@inline Base.:*(a::MomentSource, s::Number) = MomentSource(a.dn * s, a.dmom * s, a.dE * s)

@inline Base.zero(::Type{MomentSource{TN, TMom, TE}}) where {TN, TMom, TE} =
    MomentSource(zero(TN), SVector{3, TMom}(zero(TMom), zero(TMom), zero(TMom)), zero(TE))

@inline Base.zero(a::MomentSource) = MomentSource(zero(a.dn), zero(a.dmom), zero(a.dE))

"""
    CoupledExchange(plasma::MomentSource, neutral::MomentSource)

Paired volumetric [`MomentSource`](@ref) terms acting on the plasma (`plasma`) and neutral
(`neutral`) species from one or more charge–neutral interaction channels.
"""
struct CoupledExchange{SP <: MomentSource, SN <: MomentSource}
    plasma::SP
    neutral::SN
end

@inline Base.:+(a::CoupledExchange, b::CoupledExchange) =
    CoupledExchange(a.plasma + b.plasma, a.neutral + b.neutral)

@inline Base.zero(a::CoupledExchange) = CoupledExchange(zero(a.plasma), zero(a.neutral))

"""
    NeutralParticle(pos, vel, mass, weight, species = Int32(1))

Lightweight `isbits` neutral macroparticle (e.g., an energetic neutral atom produced by
charge exchange or a sputtered surface neutral) for kinetic/PIC/DSMC two-way coupling.
"""
struct NeutralParticle{FT <: AbstractFloat}
    pos::SVector{3, FT}
    vel::SVector{3, FT}
    mass::FT
    weight::FT
    species::Int32
end

function NeutralParticle(
        pos::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}},
        vel::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}},
        mass::Real,
        weight::Real,
        species::Integer = Int32(1),
    )
    FT = float(
        promote_type(
            typeof(pos[1]), typeof(pos[2]), typeof(pos[3]),
            typeof(vel[1]), typeof(vel[2]), typeof(vel[3]),
            typeof(mass), typeof(weight),
        )
    )
    p = SVector{3, FT}(FT(pos[1]), FT(pos[2]), FT(pos[3]))
    v = SVector{3, FT}(FT(vel[1]), FT(vel[2]), FT(vel[3]))
    return NeutralParticle{FT}(p, v, FT(mass), FT(weight), Int32(species))
end

# ------------------------------------------------------------------------------
# Generic Neutral Model Query & Feedback Protocol
# ------------------------------------------------------------------------------

"""
    neutral_state(model::AbstractNeutralModel, loc; t = nothing, species = 1) -> NeutralState

Evaluate the local macroscopic neutral state `(n, u, T)` of `species` at spatial position
or grid index `loc` and optional time `t`.
"""
function neutral_state end

"""
    local_vdf(model, loc; t = nothing, species = 1, kB = 1) -> AbstractNeutralVDF

Return the local velocity distribution function of `species` at position or grid index
`loc` and optional time `t`.
"""
function local_vdf end

"""
    species_mass(model, c = 1)

Return the particle mass of neutral component `c`.
"""
function species_mass end

"""
    deposit_neutral_source!(model::AbstractNeutralModel, loc, src::MomentSource, species = 1)

Accumulate volumetric neutral source `src` for `species` at grid index or position `loc`.
Dispatches on [`coupling_mode`](@ref): compiles to a zero-cost no-op for
[`OneWayCoupling`](@ref) models.
"""
@inline function deposit_neutral_source!(
        model::AbstractNeutralModel, loc, src::MomentSource, species::Integer = 1,
    )
    return _deposit_neutral_source!(coupling_mode(model), model, loc, src, species)
end

@inline _deposit_neutral_source!(
    ::OneWayCoupling, ::AbstractNeutralModel, loc, ::MomentSource, ::Integer,
) = nothing

function _deposit_neutral_source!(
        ::TwoWayCoupling, model::AbstractNeutralModel, loc, src::MomentSource, species::Integer,
    )
    throw(MethodError(_deposit_neutral_source!, (TwoWayCoupling(), model, loc, src, species)))
end

"""
    deposit_neutral_particle!(model::AbstractNeutralModel, p::NeutralParticle)

Deposit a discrete neutral macroparticle (e.g., an ENA or sputtered neutral) into `model`.
Dispatches on [`coupling_mode`](@ref): compiles to a zero-cost no-op for
[`OneWayCoupling`](@ref) models.
"""
@inline function deposit_neutral_particle!(model::AbstractNeutralModel, p::NeutralParticle)
    return _deposit_neutral_particle!(coupling_mode(model), model, p)
end

@inline _deposit_neutral_particle!(
    ::OneWayCoupling, ::AbstractNeutralModel, ::NeutralParticle,
) = nothing

function _deposit_neutral_particle!(
        ::TwoWayCoupling, model::AbstractNeutralModel, p::NeutralParticle,
    )
    throw(MethodError(_deposit_neutral_particle!, (TwoWayCoupling(), model, p)))
end

"""
    reset_sources!(model::AbstractNeutralModel)

Reset accumulated two-way feedback buffers at the start of a coupling step. Compiles to a
zero-cost no-op for [`OneWayCoupling`](@ref) models.
"""
@inline reset_sources!(model::AbstractNeutralModel) =
    _reset_sources!(coupling_mode(model), model)

@inline _reset_sources!(::OneWayCoupling, ::AbstractNeutralModel) = nothing

"""
    neutral_sources(model::AbstractNeutralModel)

Return the Eulerian source accumulator buffer of `model`, or `nothing` for static models.
"""
@inline neutral_sources(::AbstractNeutralModel) = nothing

"""
    neutral_particles(model::AbstractNeutralModel)

Return the kinetic neutral particle buffer of `model`, or `nothing` when absent.
"""
@inline neutral_particles(::AbstractNeutralModel) = nothing

# ------------------------------------------------------------------------------
# Dimensionless Code-Unit Scaling Utility
# ------------------------------------------------------------------------------

"""
    UnitScale{FT<:AbstractFloat}
    UnitScale(; length = 1.0, density = 1.0, velocity = 1.0, mass = MP_SI,
              temperature = nothing, FT = Float64)

Reference scales for converting between physical (SI or `Unitful.Quantity` via
`ExosphereUnitfulExt`) and dimensionless simulation code units:
- `L0`: reference length [m]
- `n0`: reference number density [m⁻³]
- `v0`: reference velocity [m/s]
- `m0`: reference particle mass [kg]
- `t0 = L0 / v0`: reference time [s]
- `T0`: reference temperature energy `k_B * T` [J] (defaults to `m0 * v0^2`)
"""
struct UnitScale{FT <: AbstractFloat}
    L0::FT
    n0::FT
    v0::FT
    m0::FT
    t0::FT
    T0::FT
end

function UnitScale(;
        length::Number = 1.0,
        density::Number = 1.0,
        velocity::Number = 1.0,
        mass::Number = MP_SI,
        temperature::Union{Nothing, Number} = nothing,
        FT::Type{<:AbstractFloat} = Float64,
    )
    L0 = FT(si_length(length))
    n0 = FT(si_number_density(density))
    v0 = FT(si_velocity(velocity))
    m0 = FT(si_mass(mass))
    (L0 > 0 && isfinite(L0)) || throw(ArgumentError("UnitScale requires finite length > 0"))
    (n0 > 0 && isfinite(n0)) || throw(ArgumentError("UnitScale requires finite density > 0"))
    (v0 > 0 && isfinite(v0)) || throw(ArgumentError("UnitScale requires finite velocity > 0"))
    (m0 > 0 && isfinite(m0)) || throw(ArgumentError("UnitScale requires finite mass > 0"))
    t0 = L0 / v0
    T0 = temperature === nothing ? m0 * v0 * v0 : FT(si_energy(temperature))
    (T0 > 0 && isfinite(T0)) || throw(ArgumentError("UnitScale requires finite temperature > 0"))
    return UnitScale{FT}(L0, n0, v0, m0, t0, T0)
end

"""
    strip_units(p::AbstractNeutralProfile, scale::UnitScale)

Convert a physical-unit profile (created via `ExosphereUnitfulExt`) into a zero-allocation,
`isbits` simulation code-unit profile using `scale`. Returns unitless profiles unchanged.
"""
@inline strip_units(p::AbstractNeutralProfile, ::UnitScale) = p

"""
    to_code_units(scale::UnitScale, val, quantity::Symbol)
    to_code_units(scale::UnitScale, x)

Convert a physical SI value `val` of kind `quantity` (`:length`, `:density`, `:velocity`,
`:mass`, `:time`, `:frequency`, `:cross_section`, `:energy`, `:temperature`,
`:number_rate`, `:momentum_rate`, `:energy_rate`) to simulation code units defined by
`scale`. With `Unitful.jl` loaded, `Unitful.Quantity` arguments infer dimensions
automatically.
"""
@inline function to_code_units(scale::UnitScale{FT}, val::Real, quantity::Symbol) where {FT}
    v = FT(val)
    quantity === :length && return v / scale.L0
    quantity === :density && return v / scale.n0
    quantity === :velocity && return v / scale.v0
    quantity === :mass && return v / scale.m0
    quantity === :time && return v / scale.t0
    quantity === :frequency && return v * scale.t0
    quantity === :cross_section && return v * (scale.n0 * scale.L0)
    quantity === :energy && return v / scale.T0
    quantity === :temperature && return (v * FT(KB_SI)) / scale.T0
    quantity === :number_rate && return v * (scale.t0 / scale.n0)
    quantity === :momentum_rate && return v * (scale.t0 / (scale.m0 * scale.n0 * scale.v0))
    quantity === :energy_rate && return v * (scale.t0 / (scale.n0 * scale.T0))
    throw(ArgumentError("to_code_units: unknown quantity :$quantity"))
end

"""
    from_code_units(scale::UnitScale, val, quantity::Symbol)

Convert a dimensionless code-unit value `val` of kind `quantity` back to SI units using
`scale`.
"""
@inline function from_code_units(scale::UnitScale{FT}, val::Real, quantity::Symbol) where {FT}
    v = FT(val)
    quantity === :length && return v * scale.L0
    quantity === :density && return v * scale.n0
    quantity === :velocity && return v * scale.v0
    quantity === :mass && return v * scale.m0
    quantity === :time && return v * scale.t0
    quantity === :frequency && return v / scale.t0
    quantity === :cross_section && return v / (scale.n0 * scale.L0)
    quantity === :energy && return v * scale.T0
    quantity === :temperature && return (v * scale.T0) / FT(KB_SI)
    quantity === :number_rate && return v * (scale.n0 / scale.t0)
    quantity === :momentum_rate && return v * (scale.m0 * scale.n0 * scale.v0 / scale.t0)
    quantity === :energy_rate && return v * (scale.n0 * scale.T0 / scale.t0)
    throw(ArgumentError("from_code_units: unknown quantity :$quantity"))
end
