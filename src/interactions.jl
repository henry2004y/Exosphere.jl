# ==============================================================================
# Conservative Charge-Neutral Interactions, Kinetic Kernels & Coupling Entry Point
# ==============================================================================

Base.broadcastable(ch::AbstractInteractionChannel) = Ref(ch)

@inline _eval_rate_coeff(c::Number, ::Number) = c
@inline _eval_rate_coeff(f::F, arg::Number) where {F} = f(arg)

@inline _add_energy_shift(e::Number, ::Nothing, ::Number) = e
@inline _add_energy_shift(e::Number, delta::Number, rate::Number) = e + rate * delta

@inline _sub_energy_shift(e::Number, ::Nothing, ::Number) = e
@inline _sub_energy_shift(e::Number, delta::Number, rate::Number) = e - rate * delta

@inline function _needs_attenuation(::Type{T}) where {T <: AbstractInteractionChannel}
    return T <: Photoionization
end

@inline _needs_attenuation(::Type{Tuple{}}) = false

@inline function _needs_attenuation(::Type{T}) where {T <: Tuple}
    return _needs_attenuation(Base.tuple_type_head(T)) ||
        _needs_attenuation(Base.tuple_type_tail(T))
end

# ------------------------------------------------------------------------------
# Effective Relative Speed (Pauls et al. 1995)
# ------------------------------------------------------------------------------

"""
    effective_relative_speed(plasma::PlasmaState, neutral::NeutralState, m_n; kB = 1)

Effective relative ion–neutral interaction speed combining bulk drift and thermal motions
(Pauls et al. 1995):

```math
v_{\\text{eff}} = \\sqrt{|\\mathbf{u}_i - \\mathbf{u}_n|^2
+ \\frac{8}{\\pi}\\left(\\frac{k_B T_i}{m_i} + \\frac{k_B T_n}{m_n}\\right)}.
```
"""
@inline @muladd function effective_relative_speed(
        plasma::PlasmaState, neutral::NeutralState, m_n::Number; kB::Number = 1,
    )
    du1 = plasma.u[1] - neutral.u[1]
    du2 = plasma.u[2] - neutral.u[2]
    du3 = plasma.u[3] - neutral.u[3]
    du2_sum = du1 * du1 + du2 * du2 + du3 * du3
    vth_i2 = (kB * plasma.T) / plasma.m
    vth_n2 = (kB * neutral.T) / m_n
    c8pi = oftype(float(one(real(typeof(du1 / du1)))), 8 / π)
    v2 = du2_sum + c8pi * (vth_i2 + vth_n2)
    return v2 > zero(v2) ? sqrt(v2) : zero(du1)
end

# ------------------------------------------------------------------------------
# Interaction Channel Types (all isbits)
# ------------------------------------------------------------------------------

"""
    Photoionization(nu_photo; E_excess = nothing) <: AbstractInteractionChannel

Photoionization channel ``N + h\\nu \\to I^+ + e^-`` with unattenuated ionization frequency
`nu_photo` and optional photoelectron excess energy `E_excess` added to the plasma energy.
"""
struct Photoionization{TNu <: Number, TE <: Union{Nothing, Number}} <:
    AbstractInteractionChannel
    nu_photo::TNu
    E_excess::TE
end

Photoionization(nu_photo::Number; E_excess::Union{Nothing, Number} = nothing) =
    Photoionization{typeof(nu_photo), typeof(E_excess)}(nu_photo, E_excess)

"""
    ElectronImpactIonization(k_imp; E_ion = nothing) <: AbstractInteractionChannel

Electron-impact ionization channel ``N + e^- \\to I^+ + 2e^-`` with rate coefficient `k_imp`
(either a constant scalar or a callable `k_imp(T_e)`) and optional ionization energy loss
`E_ion` deducted from the plasma energy.
"""
struct ElectronImpactIonization{TK, TE <: Union{Nothing, Number}} <:
    AbstractInteractionChannel
    k_imp::TK
    E_ion::TE
end

ElectronImpactIonization(k_imp; E_ion::Union{Nothing, Number} = nothing) =
    ElectronImpactIonization{typeof(k_imp), typeof(E_ion)}(k_imp, E_ion)

"""
    ResonantChargeExchange(sigma_cx) <: AbstractInteractionChannel

Resonant charge-exchange channel ``A^+ + A \\to A + A^+`` with cross section `sigma_cx`
(scalar or callable `sigma_cx(v_eff)`). Conserves number (`dn = 0`), momentum, and total
energy exactly between plasma and neutral species.
"""
struct ResonantChargeExchange{TS} <: AbstractInteractionChannel
    sigma_cx::TS
end

"""
    NonResonantChargeExchange(sigma_cx; E_defect = nothing) <: AbstractInteractionChannel

Non-resonant charge-exchange channel ``A^+ + B \\to A + B^+`` with cross section `sigma_cx`
(scalar or callable `sigma_cx(v_eff)`) and optional energy defect `E_defect`.
"""
struct NonResonantChargeExchange{TS, TE <: Union{Nothing, Number}} <:
    AbstractInteractionChannel
    sigma_cx::TS
    E_defect::TE
end

NonResonantChargeExchange(sigma_cx; E_defect::Union{Nothing, Number} = nothing) =
    NonResonantChargeExchange{typeof(sigma_cx), typeof(E_defect)}(sigma_cx, E_defect)

"""
    NonResonantExchange(ion_loss, ion_gain, neutral_loss, neutral_gain)

Four-species conservation-law source bundle for ``A^+ + B \\to A + B^+``:
- `ion_loss`: loss from reactant ion ``A^+``
- `ion_gain`: production of product ion ``B^+``
- `neutral_loss`: loss from reactant neutral ``B``
- `neutral_gain`: production of product neutral ``A``
"""
struct NonResonantExchange{
        SIL <: MomentSource, SIG <: MomentSource, SNL <: MomentSource, SNG <: MomentSource,
    }
    ion_loss::SIL
    ion_gain::SIG
    neutral_loss::SNL
    neutral_gain::SNG
end

"""
    ElasticCollision(coeff; is_cross_section = true) <: AbstractInteractionChannel

Elastic ion–neutral momentum- and energy-transfer collision channel (Schunk & Nagy 2009)
for arbitrary ion/neutral mass ratio ``m_i / m_n``:
- `is_cross_section = true`: `coeff` is the momentum-transfer cross section
  ``\\sigma_{mt}`` (scalar or callable `coeff(v_eff)`), with symmetric rate factor
  ``C = \\frac{m_i m_n}{m_i + m_n} n_i n_n \\sigma_{mt} v_{\\text{eff}}``.
- `is_cross_section = false`: `coeff` is a collision-frequency coefficient ``k_{in}``
  (scalar or callable `coeff(T_eff)` such that ``\\nu_{in} = k_{in} n_n`` and
  ``C = m_i n_i n_n k_{in}``).
"""
struct ElasticCollision{TC} <: AbstractInteractionChannel
    coeff::TC
    is_cross_section::Bool
end

ElasticCollision(coeff; is_cross_section::Bool = true) =
    ElasticCollision{typeof(coeff)}(coeff, is_cross_section)

"""
    Recombination(alpha_rec; E_rad = nothing) <: AbstractInteractionChannel

Ion–electron recombination channel ``I^+ + e^- \\to N + h\\nu`` with recombination rate
coefficient `alpha_rec` (scalar or callable `alpha_rec(T_e)`) and optional radiated energy
loss `E_rad`.
"""
struct Recombination{TA, TE <: Union{Nothing, Number}} <: AbstractInteractionChannel
    alpha_rec::TA
    E_rad::TE
end

Recombination(alpha_rec; E_rad::Union{Nothing, Number} = nothing) =
    Recombination{typeof(alpha_rec), typeof(E_rad)}(alpha_rec, E_rad)

# ------------------------------------------------------------------------------
# Conservative Volumetric Exchange Evaluation (`compute_exchange`)
# ------------------------------------------------------------------------------

"""
    compute_exchange(channel, plasma::PlasmaState, neutral::NeutralState, m_n;
                     atten = 1, n_e = plasma.n, T_e = plasma.T, kB = 1) -> CoupledExchange

Evaluate paired conservative plasma and neutral volumetric [`MomentSource`](@ref) terms for
one [`AbstractInteractionChannel`](@ref) or a `Tuple` of channels.
"""
@inline @muladd function compute_exchange(
        ch::Photoionization,
        ::PlasmaState,
        neutral::NeutralState,
        m_n::Number;
        atten::Number = 1,
        n_e::Number = 0,
        T_e::Number = 0,
        kB::Number = 1,
    )
    R = photoionization_rate(ch.nu_photo, neutral.n, atten)
    u2 = neutral.u[1] * neutral.u[1] + neutral.u[2] * neutral.u[2] +
        neutral.u[3] * neutral.u[3]
    half = oftype(float(one(real(typeof(atten / atten)))), 0.5)
    three_halves = oftype(half, 1.5)
    e_n = R * (half * m_n * u2 + three_halves * kB * neutral.T)
    dmom_n = (m_n * R) * neutral.u
    e_p = _add_energy_shift(e_n, ch.E_excess, R)
    return CoupledExchange(
        MomentSource(R, dmom_n, e_p),
        MomentSource(-R, -dmom_n, -e_n),
    )
end

@inline @muladd function compute_exchange(
        ch::ElectronImpactIonization,
        plasma::PlasmaState,
        neutral::NeutralState,
        m_n::Number;
        atten::Number = 1,
        n_e::Number = plasma.n,
        T_e::Number = plasma.T,
        kB::Number = 1,
    )
    k = _eval_rate_coeff(ch.k_imp, T_e)
    R = k * n_e * neutral.n
    u2 = neutral.u[1] * neutral.u[1] + neutral.u[2] * neutral.u[2] +
        neutral.u[3] * neutral.u[3]
    half = oftype(float(one(real(typeof(R / R)))), 0.5)
    three_halves = oftype(half, 1.5)
    e_n = R * (half * m_n * u2 + three_halves * kB * neutral.T)
    dmom = (m_n * R) * neutral.u
    e_p = _sub_energy_shift(e_n, ch.E_ion, R)
    return CoupledExchange(
        MomentSource(R, dmom, e_p),
        MomentSource(-R, -dmom, -e_n),
    )
end

@inline @muladd function compute_exchange(
        ch::ResonantChargeExchange,
        plasma::PlasmaState,
        neutral::NeutralState,
        m_n::Number;
        atten::Number = 1,
        n_e::Number = plasma.n,
        T_e::Number = plasma.T,
        kB::Number = 1,
    )
    v_eff = effective_relative_speed(plasma, neutral, m_n; kB = kB)
    σ = _eval_rate_coeff(ch.sigma_cx, v_eff)
    R = charge_exchange_rate(σ, plasma.n, neutral.n, v_eff)
    ui2 = plasma.u[1] * plasma.u[1] + plasma.u[2] * plasma.u[2] + plasma.u[3] * plasma.u[3]
    un2 = neutral.u[1] * neutral.u[1] + neutral.u[2] * neutral.u[2] +
        neutral.u[3] * neutral.u[3]
    half = oftype(float(one(real(typeof(v_eff / v_eff)))), 0.5)
    three_halves = oftype(half, 1.5)
    dmom_i = (m_n * R) * (neutral.u - plasma.u)
    dE_i = R * (half * m_n * (un2 - ui2) + three_halves * kB * (neutral.T - plasma.T))
    p_src = MomentSource(zero(R), dmom_i, dE_i)
    return CoupledExchange(p_src, -p_src)
end

"""
    compute_nonresonant_exchange(ch::NonResonantChargeExchange, plasma::PlasmaState,
                                 neutral::NeutralState, m_n; kB = 1) -> NonResonantExchange

Compute the four-species conservative source terms for ``A^+ + B \\to A + B^+``.
"""
@inline @muladd function compute_nonresonant_exchange(
        ch::NonResonantChargeExchange,
        plasma::PlasmaState,
        neutral::NeutralState,
        m_n::Number;
        kB::Number = 1,
    )
    v_eff = effective_relative_speed(plasma, neutral, m_n; kB = kB)
    σ = _eval_rate_coeff(ch.sigma_cx, v_eff)
    R = charge_exchange_rate(σ, plasma.n, neutral.n, v_eff)
    ui2 = plasma.u[1] * plasma.u[1] + plasma.u[2] * plasma.u[2] + plasma.u[3] * plasma.u[3]
    un2 = neutral.u[1] * neutral.u[1] + neutral.u[2] * neutral.u[2] +
        neutral.u[3] * neutral.u[3]
    half = oftype(float(one(real(typeof(v_eff / v_eff)))), 0.5)
    three_halves = oftype(half, 1.5)
    mom_A = (plasma.m * R) * plasma.u
    e_A = R * (half * plasma.m * ui2 + three_halves * kB * plasma.T)
    mom_B = (m_n * R) * neutral.u
    e_B = R * (half * m_n * un2 + three_halves * kB * neutral.T)
    e_B_ion = _add_energy_shift(e_B, ch.E_defect, R)
    return NonResonantExchange(
        MomentSource(-R, -mom_A, -e_A),
        MomentSource(R, mom_B, e_B_ion),
        MomentSource(-R, -mom_B, -e_B),
        MomentSource(R, mom_A, e_A),
    )
end

@inline function compute_exchange(
        ch::NonResonantChargeExchange,
        plasma::PlasmaState,
        neutral::NeutralState,
        m_n::Number;
        atten::Number = 1,
        n_e::Number = plasma.n,
        T_e::Number = plasma.T,
        kB::Number = 1,
    )
    nr = compute_nonresonant_exchange(ch, plasma, neutral, m_n; kB = kB)
    return CoupledExchange(nr.ion_loss + nr.ion_gain, nr.neutral_loss + nr.neutral_gain)
end

@inline @muladd function compute_exchange(
        ch::ElasticCollision,
        plasma::PlasmaState,
        neutral::NeutralState,
        m_n::Number;
        atten::Number = 1,
        n_e::Number = plasma.n,
        T_e::Number = plasma.T,
        kB::Number = 1,
    )
    m_i = plasma.m
    m_tot = m_i + m_n
    C = if ch.is_cross_section
        v_eff = effective_relative_speed(plasma, neutral, m_n; kB = kB)
        σ = _eval_rate_coeff(ch.coeff, v_eff)
        ((m_i * m_n) / m_tot) * plasma.n * neutral.n * σ * v_eff
    else
        T_eff = (m_n * plasma.T + m_i * neutral.T) / m_tot
        k_in = _eval_rate_coeff(ch.coeff, T_eff)
        m_i * plasma.n * neutral.n * k_in
    end
    du = neutral.u - plasma.u
    dmom_i = C * du
    u_cm = (m_i * plasma.u + m_n * neutral.u) / m_tot
    work = u_cm[1] * du[1] + u_cm[2] * du[2] + u_cm[3] * du[3]
    dE_i = C * ((3 * kB * (neutral.T - plasma.T)) / m_tot + work)
    zero_dn = zero(C / m_tot)
    p_src = MomentSource(zero_dn, dmom_i, dE_i)
    return CoupledExchange(p_src, -p_src)
end

@inline @muladd function compute_exchange(
        ch::Recombination,
        plasma::PlasmaState,
        ::NeutralState,
        ::Number;
        atten::Number = 1,
        n_e::Number = plasma.n,
        T_e::Number = plasma.T,
        kB::Number = 1,
    )
    α = _eval_rate_coeff(ch.alpha_rec, T_e)
    R = α * plasma.n * n_e
    u2 = plasma.u[1] * plasma.u[1] + plasma.u[2] * plasma.u[2] + plasma.u[3] * plasma.u[3]
    half = oftype(float(one(real(typeof(R / R)))), 0.5)
    three_halves = oftype(half, 1.5)
    dmom = (plasma.m * R) * plasma.u
    e_p = R * (half * plasma.m * u2 + three_halves * kB * plasma.T)
    e_n = _sub_energy_shift(e_p, ch.E_rad, R)
    return CoupledExchange(
        MomentSource(-R, -dmom, -e_p),
        MomentSource(R, dmom, e_n),
    )
end

# Tuple-unrolled evaluation over multiple interaction channels (0 allocations)
@inline function compute_exchange(
        channels::Tuple{C},
        plasma::PlasmaState,
        neutral::NeutralState,
        m_n::Number;
        atten::Number = 1,
        n_e::Number = plasma.n,
        T_e::Number = plasma.T,
        kB::Number = 1,
    ) where {C <: AbstractInteractionChannel}
    return compute_exchange(
        channels[1], plasma, neutral, m_n; atten = atten, n_e = n_e, T_e = T_e, kB = kB,
    )
end

@inline function compute_exchange(
        channels::Tuple{C, Vararg{AbstractInteractionChannel}},
        plasma::PlasmaState,
        neutral::NeutralState,
        m_n::Number;
        atten::Number = 1,
        n_e::Number = plasma.n,
        T_e::Number = plasma.T,
        kB::Number = 1,
    ) where {C <: AbstractInteractionChannel}
    e1 = compute_exchange(
        channels[1], plasma, neutral, m_n; atten = atten, n_e = n_e, T_e = T_e, kB = kB,
    )
    e2 = compute_exchange(
        Base.tail(channels), plasma, neutral, m_n; atten = atten, n_e = n_e, T_e = T_e,
        kB = kB,
    )
    return e1 + e2
end

# ------------------------------------------------------------------------------
# Unified Trait-Dispatched Coupling Entry Point (`couple_point!`)
# ------------------------------------------------------------------------------

"""
    couple_point!(model::AbstractNeutralModel, loc, channels, plasma::PlasmaState;
                  x = loc, t = nothing, species = 1, n_e = plasma.n, T_e = plasma.T,
                  kB = 1) -> MomentSource

Query `model` at location `loc` (spatial 3-vector or grid `CartesianIndex`), evaluate
`channels` (one [`AbstractInteractionChannel`](@ref) or a `Tuple` of channels), deposit the
conservative neutral feedback into `model` when `coupling_mode(model) === TwoWayCoupling()`,
and return the plasma [`MomentSource`](@ref).

For [`OneWayCoupling`](@ref) models, feedback deposition compiles away completely (zero
extra instructions and zero heap allocations), and EUV attenuation is skipped at compile
time when `channels` contains no [`Photoionization`](@ref).
"""
@inline function couple_point!(
        model::AbstractNeutralModel,
        loc,
        channels::Union{AbstractInteractionChannel, Tuple{Vararg{AbstractInteractionChannel}}},
        plasma::PlasmaState;
        x = loc,
        t = nothing,
        species::Integer = 1,
        n_e::Number = plasma.n,
        T_e::Number = plasma.T,
        kB::Number = 1,
    )
    ns = neutral_state(model, loc; t = t, species = species, kB = kB)
    m_n = species_mass(model, species)
    att = _needs_attenuation(typeof(channels)) ?
        photo_attenuation(model, x; t = t) : one(ns.n / one(ns.n))
    ex = compute_exchange(
        channels, plasma, ns, m_n; atten = att, n_e = n_e, T_e = T_e, kB = kB,
    )
    deposit_neutral_source!(model, loc, ex.neutral, species)
    return ex.plasma
end

# ------------------------------------------------------------------------------
# Kinetic / Hybrid-PIC / DSMC Interaction Kernels
# ------------------------------------------------------------------------------

"""
    collision_probability(rate::Number, dt::Number)

Exact Poisson collision probability ``1 - \\exp(-\\text{rate} \\cdot \\Delta t)`` evaluated
accurately via `-expm1(-(rate * dt))`.
"""
@inline function collision_probability(rate::Number, dt::Number)
    x = rate * dt
    return x <= zero(x) ? zero(float(x)) : -expm1(-x)
end

"""
    particle_charge_exchange_rate(sigma_cx, neutral_n::Number, v_rel::Number)

Per-ion charge-exchange frequency ``\\nu_{\\text{cx}} = \\sigma_{\\text{cx}}(v_{\\text{rel}}) n_n v_{\\text{rel}}``.
`sigma_cx` may be a scalar cross section or a speed-dependent callable `sigma_cx(v_rel)`.
"""
@inline function particle_charge_exchange_rate(sigma_cx, neutral_n::Number, v_rel::Number)
    return _eval_rate_coeff(sigma_cx, v_rel) * neutral_n * v_rel
end

"""
    photoionization_creation_weight(nu_photo, neutral_n, atten, cell_vol, dt)

Total number of real photoions created inside a cell of volume `cell_vol` over timestep
`dt`: ``\\Delta N = \\nu_{\\text{photo}} n_n A V_{\\text{cell}} \\Delta t``.
"""
@inline function photoionization_creation_weight(
        nu_photo::Number, neutral_n::Number, atten::Number, cell_vol::Number, dt::Number,
    )
    return photoionization_rate(nu_photo, neutral_n, atten) * cell_vol * dt
end

"""
    charge_exchange_collision(pos, v_ion, m_ion, weight, v_neutral, species_id = Int32(1))

Perform a single resonant macroparticle charge-exchange event ``I^+ + N \\to N_{\\text{ENA}} + I^+_{\\text{new}}``.
Returns a named tuple `(ion_vel, ena, neutral_delta)` containing:
- `ion_vel::SVector{3}`: replacement ion velocity (`v_neutral`)
- `ena::NeutralParticle`: emitted energetic neutral atom carrying the incident ion velocity
- `neutral_delta::MomentSource`: exact integrated momentum and kinetic-energy delta
  transferred to the neutral population (`dn = 0`, exact conservation with the ion change).
"""
@inline @muladd function charge_exchange_collision(
        pos::SVector{3, <:Real},
        v_ion::SVector{3, <:Real},
        m_ion::Real,
        weight::Real,
        v_neutral::SVector{3, <:Real},
        species_id::Integer = Int32(1),
    )
    FT = float(
        promote_type(eltype(pos), eltype(v_ion), eltype(v_neutral), typeof(m_ion), typeof(weight)),
    )
    vi = SVector{3, FT}(FT(v_ion[1]), FT(v_ion[2]), FT(v_ion[3]))
    vn = SVector{3, FT}(FT(v_neutral[1]), FT(v_neutral[2]), FT(v_neutral[3]))
    mw = FT(m_ion) * FT(weight)
    vi2 = vi[1] * vi[1] + vi[2] * vi[2] + vi[3] * vi[3]
    vn2 = vn[1] * vn[1] + vn[2] * vn[2] + vn[3] * vn[3]
    dmom = mw * (vi - vn)
    dE = FT(0.5) * mw * (vi2 - vn2)
    ena = NeutralParticle{FT}(
        SVector{3, FT}(FT(pos[1]), FT(pos[2]), FT(pos[3])),
        vi, FT(m_ion), FT(weight), Int32(species_id),
    )
    return (ion_vel = vn, ena = ena, neutral_delta = MomentSource(zero(FT), dmom, dE))
end

"""
    sputtering_flux(ion_flux::Number, yield::Number)

Sputtered neutral surface flux ``\\Phi_n = Y \\, \\Phi_i`` produced by precipitating ion
flux `ion_flux` with sputtering yield `yield`.
"""
@inline sputtering_flux(ion_flux::Number, yield::Number) = ion_flux * yield
