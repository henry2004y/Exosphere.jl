# ==============================================================================
# Neutral Velocity Distribution Functions (PlasmaPopulations.jl Style)
# ==============================================================================

Base.broadcastable(d::AbstractNeutralVDF) = Ref(d)

@inline function _to_svec3(u::SVector{3, <:Number})
    return SVector{3}(si_velocity(u[1]), si_velocity(u[2]), si_velocity(u[3]))
end

@inline function _to_svec3(u::NTuple{3, Number})
    return SVector{3}(si_velocity(u[1]), si_velocity(u[2]), si_velocity(u[3]))
end

@inline function _to_svec3(u::AbstractVector{<:Number})
    length(u) == 3 || throw(ArgumentError("Velocity vector must have length 3"))
    return SVector{3}(si_velocity(u[1]), si_velocity(u[2]), si_velocity(u[3]))
end

"""
    thermal_speed(species::NeutralSpecies, T; kB = 1)
    thermal_speed(vdf::AbstractNeutralVDF)

Compute the most probable thermal speed ``v_{\\text{th}} = \\sqrt{2 k_B T / m}`` for
`species` at temperature `T`, or extract the thermal speed of a velocity distribution
`vdf` (matching the `PlasmaPopulations.jl` convention ``T = m v_{\\text{th}}^2 / (2 k_B)``).
"""
@inline function thermal_speed(species::NeutralSpecies, T::Number; kB::Number = 1)
    T >= zero(T) || throw(ArgumentError("thermal_speed requires T >= 0"))
    return sqrt((2 * kB * T) / species.m)
end

"""
    vdf_temperature(species::NeutralSpecies, vdf::AbstractNeutralVDF; kB = 1)

Return the isotropic (or effective scalar ``(2 T_\\perp + T_\\parallel)/3``) temperature
corresponding to `vdf` for `species` using ``T = m v_{\\text{th}}^2 / (2 k_B)``.
"""
function vdf_temperature end

"""
    temperature_perp_para(species::NeutralSpecies, vdf::AbstractNeutralVDF; kB = 1)

Return `(T_perp, T_para)` for `species` with velocity distribution `vdf`.
"""
function temperature_perp_para end

"""
    pdf(vdf::AbstractNeutralVDF, v)
    pdf(vdf::AbstractNeutralVDF)

Evaluate the normalized 3D velocity probability density ``f(\\mathbf{v})`` satisfying
``\\int f(\\mathbf{v})\\,d^3v = 1`` at 3-vector `v`, or return a callable `v -> pdf(vdf, v)`.
"""
function pdf end

struct _VDFPDF{D <: AbstractNeutralVDF} <: Function
    d::D
end

@inline pdf(d::AbstractNeutralVDF) = _VDFPDF(d)
@inline (f::_VDFPDF)(v) = pdf(f.d, v)

@inline Random.Sampler(
    ::Type{<:AbstractRNG}, d::AbstractNeutralVDF, ::Random.Repetition,
) = Random.SamplerTrivial(d)

# ------------------------------------------------------------------------------
# ColdNeutral (Zero-Temperature Mono-Energetic Drift)
# ------------------------------------------------------------------------------

"""
    ColdNeutral{T<:Real} <: AbstractNeutralVDF
    ColdNeutral(; u = SVector(0.0, 0.0, 0.0))
    ColdNeutral(species::NeutralSpecies; u = SVector(0.0, 0.0, 0.0))

Zero-temperature drifting neutral distribution ``f(\\mathbf{v}) = \\delta(\\mathbf{v} - \\mathbf{u})``.
"""
struct ColdNeutral{T <: Real} <: AbstractNeutralVDF
    u::SVector{3, T}
end

function ColdNeutral(;
        u::Union{NTuple{3, Number}, SVector{3, <:Number}, AbstractVector{<:Number}} = (
            0.0, 0.0, 0.0,
        ),
    )
    uv = _to_svec3(u)
    T = float(promote_type(typeof(uv[1]), typeof(uv[2]), typeof(uv[3])))
    return ColdNeutral{T}(SVector{3, T}(T(uv[1]), T(uv[2]), T(uv[3])))
end

function ColdNeutral(
        s::NeutralSpecies{TS};
        u::Union{NTuple{3, Number}, SVector{3, <:Number}, AbstractVector{<:Number}} = (
            zero(TS), zero(TS), zero(TS),
        ),
    ) where {TS}
    uv = _to_svec3(u)
    T = float(promote_type(TS, typeof(uv[1]), typeof(uv[2]), typeof(uv[3])))
    return ColdNeutral{T}(SVector{3, T}(T(uv[1]), T(uv[2]), T(uv[3])))
end

@inline thermal_speed(::ColdNeutral{T}) where {T} = zero(T)
@inline vdf_temperature(s::NeutralSpecies, d::ColdNeutral; kB::Number = 1) =
    zero(s.m * d.u[1] * d.u[1] / kB)
@inline function temperature_perp_para(s::NeutralSpecies, d::ColdNeutral; kB::Number = 1)
    t0 = vdf_temperature(s, d; kB = kB)
    return (t0, t0)
end

@inline Statistics.mean(d::ColdNeutral) = d.u
@inline Statistics.cov(::ColdNeutral{T}) where {T} = zero(SMatrix{3, 3, T})

@inline function pdf(d::ColdNeutral{T}, v::Union{NTuple{3, Real}, AbstractVector{<:Real}}) where {T}
    vv = _to_svec3(v)
    return vv == d.u ? T(Inf) : zero(T)
end

@inline Random.rand(::AbstractRNG, sp::Random.SamplerTrivial{<:ColdNeutral}) = sp[].u

# ------------------------------------------------------------------------------
# Maxwellian (Isotropic 3D Drifting Maxwellian)
# ------------------------------------------------------------------------------

"""
    Maxwellian{T<:Real} <: AbstractNeutralVDF
    Maxwellian(; vth, u = (0.0, 0.0, 0.0))
    Maxwellian(species::NeutralSpecies; T, u = (0.0, 0.0, 0.0), kB = 1)

Isotropic 3D drifting Maxwellian neutral velocity distribution:

```math
f(\\mathbf{v}) = \\frac{1}{(\\pi v_{\\text{th}}^2)^{3/2}}
\\exp\\!\\left(-\\frac{|\\mathbf{v} - \\mathbf{u}|^2}{v_{\\text{th}}^2}\\right),
\\qquad v_{\\text{th}} = \\sqrt{\\frac{2 k_B T}{m}}.
```
"""
struct Maxwellian{T <: Real} <: AbstractNeutralVDF
    vth::T
    u::SVector{3, T}
    function Maxwellian{T}(vth::T, u::SVector{3, T}) where {T <: Real}
        (isfinite(vth) && vth > zero(T)) ||
            throw(ArgumentError("Maxwellian requires finite vth > 0 (got $vth)"))
        return new{T}(vth, u)
    end
end

function Maxwellian(;
        vth::Number,
        u::Union{NTuple{3, Number}, SVector{3, <:Number}, AbstractVector{<:Number}} = (
            0.0, 0.0, 0.0,
        ),
    )
    vt = si_velocity(vth)
    uv = _to_svec3(u)
    T = float(promote_type(typeof(vt), typeof(uv[1]), typeof(uv[2]), typeof(uv[3])))
    return Maxwellian{T}(T(vt), SVector{3, T}(T(uv[1]), T(uv[2]), T(uv[3])))
end

function Maxwellian(
        species::NeutralSpecies;
        T::Number,
        u::Union{NTuple{3, Number}, SVector{3, <:Number}, AbstractVector{<:Number}} = (
            0.0, 0.0, 0.0,
        ),
        kB::Number = 1,
    )
    vth = thermal_speed(species, T; kB = kB)
    return Maxwellian(; vth = vth, u = u)
end

@inline thermal_speed(d::Maxwellian) = d.vth

@inline vdf_temperature(s::NeutralSpecies, d::Maxwellian; kB::Number = 1) =
    (s.m * (d.vth * d.vth)) / (2 * kB)

@inline function temperature_perp_para(s::NeutralSpecies, d::Maxwellian; kB::Number = 1)
    t0 = vdf_temperature(s, d; kB = kB)
    return (t0, t0)
end

@inline Statistics.mean(d::Maxwellian) = d.u

@inline function Statistics.cov(d::Maxwellian{T}) where {T}
    s2 = T(0.5) * d.vth * d.vth
    z = zero(s2)
    return SMatrix{3, 3, T}(s2, z, z, z, s2, z, z, z, s2)
end

@inline @muladd function pdf(
        d::Maxwellian, v::Union{NTuple{3, Real}, AbstractVector{<:Real}},
    )
    vv = _to_svec3(v)
    dv1 = vv[1] - d.u[1]
    dv2 = vv[2] - d.u[2]
    dv3 = vv[3] - d.u[3]
    inv_vth2 = inv(d.vth * d.vth)
    dv2_sum = dv1 * dv1 + dv2 * dv2 + dv3 * dv3
    FT = float(typeof(dv2_sum * inv_vth2))
    norm_f = inv(FT(π) * d.vth * d.vth)
    return norm_f * sqrt(norm_f) * exp(-dv2_sum * inv_vth2)
end

@inline function Random.rand(
        rng::AbstractRNG, sp::Random.SamplerTrivial{Maxwellian{T}},
    ) where {T}
    d = sp[]
    FT = float(T)
    sigma = d.vth * inv(sqrt(FT(2)))
    return SVector{3, FT}(
        d.u[1] + sigma * randn(rng, FT),
        d.u[2] + sigma * randn(rng, FT),
        d.u[3] + sigma * randn(rng, FT),
    )
end

# ------------------------------------------------------------------------------
# BiMaxwellianNeutral (Axis-Aligned 3D Anisotropic Drifting Maxwellian)
# ------------------------------------------------------------------------------

"""
    BiMaxwellianNeutral{T<:Real} <: AbstractNeutralVDF
    BiMaxwellianNeutral(; vth_perp, vth_para, u = (0.0, 0.0, 0.0), axis = (0.0, 0.0, 1.0))
    BiMaxwellianNeutral(species::NeutralSpecies; T_perp, T_para, u = (0.0, 0.0, 0.0),
                        axis = (0.0, 0.0, 1.0), kB = 1)

Anisotropic 3D drifting Maxwellian neutral velocity distribution about unit direction `axis`
(e.g., local radial unit vector ``\\hat{\\mathbf{r}}`` for radial vs. tangential exospheric
temperatures):

```math
f(\\mathbf{v}) = \\frac{1}{\\pi^{3/2} v_{\\text{th}\\perp}^2 v_{\\text{th}\\parallel}}
\\exp\\!\\left(-\\frac{v_\\perp^2}{v_{\\text{th}\\perp}^2}
- \\frac{v_\\parallel^2}{v_{\\text{th}\\parallel}^2}\\right).
```
"""
struct BiMaxwellianNeutral{T <: Real} <: AbstractNeutralVDF
    vth_perp::T
    vth_para::T
    u::SVector{3, T}
    axis::SVector{3, T}
    function BiMaxwellianNeutral{T}(
            vth_perp::T, vth_para::T, u::SVector{3, T}, axis::SVector{3, T},
        ) where {T <: Real}
        (isfinite(vth_perp) && vth_perp > zero(T)) ||
            throw(ArgumentError("BiMaxwellianNeutral requires finite vth_perp > 0"))
        (isfinite(vth_para) && vth_para > zero(T)) ||
            throw(ArgumentError("BiMaxwellianNeutral requires finite vth_para > 0"))
        return new{T}(vth_perp, vth_para, u, axis)
    end
end

function BiMaxwellianNeutral(;
        vth_perp::Number,
        vth_para::Number,
        u::Union{NTuple{3, Number}, SVector{3, <:Number}, AbstractVector{<:Number}} = (
            0.0, 0.0, 0.0,
        ),
        axis::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            0.0, 0.0, 1.0,
        ),
    )
    vp = si_velocity(vth_perp)
    vl = si_velocity(vth_para)
    uv = _to_svec3(u)
    length(axis) == 3 || throw(ArgumentError("BiMaxwellianNeutral: axis must have length 3"))
    T = float(
        promote_type(
            typeof(vp), typeof(vl),
            typeof(uv[1]), typeof(uv[2]), typeof(uv[3]),
            typeof(axis[1]), typeof(axis[2]), typeof(axis[3]),
        )
    )
    ax = SVector{3, T}(T(axis[1]), T(axis[2]), T(axis[3]))
    an = norm(ax)
    an > zero(T) || throw(ArgumentError("BiMaxwellianNeutral: axis must be non-zero"))
    return BiMaxwellianNeutral{T}(
        T(vp), T(vl), SVector{3, T}(T(uv[1]), T(uv[2]), T(uv[3])), ax / an,
    )
end

function BiMaxwellianNeutral(
        species::NeutralSpecies;
        T_perp::Number,
        T_para::Number,
        u::Union{NTuple{3, Number}, SVector{3, <:Number}, AbstractVector{<:Number}} = (
            0.0, 0.0, 0.0,
        ),
        axis::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            0.0, 0.0, 1.0,
        ),
        kB::Number = 1,
    )
    vp = thermal_speed(species, T_perp; kB = kB)
    vl = thermal_speed(species, T_para; kB = kB)
    return BiMaxwellianNeutral(; vth_perp = vp, vth_para = vl, u = u, axis = axis)
end

@inline function thermal_speed(d::BiMaxwellianNeutral{T}) where {T}
    return sqrt((T(2) * d.vth_perp * d.vth_perp + d.vth_para * d.vth_para) / T(3))
end

@inline function temperature_perp_para(
        s::NeutralSpecies, d::BiMaxwellianNeutral; kB::Number = 1,
    )
    inv_2kB = inv(2 * kB)
    return (s.m * d.vth_perp * d.vth_perp * inv_2kB, s.m * d.vth_para * d.vth_para * inv_2kB)
end

@inline function vdf_temperature(
        s::NeutralSpecies, d::BiMaxwellianNeutral; kB::Number = 1,
    )
    tp, tl = temperature_perp_para(s, d; kB = kB)
    return (2 * tp + tl) / 3
end

@inline Statistics.mean(d::BiMaxwellianNeutral) = d.u

@inline function Statistics.cov(d::BiMaxwellianNeutral{T}) where {T}
    s_perp2 = T(0.5) * d.vth_perp * d.vth_perp
    s_para2 = T(0.5) * d.vth_para * d.vth_para
    ds = s_para2 - s_perp2
    a = d.axis
    return SMatrix{3, 3, T}(
        s_perp2 + ds * a[1] * a[1], ds * a[2] * a[1], ds * a[3] * a[1],
        ds * a[1] * a[2], s_perp2 + ds * a[2] * a[2], ds * a[3] * a[2],
        ds * a[1] * a[3], ds * a[2] * a[3], s_perp2 + ds * a[3] * a[3],
    )
end

@inline @muladd function pdf(
        d::BiMaxwellianNeutral, v::Union{NTuple{3, Real}, AbstractVector{<:Real}},
    )
    vv = _to_svec3(v)
    dv = SVector{3}(vv[1] - d.u[1], vv[2] - d.u[2], vv[3] - d.u[3])
    v_para = dv[1] * d.axis[1] + dv[2] * d.axis[2] + dv[3] * d.axis[3]
    dv2 = dv[1] * dv[1] + dv[2] * dv[2] + dv[3] * dv[3]
    v_perp2 = max(dv2 - v_para * v_para, zero(dv2))
    vp2 = d.vth_perp * d.vth_perp
    vl2 = d.vth_para * d.vth_para
    FT = float(typeof(v_perp2 / vp2))
    pref = inv(FT(π) * sqrt(FT(π)) * vp2 * d.vth_para)
    return pref * exp(-v_perp2 / vp2 - (v_para * v_para) / vl2)
end

@inline function _orthonormal_perp_pair(a::SVector{3, FT}) where {FT}
    # Pick coordinate axis least aligned with `a` to avoid near-collinear cross product.
    ref = abs(a[1]) < FT(0.9) ? SVector{3, FT}(one(FT), zero(FT), zero(FT)) :
        SVector{3, FT}(zero(FT), one(FT), zero(FT))
    e1_raw = cross(a, ref)
    e1 = e1_raw / norm(e1_raw)
    e2 = cross(a, e1)
    return e1, e2
end

@inline function Random.rand(
        rng::AbstractRNG, sp::Random.SamplerTrivial{BiMaxwellianNeutral{T}},
    ) where {T}
    d = sp[]
    FT = float(T)
    inv_s2 = inv(sqrt(FT(2)))
    sp_std = d.vth_perp * inv_s2
    sl_std = d.vth_para * inv_s2
    e1, e2 = _orthonormal_perp_pair(d.axis)
    g1 = sp_std * randn(rng, FT)
    g2 = sp_std * randn(rng, FT)
    g3 = sl_std * randn(rng, FT)
    return d.u + g1 * e1 + g2 * e2 + g3 * d.axis
end
