# ==============================================================================
# Neutral Exosphere Density Profiles (Allocation-Free / GPU-Friendly)
# ==============================================================================

const _EXO_EXPONENTIAL = 1
const _EXO_POWERLAW = 2
const _EXO_CHAMBERLAIN = 3

"""
    AbstractNeutralProfile

Abstract supertype for radial planetary/cometary neutral density profiles.
"""
abstract type AbstractNeutralProfile end

"""
    ExponentialProfile{FT<:AbstractFloat} <: AbstractNeutralProfile

Exponential neutral exosphere `n(r) = n0 * exp(-max(r - r0, 0) / H)` around `center`.
"""
struct ExponentialProfile{FT <: AbstractFloat} <: AbstractNeutralProfile
    n0::FT
    r0::FT
    inv_H::FT
    center::SVector{3, FT}
end

function ExponentialProfile(
        n0::Real, r0::Real, H::Real;
        center::NTuple{3, Real} = (0.0, 0.0, 0.0),
        FT::Type{<:AbstractFloat} = Float64,
    )
    (isfinite(n0) && n0 >= 0) || throw(ArgumentError("ExponentialProfile requires finite n0 >= 0"))
    (isfinite(r0) && r0 > 0) || throw(ArgumentError("ExponentialProfile requires finite r0 > 0"))
    H > 0 || throw(ArgumentError("ExponentialProfile requires H > 0"))
    inv_H = isinf(H) ? zero(FT) : inv(FT(H))
    c = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    return ExponentialProfile{FT}(FT(n0), FT(r0), inv_H, c)
end

@inline function neutral_density(p::ExponentialProfile{FT}, r::Real) where {FT}
    h = max(FT(r) - p.r0, zero(FT))
    return p.n0 * exp(-h * p.inv_H)
end

@inline local_scale_height(p::ExponentialProfile{FT}, ::Real) where {FT} =
    p.inv_H > zero(FT) ? inv(p.inv_H) : zero(FT)

"""
    PowerLawProfile{FT<:AbstractFloat} <: AbstractNeutralProfile

Power-law neutral density profile `n(r) = n0 * (r0 / max(r, r0))^k` around `center`.
"""
struct PowerLawProfile{FT <: AbstractFloat} <: AbstractNeutralProfile
    n0::FT
    r0::FT
    k::FT
    center::SVector{3, FT}
end

function PowerLawProfile(
        n0::Real, r0::Real, k::Real;
        center::NTuple{3, Real} = (0.0, 0.0, 0.0),
        FT::Type{<:AbstractFloat} = Float64,
    )
    (isfinite(n0) && n0 >= 0) || throw(ArgumentError("PowerLawProfile requires finite n0 >= 0"))
    (isfinite(r0) && r0 > 0) || throw(ArgumentError("PowerLawProfile requires finite r0 > 0"))
    (isfinite(k) && k >= 0) || throw(ArgumentError("PowerLawProfile requires finite k >= 0"))
    c = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    return PowerLawProfile{FT}(FT(n0), FT(r0), FT(k), c)
end

@inline function neutral_density(p::PowerLawProfile{FT}, r::Real) where {FT}
    rf = FT(r)
    return rf > p.r0 ? p.n0 * (p.r0 / rf)^p.k : p.n0
end

@inline local_scale_height(p::PowerLawProfile{FT}, r::Real) where {FT} =
    (p.k > zero(FT) && FT(r) > zero(FT)) ? FT(r) / p.k : zero(FT)

"""
    ChamberlainProfile{FT<:AbstractFloat} <: AbstractNeutralProfile

Chamberlain-type hydrostatic exosphere `n(r) = n0 * exp(-H * (1/r0 - 1/r))` for `r >= r0`.
"""
struct ChamberlainProfile{FT <: AbstractFloat} <: AbstractNeutralProfile
    n0::FT
    r0::FT
    H::FT
    center::SVector{3, FT}
end

function ChamberlainProfile(
        n0::Real, r0::Real, H::Real;
        center::NTuple{3, Real} = (0.0, 0.0, 0.0),
        FT::Type{<:AbstractFloat} = Float64,
    )
    (isfinite(n0) && n0 >= 0) || throw(ArgumentError("ChamberlainProfile requires finite n0 >= 0"))
    (isfinite(r0) && r0 > 0) || throw(ArgumentError("ChamberlainProfile requires finite r0 > 0"))
    (isfinite(H) && H > 0) || throw(ArgumentError("ChamberlainProfile requires finite H > 0"))
    c = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    return ChamberlainProfile{FT}(FT(n0), FT(r0), FT(H), c)
end

@inline function neutral_density(p::ChamberlainProfile{FT}, r::Real) where {FT}
    rf = FT(r)
    return rf > p.r0 ? p.n0 * exp(-p.H * (inv(p.r0) - inv(rf))) : p.n0
end

@inline local_scale_height(p::ChamberlainProfile{FT}, r::Real) where {FT} =
    (p.H > zero(FT) && FT(r) > zero(FT)) ? (FT(r) * FT(r)) / p.H : zero(FT)

"""
    ExosphereProfile(n0, r0, inv_H, center, power)

Concrete, allocation-free radial neutral density/rate profile for planetary exospheres and
cometary comae:
`p(x) = n0 * (r0 / max(r, r0))^power * exp(-max(r - r0, 0) * inv_H)`.
Callable on 1D `(x,)`, 2D `(x, y)`, and 3D `(x, y, z)` coordinates; construct via
[`exosphere_profile`](@ref).
"""
struct ExosphereProfile <: AbstractNeutralProfile
    n0::Float64
    r0::Float64
    inv_H::Float64
    center::NTuple{3, Float64}
    power::Float64
end

"""
    exosphere_profile(; n0 = 1.0, r0 = 1.0, H = 0.5, center = (0.0, 0.0, 0.0), power = 0.0) -> ExosphereProfile

Construct a radial neutral envelope profile:
- `power = 0`, `H > 0`: exponential planetary exosphere `n0 * exp(-max(r - r0, 0) / H)`.
- `power = 2`, `H = Inf`: cometary `1 / r^2` Haser-limit coma.
"""
function exosphere_profile(;
        n0::Real = 1.0,
        r0::Real = 1.0,
        H::Real = 0.5,
        center::NTuple{3, Real} = (0.0, 0.0, 0.0),
        power::Real = 0.0,
    )
    n0_f = Float64(n0)
    r0_f = Float64(r0)
    H_f = Float64(H)
    p_f = Float64(power)
    c_f = (Float64(center[1]), Float64(center[2]), Float64(center[3]))
    (isfinite(n0_f) && n0_f >= 0.0) || throw(ArgumentError("exosphere_profile requires finite n0 >= 0 (got $n0)"))
    (isfinite(r0_f) && r0_f > 0.0) || throw(ArgumentError("exosphere_profile requires finite r0 > 0 (got $r0)"))
    (H_f > 0.0) || throw(ArgumentError("exosphere_profile requires H > 0 (got $H)"))
    (isfinite(p_f) && p_f >= 0.0) || throw(ArgumentError("exosphere_profile requires finite power >= 0 (got $power)"))
    all(isfinite, c_f) || throw(ArgumentError("exosphere_profile requires finite center (got $center)"))
    inv_H = isinf(H_f) ? 0.0 : inv(H_f)
    return ExosphereProfile(n0_f, r0_f, inv_H, c_f, p_f)
end

@inline function neutral_density(p::ExosphereProfile, r::Real)
    rf = Float64(r)
    val = p.n0
    if p.power > 0.0 && rf > p.r0
        val *= (p.r0 / rf)^p.power
    end
    if p.inv_H > 0.0 && rf > p.r0
        val *= exp(-(rf - p.r0) * p.inv_H)
    end
    return val
end

@inline (p::ExosphereProfile)(x::Real) = neutral_density(p, abs(Float64(x) - p.center[1]))
@inline (p::ExosphereProfile)(x::Real, y::Real) =
    neutral_density(p, hypot(Float64(x) - p.center[1], Float64(y) - p.center[2]))
@inline (p::ExosphereProfile)(x::Real, y::Real, z::Real) =
    neutral_density(p, hypot(Float64(x) - p.center[1], Float64(y) - p.center[2], Float64(z) - p.center[3]))

"""
    NeutralComponentCode(kind, n0, H, k, T0, name)

Code-unit specification of one neutral exosphere component (`kind` in
`1 = :exponential`, `2 = :powerlaw`, `3 = :chamberlain`).
"""
struct NeutralComponentCode
    kind::Int
    n0::Float64
    H::Float64
    k::Float64
    T0::Float64
    name::String
end

function NeutralComponentCode(
        profile::Symbol, n0::Real, H::Real, k::Real, T0::Real, name::AbstractString = "neutral"
    )
    kind = profile === :exponential ? _EXO_EXPONENTIAL :
        profile === :powerlaw ? _EXO_POWERLAW :
        profile === :chamberlain ? _EXO_CHAMBERLAIN :
        throw(ArgumentError("NeutralComponentCode: unknown profile :$profile"))
    Hf = Float64(H)
    (kind != _EXO_POWERLAW && Hf <= 0.0) &&
        throw(ArgumentError("NeutralComponentCode $name: profile :$profile needs H > 0"))
    T0f = Float64(T0)
    T0f > 0.0 || throw(ArgumentError("NeutralComponentCode $name: T0 must be positive"))
    return NeutralComponentCode(kind, Float64(n0), Hf, Float64(k), T0f, String(name))
end
