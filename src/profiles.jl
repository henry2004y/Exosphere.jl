# ==============================================================================
# Neutral Exosphere Density Profiles, Kinematics & Populations
# (Allocation-Free / GPU-Friendly / AD-Safe)
# ==============================================================================

const _EXO_EXPONENTIAL = 1
const _EXO_POWERLAW = 2
const _EXO_CHAMBERLAIN = 3

@inline function _fast_pow(base::Number, k::Real)
    k == 0 && return one(base)
    k == 1 && return base
    k == 2 && return base * base
    k == 3 && return base * base * base
    if k == 4
        b2 = base * base
        return b2 * b2
    end
    return base^k
end

@inline @muladd function _safe_norm(v::SVector{3})
    r2 = v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    return iszero(r2) ? zero(sqrt(one(r2))) : sqrt(r2)
end

@inline function _as_svec3(x::SVector{3})
    return x
end

@inline function _as_svec3(x::NTuple{3, Number})
    return SVector{3}(x[1], x[2], x[3])
end

@inline function _as_svec3(x::AbstractVector{<:Number})
    length(x) == 3 || throw(ArgumentError("Position vector must have length 3"))
    return SVector{3}(x[1], x[2], x[3])
end

@inline function _relative_pos(center::SVector{3}, x::SVector{3})
    return SVector{3}(x[1] - center[1], x[2] - center[2], x[3] - center[3])
end

@inline function _relative_pos(center::NTuple{3, Real}, x::SVector{3})
    return SVector{3}(x[1] - center[1], x[2] - center[2], x[3] - center[3])
end

Base.broadcastable(p::AbstractNeutralProfile) = Ref(p)
Base.broadcastable(v::AbstractVelocityModel) = Ref(v)
Base.broadcastable(t::AbstractTemperatureModel) = Ref(t)

# ------------------------------------------------------------------------------
# Single-Component Radial Profiles
# ------------------------------------------------------------------------------

"""
    ExponentialProfile{FT<:AbstractFloat} <: AbstractNeutralProfile
    ExponentialProfile(n0, r0, H; center = (0.0, 0.0, 0.0), FT = Float64)

Exponential neutral exosphere `n(r) = n0 * exp(-max(r - r0, 0) / H)` around `center`,
clamped to `n0` for `r <= r0`. Setting `H = Inf` yields a uniform density `n0`.
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
    (isfinite(n0) && n0 >= 0) ||
        throw(ArgumentError("ExponentialProfile requires finite n0 >= 0"))
    (isfinite(r0) && r0 > 0) ||
        throw(ArgumentError("ExponentialProfile requires finite r0 > 0"))
    H > 0 || throw(ArgumentError("ExponentialProfile requires H > 0"))
    inv_H = isinf(H) ? zero(FT) : inv(FT(H))
    c = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    return ExponentialProfile{FT}(FT(n0), FT(r0), inv_H, c)
end

"""
    neutral_density(p::AbstractNeutralProfile, r::Real)
    neutral_density(p::AbstractNeutralProfile, x::AbstractVector; t = nothing)
    neutral_density(model::AbstractNeutralModel, loc; t = nothing, species = nothing)

Evaluate the neutral number density of profile `p` or neutral `model` at radius `r`,
3D position `x`, or grid index `loc` and optional time `t`. Radial profiles clamp to `n0`
for `r <= r0`.
"""
@inline function neutral_density(p::ExponentialProfile{FT}, r::Real) where {FT}
    T = float(promote_type(FT, typeof(r)))
    h = r > p.r0 ? T(r) - p.r0 : zero(T)
    return T(p.n0) * exp(-h * p.inv_H)
end

"""
    local_scale_height(p::AbstractNeutralProfile, r::Real)
    local_scale_height(p::AbstractNeutralProfile, x::AbstractVector; t = nothing)
    local_scale_height(model::AbstractNeutralModel, loc; t = nothing, species = 1)

Return the local density scale height ``-1 / (d\\ln n / dr)`` of profile `p` or neutral
`model` (`H` for `ExponentialProfile`, `r / k` for `PowerLawProfile`, `r^2 / H` for
`ChamberlainProfile`; returns `0` when degenerate or `r <= 0`).
"""
@inline local_scale_height(p::ExponentialProfile{FT}, ::Real) where {FT} =
    p.inv_H > zero(FT) ? inv(p.inv_H) : zero(FT)

"""
    PowerLawProfile{FT<:AbstractFloat} <: AbstractNeutralProfile
    PowerLawProfile(n0, r0, k; center = (0.0, 0.0, 0.0), FT = Float64)

Power-law neutral density profile `n(r) = n0 * (r0 / max(r, r0))^k` around `center`,
clamped to `n0` for `r <= r0`.
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
    (isfinite(n0) && n0 >= 0) ||
        throw(ArgumentError("PowerLawProfile requires finite n0 >= 0"))
    (isfinite(r0) && r0 > 0) ||
        throw(ArgumentError("PowerLawProfile requires finite r0 > 0"))
    (isfinite(k) && k >= 0) ||
        throw(ArgumentError("PowerLawProfile requires finite k >= 0"))
    c = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    return PowerLawProfile{FT}(FT(n0), FT(r0), FT(k), c)
end

@inline function neutral_density(p::PowerLawProfile{FT}, r::Real) where {FT}
    T = float(promote_type(FT, typeof(r)))
    return r > p.r0 ? T(p.n0) * _fast_pow(p.r0 / T(r), p.k) : T(p.n0)
end

@inline function local_scale_height(p::PowerLawProfile{FT}, r::Real) where {FT}
    T = float(promote_type(FT, typeof(r)))
    return (p.k > zero(FT) && r > zero(r)) ? T(r) / p.k : zero(T)
end

"""
    ChamberlainProfile{FT<:AbstractFloat} <: AbstractNeutralProfile
    ChamberlainProfile(n0, r0, H; center = (0.0, 0.0, 0.0), FT = Float64)

Chamberlain-type hydrostatic exosphere `n(r) = n0 * exp(-H * (1/r0 - 1/r))` around `center`,
clamped to `n0` for `r <= r0`.
"""
struct ChamberlainProfile{FT <: AbstractFloat} <: AbstractNeutralProfile
    n0::FT
    r0::FT
    inv_r0::FT
    H::FT
    center::SVector{3, FT}
end

ChamberlainProfile{FT}(
    n0::FT, r0::FT, H::FT, center::SVector{3, FT},
) where {FT <: AbstractFloat} =
    ChamberlainProfile{FT}(n0, r0, inv(r0), H, center)

function ChamberlainProfile(
        n0::Real, r0::Real, H::Real;
        center::NTuple{3, Real} = (0.0, 0.0, 0.0),
        FT::Type{<:AbstractFloat} = Float64,
    )
    (isfinite(n0) && n0 >= 0) ||
        throw(ArgumentError("ChamberlainProfile requires finite n0 >= 0"))
    (isfinite(r0) && r0 > 0) ||
        throw(ArgumentError("ChamberlainProfile requires finite r0 > 0"))
    (isfinite(H) && H > 0) ||
        throw(ArgumentError("ChamberlainProfile requires finite H > 0"))
    r0_f = FT(r0)
    c = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    return ChamberlainProfile{FT}(FT(n0), r0_f, inv(r0_f), FT(H), c)
end

@inline @muladd function neutral_density(p::ChamberlainProfile{FT}, r::Real) where {FT}
    T = float(promote_type(FT, typeof(r)))
    return r > p.r0 ? T(p.n0) * exp(-p.H * (p.inv_r0 - inv(T(r)))) : T(p.n0)
end

@inline function local_scale_height(p::ChamberlainProfile{FT}, r::Real) where {FT}
    T = float(promote_type(FT, typeof(r)))
    return (p.H > zero(FT) && r > zero(r)) ? (T(r) * T(r)) / p.H : zero(T)
end

"""
    ExosphereProfile{FT<:AbstractFloat}(n0, r0, inv_H, center, power)

Concrete, allocation-free radial neutral density/rate profile for planetary exospheres and
cometary comae:
`p(x) = n0 * (r0 / max(r, r0))^power * exp(-max(r - r0, 0) * inv_H)`.
Callable on 1D `(x,)`, 2D `(x, y)`, and 3D `(x, y, z)` coordinates; construct via
[`exosphere_profile`](@ref).
"""
struct ExosphereProfile{FT <: AbstractFloat} <: AbstractNeutralProfile
    n0::FT
    r0::FT
    inv_H::FT
    center::NTuple{3, FT}
    power::FT
end

"""
    exosphere_profile(; n0 = 1.0, r0 = 1.0, H = 0.5, center = (0.0, 0.0, 0.0),
                      power = 0.0, FT = Float64) -> ExosphereProfile{FT}

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
        FT::Type{<:AbstractFloat} = Float64,
    )
    n0_f = FT(n0)
    r0_f = FT(r0)
    H_f = FT(H)
    p_f = FT(power)
    c_f = (FT(center[1]), FT(center[2]), FT(center[3]))
    (isfinite(n0_f) && n0_f >= zero(FT)) ||
        throw(ArgumentError("exosphere_profile requires finite n0 >= 0 (got $n0)"))
    (isfinite(r0_f) && r0_f > zero(FT)) ||
        throw(ArgumentError("exosphere_profile requires finite r0 > 0 (got $r0)"))
    (H_f > zero(FT)) || throw(ArgumentError("exosphere_profile requires H > 0 (got $H)"))
    (isfinite(p_f) && p_f >= zero(FT)) ||
        throw(ArgumentError("exosphere_profile requires finite power >= 0 (got $power)"))
    all(isfinite, c_f) ||
        throw(ArgumentError("exosphere_profile requires finite center (got $center)"))
    inv_H = isinf(H_f) ? zero(FT) : inv(H_f)
    return ExosphereProfile{FT}(n0_f, r0_f, inv_H, c_f, p_f)
end

@inline function neutral_density(p::ExosphereProfile{FT}, r::Real) where {FT}
    T = float(promote_type(FT, typeof(r)))
    rf = T(r)
    val = T(p.n0)
    if rf > p.r0
        if p.power > zero(FT)
            val *= _fast_pow(p.r0 / rf, p.power)
        end
        if p.inv_H > zero(FT)
            val *= exp(-(rf - p.r0) * p.inv_H)
        end
    end
    return val
end

@inline function local_scale_height(p::ExosphereProfile{FT}, r::Real) where {FT}
    T = float(promote_type(FT, typeof(r)))
    r <= zero(r) && return zero(T)
    inv_h = (p.power > zero(FT) ? p.power / T(r) : zero(T)) + p.inv_H
    return inv_h > zero(inv_h) ? inv(inv_h) : zero(T)
end

@inline (p::ExosphereProfile)(x::Real) = neutral_density(p, abs(x - p.center[1]))

@inline @muladd function (p::ExosphereProfile{FT})(x::Real, y::Real) where {FT}
    dx = x - p.center[1]
    dy = y - p.center[2]
    r2 = dx * dx + dy * dy
    T = float(promote_type(FT, typeof(x), typeof(y)))
    r2 <= p.r0 * p.r0 && return T(p.n0)
    return neutral_density(p, sqrt(r2))
end

@inline @muladd function (p::ExosphereProfile{FT})(x::Real, y::Real, z::Real) where {FT}
    dx = x - p.center[1]
    dy = y - p.center[2]
    dz = z - p.center[3]
    r2 = dx * dx + dy * dy + dz * dz
    T = float(promote_type(FT, typeof(x), typeof(y), typeof(z)))
    r2 <= p.r0 * p.r0 && return T(p.n0)
    return neutral_density(p, sqrt(r2))
end

# Generic 3D position fallback for radial profiles carrying `.center`
@inline function neutral_density(
        p::AbstractNeutralProfile, x::AbstractVector{<:Real}; t = nothing,
    )
    d = _relative_pos(p.center, _as_svec3(x))
    return neutral_density(p, _safe_norm(d))
end

@inline function local_scale_height(
        p::AbstractNeutralProfile, x::AbstractVector{<:Real}; t = nothing,
    )
    d = _relative_pos(p.center, _as_svec3(x))
    return local_scale_height(p, _safe_norm(d))
end

# ------------------------------------------------------------------------------
# 3D Asymmetric, Composite, Tabulated & Time-Dependent Profiles
# ------------------------------------------------------------------------------

"""
    DayNightProfile{P<:AbstractNeutralProfile, FT<:AbstractFloat} <: AbstractNeutralProfile
    DayNightProfile(base; solar_dir = (1.0, 0.0, 0.0), asymmetry = 0.5,
                    min_factor = 0.1, center = (0.0, 0.0, 0.0), FT = Float64)

3D dayside–nightside modulated neutral profile around `center`:

```math
n(\\mathbf{x}) = n_{\\text{base}}(r)\\,\\max\\bigl(f_{\\min},\\,1 + \\alpha \\cos\\chi\\bigr),
```

where ``\\cos\\chi = ((\\mathbf{x} - \\mathbf{c}) \\cdot \\hat{\\mathbf{s}}) / r`` is the
solar zenith angle cosine along unit vector `solar_dir`.
"""
struct DayNightProfile{P <: AbstractNeutralProfile, FT <: AbstractFloat} <:
    AbstractNeutralProfile
    base::P
    solar_dir::SVector{3, FT}
    asymmetry::FT
    min_factor::FT
    center::SVector{3, FT}
end

function DayNightProfile(
        base::P;
        solar_dir::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            1.0, 0.0, 0.0,
        ),
        asymmetry::Real = 0.5,
        min_factor::Real = 0.1,
        center::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            0.0, 0.0, 0.0,
        ),
        FT::Type{<:AbstractFloat} = Float64,
    ) where {P <: AbstractNeutralProfile}
    length(solar_dir) == 3 ||
        throw(ArgumentError("DayNightProfile: solar_dir must have length 3"))
    d = SVector{3, FT}(FT(solar_dir[1]), FT(solar_dir[2]), FT(solar_dir[3]))
    dn = norm(d)
    dn > zero(FT) || throw(ArgumentError("DayNightProfile: solar_dir must be non-zero"))
    isfinite(asymmetry) || throw(ArgumentError("DayNightProfile: asymmetry must be finite"))
    (isfinite(min_factor) && min_factor >= 0) ||
        throw(ArgumentError("DayNightProfile: min_factor must be >= 0"))
    c = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    return DayNightProfile{P, FT}(base, d / dn, FT(asymmetry), FT(min_factor), c)
end

@inline neutral_density(p::DayNightProfile, r::Real) = neutral_density(p.base, r)
@inline local_scale_height(p::DayNightProfile, r::Real) = local_scale_height(p.base, r)

@inline @muladd function neutral_density(
        p::DayNightProfile{P, FT}, x::AbstractVector{<:Real}; t = nothing,
    ) where {P, FT}
    d = _relative_pos(p.center, _as_svec3(x))
    r = _safe_norm(d)
    n_base = neutral_density(p.base, r)
    iszero(r) && return n_base
    cos_sza = (d[1] * p.solar_dir[1] + d[2] * p.solar_dir[2] + d[3] * p.solar_dir[3]) / r
    fac = max(p.min_factor, one(cos_sza) + p.asymmetry * cos_sza)
    return n_base * fac
end

@inline function local_scale_height(
        p::DayNightProfile, x::AbstractVector{<:Real}; t = nothing,
    )
    d = _relative_pos(p.center, _as_svec3(x))
    return local_scale_height(p.base, _safe_norm(d))
end

"""
    PlumeProfile{FT<:AbstractFloat} <: AbstractNeutralProfile
    PlumeProfile(; n0, origin, axis, r0 = 0.05, H = Inf, k = 2.0,
                 opening_angle = 0.35, FT = Float64)

3D localized volcanic, cryovolcanic, or outgassing plume centered at surface vent `origin`
along unit direction `axis` with smooth von Mises–Fisher angular decay (matching a Gaussian
``\\exp(-\\theta^2 / (2\\theta_0^2))`` for small angles while remaining smooth at
``\\theta = 0`` for automatic differentiation):

```math
n(\\mathbf{x}) = n_0 \\left(\\frac{r_0}{\\max(s, r_0)}\\right)^k
\\exp\\!\\left(-\\frac{\\max(s - r_0, 0)}{H}
- \\frac{1 - \\cos\\theta}{2(1 - \\cos\\theta_0)}\\right),
```

for ``\\mathbf{d} \\cdot \\hat{\\mathbf{a}} > 0`` (`0` behind the vent plane), where
``\\mathbf{d} = \\mathbf{x} - \\mathbf{x}_0`` and ``s = |\\mathbf{d}|``.
"""
struct PlumeProfile{FT <: AbstractFloat} <: AbstractNeutralProfile
    n0::FT
    r0::FT
    inv_H::FT
    k::FT
    origin::SVector{3, FT}
    axis::SVector{3, FT}
    inv_ang_width::FT
end

function PlumeProfile(;
        n0::Real,
        origin::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}},
        axis::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}},
        r0::Real = 0.05,
        H::Real = Inf,
        k::Real = 2.0,
        opening_angle::Real = 0.35,
        FT::Type{<:AbstractFloat} = Float64,
    )
    (isfinite(n0) && n0 >= 0) || throw(ArgumentError("PlumeProfile requires finite n0 >= 0"))
    (isfinite(r0) && r0 > 0) || throw(ArgumentError("PlumeProfile requires finite r0 > 0"))
    H > 0 || throw(ArgumentError("PlumeProfile requires H > 0"))
    (isfinite(k) && k >= 0) || throw(ArgumentError("PlumeProfile requires finite k >= 0"))
    (isfinite(opening_angle) && 0 < opening_angle < π) ||
        throw(ArgumentError("PlumeProfile requires opening_angle in (0, π)"))
    length(origin) == 3 || throw(ArgumentError("PlumeProfile: origin must have length 3"))
    length(axis) == 3 || throw(ArgumentError("PlumeProfile: axis must have length 3"))
    org = SVector{3, FT}(FT(origin[1]), FT(origin[2]), FT(origin[3]))
    ax = SVector{3, FT}(FT(axis[1]), FT(axis[2]), FT(axis[3]))
    an = norm(ax)
    an > zero(FT) || throw(ArgumentError("PlumeProfile: axis must be non-zero"))
    inv_H = isinf(H) ? zero(FT) : inv(FT(H))
    inv_ang = inv(FT(2) * (one(FT) - cos(FT(opening_angle))))
    return PlumeProfile{FT}(FT(n0), FT(r0), inv_H, FT(k), org, ax / an, inv_ang)
end

@inline function neutral_density(p::PlumeProfile{FT}, r::Real) where {FT}
    # Along-axis 1D radial profile (s = r)
    T = float(promote_type(FT, typeof(r)))
    rf = T(r)
    val = T(p.n0)
    if rf > p.r0
        p.k > zero(FT) && (val *= _fast_pow(p.r0 / rf, p.k))
        p.inv_H > zero(FT) && (val *= exp(-(rf - p.r0) * p.inv_H))
    end
    return val
end

@inline function local_scale_height(p::PlumeProfile{FT}, r::Real) where {FT}
    T = float(promote_type(FT, typeof(r)))
    r <= zero(r) && return zero(T)
    inv_h = (p.k > zero(FT) ? p.k / T(r) : zero(T)) + p.inv_H
    return inv_h > zero(inv_h) ? inv(inv_h) : zero(T)
end

@inline @muladd function neutral_density(
        p::PlumeProfile{FT}, x::AbstractVector{<:Real}; t = nothing,
    ) where {FT}
    xv = _as_svec3(x)
    d = SVector{3}(xv[1] - p.origin[1], xv[2] - p.origin[2], xv[3] - p.origin[3])
    proj = d[1] * p.axis[1] + d[2] * p.axis[2] + d[3] * p.axis[3]
    T = float(promote_type(FT, typeof(proj)))
    proj <= zero(proj) && return zero(T)
    s = _safe_norm(d)
    iszero(s) && return T(p.n0)
    cos_th = proj / s
    ang = exp(-(one(cos_th) - cos_th) * p.inv_ang_width)
    return neutral_density(p, s) * ang
end

@inline function local_scale_height(
        p::PlumeProfile, x::AbstractVector{<:Real}; t = nothing,
    )
    xv = _as_svec3(x)
    d = SVector{3}(xv[1] - p.origin[1], xv[2] - p.origin[2], xv[3] - p.origin[3])
    return local_scale_height(p, _safe_norm(d))
end

"""
    CompositeProfile{Ps<:Tuple} <: AbstractNeutralProfile
    CompositeProfile(profiles::AbstractNeutralProfile...)

Superposition of multiple [`AbstractNeutralProfile`](@ref) terms for a single neutral
species (e.g., cold thermal + hot photochemical corona). Fully `isbits` and tuple-unrolled
when all constituent profiles are `isbits`.
"""
struct CompositeProfile{Ps <: Tuple} <: AbstractNeutralProfile
    profiles::Ps
    function CompositeProfile(profiles::Tuple{Vararg{AbstractNeutralProfile}})
        isempty(profiles) &&
            throw(ArgumentError("CompositeProfile requires at least one profile"))
        return new{typeof(profiles)}(profiles)
    end
end

CompositeProfile(p::AbstractNeutralProfile, ps::AbstractNeutralProfile...) =
    CompositeProfile((p, ps...))

@inline _sum_profiles_r(ps::Tuple{P}, r::Real) where {P} = neutral_density(ps[1], r)
@inline _sum_profiles_r(ps::Tuple{P, Vararg}, r::Real) where {P} =
    neutral_density(ps[1], r) + _sum_profiles_r(Base.tail(ps), r)

@inline _sum_profiles_x(ps::Tuple{P}, x::SVector{3}, t) where {P} =
    neutral_density(ps[1], x; t = t)
@inline _sum_profiles_x(ps::Tuple{P, Vararg}, x::SVector{3}, t) where {P} =
    neutral_density(ps[1], x; t = t) + _sum_profiles_x(Base.tail(ps), x, t)

@inline function _sum_n_and_dn_r(ps::Tuple{P}, r::Real) where {P}
    n = neutral_density(ps[1], r)
    h = local_scale_height(ps[1], r)
    dn = (h > zero(h) && n > zero(n)) ? n / h : zero(n / one(h))
    return (n, dn)
end

@inline function _sum_n_and_dn_r(ps::Tuple{P, Vararg}, r::Real) where {P}
    n1, dn1 = _sum_n_and_dn_r((ps[1],), r)
    n2, dn2 = _sum_n_and_dn_r(Base.tail(ps), r)
    return (n1 + n2, dn1 + dn2)
end

@inline function _sum_n_and_dn_x(ps::Tuple{P}, x::SVector{3}, t) where {P}
    n = neutral_density(ps[1], x; t = t)
    h = local_scale_height(ps[1], x; t = t)
    dn = (h > zero(h) && n > zero(n)) ? n / h : zero(n / one(h))
    return (n, dn)
end

@inline function _sum_n_and_dn_x(ps::Tuple{P, Vararg}, x::SVector{3}, t) where {P}
    n1, dn1 = _sum_n_and_dn_x((ps[1],), x, t)
    n2, dn2 = _sum_n_and_dn_x(Base.tail(ps), x, t)
    return (n1 + n2, dn1 + dn2)
end

@inline neutral_density(p::CompositeProfile, r::Real) = _sum_profiles_r(p.profiles, r)

@inline neutral_density(p::CompositeProfile, x::AbstractVector{<:Real}; t = nothing) =
    _sum_profiles_x(p.profiles, _as_svec3(x), t)

@inline function local_scale_height(p::CompositeProfile, r::Real)
    n, dn = _sum_n_and_dn_r(p.profiles, r)
    return dn > zero(dn) ? n / dn : zero(n / one(dn))
end

@inline function local_scale_height(
        p::CompositeProfile, x::AbstractVector{<:Real}; t = nothing,
    )
    n, dn = _sum_n_and_dn_x(p.profiles, _as_svec3(x), t)
    return dn > zero(dn) ? n / dn : zero(n / one(dn))
end

"""
    TabulatedRadialProfile{FT, VR, VN} <: AbstractNeutralProfile
    TabulatedRadialProfile(r, n; center = (0.0, 0.0, 0.0), log_interp = true,
                           clamp_outer = false)

1D tabulated radial density profile (e.g., from empirical or DSMC radial profiles) with
zero-allocation binary search and linear or log-linear interpolation. Accepts `SVector`
storage for `isbits` GPU execution or `Vector` for large tables.
"""
struct TabulatedRadialProfile{
        FT <: AbstractFloat, VR <: AbstractVector{FT}, VN <: AbstractVector{FT},
    } <: AbstractNeutralProfile
    r::VR
    n::VN
    center::SVector{3, FT}
    log_interp::Bool
    clamp_outer::Bool
end

function TabulatedRadialProfile(
        r::AbstractVector{<:Real},
        n::AbstractVector{<:Real};
        center::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            0.0, 0.0, 0.0,
        ),
        log_interp::Bool = true,
        clamp_outer::Bool = false,
    )
    length(r) == length(n) ||
        throw(ArgumentError("TabulatedRadialProfile: r and n must have equal length"))
    length(r) >= 2 ||
        throw(ArgumentError("TabulatedRadialProfile requires at least 2 radial nodes"))
    for i in (firstindex(r) + 1):lastindex(r)
        r[i] > r[i - 1] ||
            throw(ArgumentError("TabulatedRadialProfile: r must be strictly increasing"))
    end
    any(<(0), n) && throw(ArgumentError("TabulatedRadialProfile: densities must be >= 0"))
    FT = float(promote_type(eltype(r), eltype(n)))
    rv = r isa SVector ? SVector{length(r), FT}(r) : FT[FT(x) for x in r]
    nv = n isa SVector ? SVector{length(n), FT}(n) : FT[FT(x) for x in n]
    c = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    return TabulatedRadialProfile{FT, typeof(rv), typeof(nv)}(
        rv, nv, c, log_interp, clamp_outer,
    )
end

@inline function _find_bracket(grid::AbstractVector, x::Real)
    lo = firstindex(grid)
    hi = lastindex(grid) - 1
    while lo < hi
        mid = (lo + hi + 1) >>> 1
        @inbounds if grid[mid] <= x
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

@inline function neutral_density(p::TabulatedRadialProfile{FT}, r::Real) where {FT}
    T = float(promote_type(FT, typeof(r)))
    rf = T(r)
    @inbounds begin
        r1 = p.r[1]
        rf <= r1 && return T(p.n[1])
        rn = p.r[end]
        if rf >= rn
            return p.clamp_outer ? T(p.n[end]) : zero(T)
        end
        i = _find_bracket(p.r, rf)
        ra = p.r[i]
        rb = p.r[i + 1]
        na = p.n[i]
        nb = p.n[i + 1]
        w = (rf - ra) / (rb - ra)
        if p.log_interp && na > zero(FT) && nb > zero(FT)
            return exp((one(w) - w) * log(na) + w * log(nb))
        end
        return (one(w) - w) * na + w * nb
    end
end

@inline function local_scale_height(p::TabulatedRadialProfile{FT}, r::Real) where {FT}
    T = float(promote_type(FT, typeof(r)))
    rf = T(r)
    @inbounds begin
        (rf < p.r[1] || rf > p.r[end]) && return zero(T)
        i = _find_bracket(p.r, rf)
        dr = p.r[i + 1] - p.r[i]
        na = p.n[i]
        nb = p.n[i + 1]
        if na > zero(FT) && nb > zero(FT)
            dln = log(na / nb)
            return dln > zero(dln) ? T(dr / dln) : zero(T)
        end
        return zero(T)
    end
end

"""
    TimeDependentProfile(density_fn; scale_height_fn = nothing)

Wrapper for arbitrary user-defined 3D and/or time-dependent neutral density profiles
`density_fn(x, t)` (with optional `scale_height_fn(x, t)`).
"""
struct TimeDependentProfile{F, FH} <: AbstractNeutralProfile
    density_fn::F
    scale_height_fn::FH
end

TimeDependentProfile(density_fn; scale_height_fn = nothing) =
    TimeDependentProfile{typeof(density_fn), typeof(scale_height_fn)}(
    density_fn, scale_height_fn,
)

@inline neutral_density(p::TimeDependentProfile, x::AbstractVector{<:Real}; t = nothing) =
    p.density_fn(_as_svec3(x), t)

@inline neutral_density(p::TimeDependentProfile, r::Real) =
    p.density_fn(SVector{3}(r, zero(r), zero(r)), nothing)

@inline function local_scale_height(
        p::TimeDependentProfile{F, Nothing}, x::AbstractVector{<:Real}; t = nothing,
    ) where {F}
    return zero(float(eltype(x)))
end

@inline function local_scale_height(
        p::TimeDependentProfile, x::AbstractVector{<:Real}; t = nothing,
    )
    return p.scale_height_fn(_as_svec3(x), t)
end

@inline function local_scale_height(p::TimeDependentProfile{F, Nothing}, r::Real) where {F}
    return zero(float(typeof(r)))
end

@inline function local_scale_height(p::TimeDependentProfile, r::Real)
    return p.scale_height_fn(SVector{3}(r, zero(r), zero(r)), nothing)
end

# ------------------------------------------------------------------------------
# Composable Spatial Velocity & Temperature Models (isbits)
# ------------------------------------------------------------------------------

"""
    neutral_velocity(v::AbstractVelocityModel, x; t = nothing) -> SVector{3}

Evaluate the local neutral bulk velocity ``\\mathbf{u}_n(\\mathbf{x}, t)``.
"""
function neutral_velocity end

"""
    neutral_temperature(temp::AbstractTemperatureModel, x; t = nothing)

Evaluate the local neutral temperature ``T_n(\\mathbf{x}, t)``.
"""
function neutral_temperature end

"""
    ZeroVelocity() <: AbstractVelocityModel

Static zero-velocity neutral background ``\\mathbf{u}_n(\\mathbf{x}, t) = \\mathbf{0}``.
"""
struct ZeroVelocity <: AbstractVelocityModel end

@inline function neutral_velocity(::ZeroVelocity, x::AbstractVector{<:Real}; t = nothing)
    T = float(eltype(x))
    return SVector{3, T}(zero(T), zero(T), zero(T))
end

"""
    ConstantVelocity(u) <: AbstractVelocityModel

Spatially uniform neutral drift velocity ``\\mathbf{u}_n(\\mathbf{x}, t) = \\mathbf{u}``.
"""
struct ConstantVelocity{FT <: AbstractFloat} <: AbstractVelocityModel
    u::SVector{3, FT}
end

function ConstantVelocity(
        u::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}};
        FT::Type{<:AbstractFloat} = Float64,
    )
    length(u) == 3 || throw(ArgumentError("ConstantVelocity: u must have length 3"))
    return ConstantVelocity{FT}(SVector{3, FT}(FT(u[1]), FT(u[2]), FT(u[3])))
end

@inline neutral_velocity(v::ConstantVelocity, ::AbstractVector{<:Real}; t = nothing) = v.u

"""
    CorotationVelocity(omega; center = (0.0, 0.0, 0.0), FT = Float64) <: AbstractVelocityModel

Rigid planetary/moon atmospheric corotation velocity
``\\mathbf{u}_n(\\mathbf{x}) = \\boldsymbol{\\Omega} \\times (\\mathbf{x} - \\mathbf{c})``.
"""
struct CorotationVelocity{FT <: AbstractFloat} <: AbstractVelocityModel
    omega::SVector{3, FT}
    center::SVector{3, FT}
end

function CorotationVelocity(
        omega::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}};
        center::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            0.0, 0.0, 0.0,
        ),
        FT::Type{<:AbstractFloat} = Float64,
    )
    length(omega) == 3 || throw(ArgumentError("CorotationVelocity: omega must have length 3"))
    length(center) == 3 || throw(ArgumentError("CorotationVelocity: center must have length 3"))
    w = SVector{3, FT}(FT(omega[1]), FT(omega[2]), FT(omega[3]))
    c = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    return CorotationVelocity{FT}(w, c)
end

@inline function neutral_velocity(
        v::CorotationVelocity, x::AbstractVector{<:Real}; t = nothing,
    )
    d = _relative_pos(v.center, _as_svec3(x))
    return cross(v.omega, d)
end

"""
    RadialOutflowVelocity(v_out; center = (0.0, 0.0, 0.0), FT = Float64) <: AbstractVelocityModel

Spherically radial outflow velocity ``\\mathbf{u}_n(\\mathbf{x}) = v_{\\text{out}} \\hat{\\mathbf{r}}``
around `center` (e.g., cometary coma or escaping atmosphere; returns zero at `r == 0`).
"""
struct RadialOutflowVelocity{FT <: AbstractFloat} <: AbstractVelocityModel
    v_out::FT
    center::SVector{3, FT}
end

function RadialOutflowVelocity(
        v_out::Real;
        center::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            0.0, 0.0, 0.0,
        ),
        FT::Type{<:AbstractFloat} = Float64,
    )
    isfinite(v_out) || throw(ArgumentError("RadialOutflowVelocity requires finite v_out"))
    length(center) == 3 ||
        throw(ArgumentError("RadialOutflowVelocity: center must have length 3"))
    c = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    return RadialOutflowVelocity{FT}(FT(v_out), c)
end

@inline function neutral_velocity(
        v::RadialOutflowVelocity{FT}, x::AbstractVector{<:Real}; t = nothing,
    ) where {FT}
    d = _relative_pos(v.center, _as_svec3(x))
    r = _safe_norm(d)
    T = float(promote_type(FT, typeof(r)))
    iszero(r) && return SVector{3, T}(zero(T), zero(T), zero(T))
    scale = v.v_out / r
    return SVector{3, T}(d[1] * scale, d[2] * scale, d[3] * scale)
end

"""
    ConstantTemperature(T0) <: AbstractTemperatureModel

Isothermal neutral background ``T_n(\\mathbf{x}, t) = T_0``.
"""
struct ConstantTemperature{FT <: AbstractFloat} <: AbstractTemperatureModel
    T0::FT
    function ConstantTemperature{FT}(T0::FT) where {FT <: AbstractFloat}
        (isfinite(T0) && T0 >= zero(FT)) ||
            throw(ArgumentError("ConstantTemperature requires finite T0 >= 0"))
        return new{FT}(T0)
    end
end

ConstantTemperature(T0::Real; FT::Type{<:AbstractFloat} = float(typeof(T0))) =
    ConstantTemperature{FT}(FT(T0))

@inline neutral_temperature(temp::ConstantTemperature, ::AbstractVector{<:Real}; t = nothing) =
    temp.T0

"""
    DayNightTemperature(; T_day, T_night, solar_dir = (1.0, 0.0, 0.0),
                        center = (0.0, 0.0, 0.0), FT = Float64) <: AbstractTemperatureModel

Solar-zenith-angle dependent exobase temperature smoothly varying between `T_day`
(subsolar) and `T_night` (antisolar):
``T(\\mathbf{x}) = T_{\\text{night}} + \\frac{1}{2}(T_{\\text{day}} - T_{\\text{night}})(1 + \\cos\\chi)``.
"""
struct DayNightTemperature{FT <: AbstractFloat} <: AbstractTemperatureModel
    T_day::FT
    T_night::FT
    solar_dir::SVector{3, FT}
    center::SVector{3, FT}
end

function DayNightTemperature(;
        T_day::Real,
        T_night::Real,
        solar_dir::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            1.0, 0.0, 0.0,
        ),
        center::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            0.0, 0.0, 0.0,
        ),
        FT::Type{<:AbstractFloat} = Float64,
    )
    (isfinite(T_day) && T_day >= 0) ||
        throw(ArgumentError("DayNightTemperature requires finite T_day >= 0"))
    (isfinite(T_night) && T_night >= 0) ||
        throw(ArgumentError("DayNightTemperature requires finite T_night >= 0"))
    length(solar_dir) == 3 ||
        throw(ArgumentError("DayNightTemperature: solar_dir must have length 3"))
    d = SVector{3, FT}(FT(solar_dir[1]), FT(solar_dir[2]), FT(solar_dir[3]))
    dn = norm(d)
    dn > zero(FT) || throw(ArgumentError("DayNightTemperature: solar_dir must be non-zero"))
    c = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    return DayNightTemperature{FT}(FT(T_day), FT(T_night), d / dn, c)
end

@inline @muladd function neutral_temperature(
        temp::DayNightTemperature{FT}, x::AbstractVector{<:Real}; t = nothing,
    ) where {FT}
    d = _relative_pos(temp.center, _as_svec3(x))
    r = _safe_norm(d)
    iszero(r) && return FT(0.5) * (temp.T_day + temp.T_night)
    cos_sza = (d[1] * temp.solar_dir[1] + d[2] * temp.solar_dir[2] + d[3] * temp.solar_dir[3]) / r
    return temp.T_night + FT(0.5) * (temp.T_day - temp.T_night) * (one(cos_sza) + cos_sza)
end

# ------------------------------------------------------------------------------
# NeutralPopulation (PlasmaPopulations.jl Style: Species + Profile + VDF/Fields)
# ------------------------------------------------------------------------------

"""
    NeutralPopulation(species::NeutralSpecies, profile::AbstractNeutralProfile;
                      velocity = ZeroVelocity(), temperature = ConstantTemperature(1.0))
    NeutralPopulation(species::NeutralSpecies, profile::AbstractNeutralProfile,
                      vdf::AbstractNeutralVDF)

A neutral exosphere population combining a physical [`NeutralSpecies`](@ref), a spatial
[`AbstractNeutralProfile`](@ref), and either spatially varying `(velocity, temperature)`
models or a spatially uniform [`AbstractNeutralVDF`](@ref). Fully `isbits` when all
component types are `isbits`.
"""
struct NeutralPopulation{
        S <: NeutralSpecies,
        P <: AbstractNeutralProfile,
        V <: Union{AbstractVelocityModel, AbstractNeutralVDF},
        Temp <: Union{AbstractTemperatureModel, Nothing},
    }
    species::S
    profile::P
    velocity::V
    temperature::Temp
end

function NeutralPopulation(
        species::S,
        profile::P;
        velocity::V = ZeroVelocity(),
        temperature::Temp = ConstantTemperature(one(float(typeof(species.m)))),
    ) where {
        S <: NeutralSpecies,
        P <: AbstractNeutralProfile,
        V <: AbstractVelocityModel,
        Temp <: AbstractTemperatureModel,
    }
    return NeutralPopulation{S, P, V, Temp}(species, profile, velocity, temperature)
end

function NeutralPopulation(
        species::S, profile::P, vdf::V,
    ) where {S <: NeutralSpecies, P <: AbstractNeutralProfile, V <: AbstractNeutralVDF}
    return NeutralPopulation{S, P, V, Nothing}(species, profile, vdf, nothing)
end

Base.broadcastable(pop::NeutralPopulation) = Ref(pop)

@inline species_mass(pop::NeutralPopulation) = pop.species.m

@inline neutral_density(pop::NeutralPopulation, r::Real) = neutral_density(pop.profile, r)

@inline neutral_density(pop::NeutralPopulation, x::AbstractVector{<:Real}; t = nothing) =
    neutral_density(pop.profile, x; t = t)

@inline local_scale_height(pop::NeutralPopulation, r::Real) =
    local_scale_height(pop.profile, r)

@inline local_scale_height(pop::NeutralPopulation, x::AbstractVector{<:Real}; t = nothing) =
    local_scale_height(pop.profile, x; t = t)

@inline function neutral_state(
        pop::NeutralPopulation{S, P, <:AbstractVelocityModel, <:AbstractTemperatureModel},
        x::AbstractVector{<:Real};
        t = nothing,
        kB::Number = 1,
    ) where {S, P}
    xv = _as_svec3(x)
    n = neutral_density(pop.profile, xv; t = t)
    u = neutral_velocity(pop.velocity, xv; t = t)
    T = neutral_temperature(pop.temperature, xv; t = t)
    return NeutralState(n, u, T)
end

@inline function neutral_state(
        pop::NeutralPopulation{S, P, <:AbstractNeutralVDF, Nothing},
        x::AbstractVector{<:Real};
        t = nothing,
        kB::Number = 1,
    ) where {S, P}
    n = neutral_density(pop.profile, _as_svec3(x); t = t)
    u = mean(pop.velocity)
    T = vdf_temperature(pop.species, pop.velocity; kB = kB)
    return NeutralState(n, u, T)
end

@inline function local_vdf(
        pop::NeutralPopulation{S, P, <:AbstractNeutralVDF, Nothing},
        ::AbstractVector{<:Real};
        t = nothing,
        kB::Number = 1,
    ) where {S, P}
    return pop.velocity
end

@inline function local_vdf(
        pop::NeutralPopulation{S, P, <:AbstractVelocityModel, <:AbstractTemperatureModel},
        x::AbstractVector{<:Real};
        t = nothing,
        kB::Number = 1,
    ) where {S, P}
    xv = _as_svec3(x)
    u = neutral_velocity(pop.velocity, xv; t = t)
    T = neutral_temperature(pop.temperature, xv; t = t)
    return _local_vdf_from_uT(pop.species, u, T, kB)
end

@inline function _local_vdf_from_uT(species::NeutralSpecies, u::SVector{3}, T::Real, kB::Number)
    if T > zero(T)
        vth = thermal_speed(species, T; kB = kB)
        FT = float(promote_type(typeof(vth), eltype(u)))
        return Maxwellian{FT}(FT(vth), SVector{3, FT}(FT(u[1]), FT(u[2]), FT(u[3])))
    end
    FT = float(promote_type(typeof(species.m), eltype(u)))
    return ColdNeutral{FT}(SVector{3, FT}(FT(u[1]), FT(u[2]), FT(u[3])))
end

# ------------------------------------------------------------------------------
# Legacy v0.1.0 NeutralComponentCode (Backwards-Compatible Bridge)
# ------------------------------------------------------------------------------

"""
    NeutralComponentCode(profile::Symbol, n0, H, k, T0, name = "neutral")

Code-unit specification of one neutral exosphere component, where `profile` is one of
`:exponential`, `:powerlaw`, or `:chamberlain` (internally encoded as `kind` `1`, `2`, `3`),
`n0` is the surface density at `planet_radius`, `H` is the exponential/Chamberlain scale
parameter, `k` is the power-law exponent, and `T0 > 0` is the neutral temperature.
"""
struct NeutralComponentCode
    kind::Int
    n0::Float64
    H::Float64
    inv_H::Float64
    k::Float64
    T0::Float64
    name::String
end

NeutralComponentCode(
    kind::Int, n0::Float64, H::Float64, k::Float64, T0::Float64, name::String,
) = NeutralComponentCode(kind, n0, H, H > 0.0 ? inv(H) : 0.0, k, T0, name)

function NeutralComponentCode(
        profile::Symbol, n0::Real, H::Real, k::Real, T0::Real,
        name::AbstractString = "neutral",
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
    inv_H = Hf > 0.0 ? inv(Hf) : 0.0
    return NeutralComponentCode(kind, Float64(n0), Hf, inv_H, Float64(k), T0f, String(name))
end

function NeutralPopulation(
        c::NeutralComponentCode,
        planet_radius::Real;
        center::NTuple{3, Real} = (0.0, 0.0, 0.0),
        mass::Real = 1.0,
        velocity::AbstractVelocityModel = ZeroVelocity(),
        FT::Type{<:AbstractFloat} = Float64,
    )
    prof = if c.kind == _EXO_EXPONENTIAL
        ExponentialProfile(c.n0, planet_radius, c.H; center = center, FT = FT)
    elseif c.kind == _EXO_POWERLAW
        PowerLawProfile(c.n0, planet_radius, c.k; center = center, FT = FT)
    else
        ChamberlainProfile(c.n0, planet_radius, c.H; center = center, FT = FT)
    end
    sp = NeutralSpecies(; m = FT(mass))
    return NeutralPopulation(
        sp, prof; velocity = velocity, temperature = ConstantTemperature(c.T0; FT = FT),
    )
end
