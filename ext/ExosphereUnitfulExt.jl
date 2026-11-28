module ExosphereUnitfulExt

import Exosphere
using Exosphere: AbstractNeutralProfile, ChamberlainProfile, ExponentialProfile,
    KB_SI, PowerLawProfile, UnitScale
using StaticArrays: SVector
using Unitful: AbstractQuantity, Units, dimension, uconvert, unit, ustrip, @u_str

# ------------------------------------------------------------------------------
# SI Unit Adapters (PlasmaPopulationsUnitfulExt Style)
# ------------------------------------------------------------------------------

Exosphere.si_mass(m::AbstractQuantity) = ustrip(u"kg", m)
Exosphere.si_length(l::AbstractQuantity) = ustrip(u"m", l)
Exosphere.si_number_density(n::AbstractQuantity) = ustrip(u"m^-3", n)
Exosphere.si_velocity(v::AbstractQuantity) = ustrip(u"m/s", v)
Exosphere.si_temperature(T::AbstractQuantity) = ustrip(u"K", T)
Exosphere.si_cross_section(s::AbstractQuantity) = ustrip(u"m^2", s)
Exosphere.si_frequency(f::AbstractQuantity) = ustrip(u"s^-1", f)

function Exosphere.si_energy(T::AbstractQuantity)
    if dimension(T) == dimension(u"K")
        tk = ustrip(u"K", T)
        return tk * oftype(float(tk), KB_SI)
    end
    return ustrip(u"J", T)
end

# ------------------------------------------------------------------------------
# UnitScale Automatic Dimensional Conversion
# ------------------------------------------------------------------------------

function Exosphere.to_code_units(scale::UnitScale{FT}, q::AbstractQuantity) where {FT}
    d = dimension(q)
    if d == dimension(u"m")
        return FT(ustrip(u"m", q)) / scale.L0
    elseif d == dimension(u"m^-3")
        return FT(ustrip(u"m^-3", q)) / scale.n0
    elseif d == dimension(u"m/s")
        return FT(ustrip(u"m/s", q)) / scale.v0
    elseif d == dimension(u"kg")
        return FT(ustrip(u"kg", q)) / scale.m0
    elseif d == dimension(u"s")
        return FT(ustrip(u"s", q)) / scale.t0
    elseif d == dimension(u"s^-1")
        return FT(ustrip(u"s^-1", q)) * scale.t0
    elseif d == dimension(u"m^2")
        return FT(ustrip(u"m^2", q)) * (scale.n0 * scale.L0)
    elseif d == dimension(u"J")
        return FT(ustrip(u"J", q)) / scale.T0
    elseif d == dimension(u"K")
        return (FT(ustrip(u"K", q)) * FT(KB_SI)) / scale.T0
    elseif d == dimension(u"m^-3 * s^-1")
        return FT(ustrip(u"m^-3 * s^-1", q)) * (scale.t0 / scale.n0)
    elseif d == dimension(u"kg * m^-2 * s^-2")
        return FT(ustrip(u"kg * m^-2 * s^-2", q)) * (scale.t0 / (scale.m0 * scale.n0 * scale.v0))
    elseif d == dimension(u"J * m^-3 * s^-1")
        return FT(ustrip(u"J * m^-3 * s^-1", q)) * (scale.t0 / (scale.n0 * scale.T0))
    end
    throw(ArgumentError("to_code_units: unsupported physical dimension $d for $q"))
end

function Exosphere.from_code_units(scale::UnitScale{FT}, val::Real, unit::Units) where {FT}
    d = dimension(unit)
    v = FT(val)
    if d == dimension(u"m")
        return uconvert(unit, (v * scale.L0) * u"m")
    elseif d == dimension(u"m^-3")
        return uconvert(unit, (v * scale.n0) * u"m^-3")
    elseif d == dimension(u"m/s")
        return uconvert(unit, (v * scale.v0) * u"m/s")
    elseif d == dimension(u"kg")
        return uconvert(unit, (v * scale.m0) * u"kg")
    elseif d == dimension(u"s")
        return uconvert(unit, (v * scale.t0) * u"s")
    elseif d == dimension(u"s^-1")
        return uconvert(unit, (v / scale.t0) * u"s^-1")
    elseif d == dimension(u"m^2")
        return uconvert(unit, (v / (scale.n0 * scale.L0)) * u"m^2")
    elseif d == dimension(u"J")
        return uconvert(unit, (v * scale.T0) * u"J")
    elseif d == dimension(u"K")
        return uconvert(unit, ((v * scale.T0) / FT(KB_SI)) * u"K")
    elseif d == dimension(u"m^-3 * s^-1")
        return uconvert(unit, (v * (scale.n0 / scale.t0)) * u"m^-3 * s^-1")
    elseif d == dimension(u"kg * m^-2 * s^-2")
        return uconvert(unit, (v * (scale.m0 * scale.n0 * scale.v0 / scale.t0)) * u"kg * m^-2 * s^-2")
    elseif d == dimension(u"J * m^-3 * s^-1")
        return uconvert(unit, (v * (scale.n0 * scale.T0 / scale.t0)) * u"J * m^-3 * s^-1")
    end
    throw(ArgumentError("from_code_units: unsupported physical dimension $d for unit $unit"))
end

# ------------------------------------------------------------------------------
# Unit-Aware Radial Profile Wrapper & Strip-to-Code-Units
# ------------------------------------------------------------------------------

struct UnitfulProfile{P <: AbstractNeutralProfile, UN <: Units, UL <: Units} <:
    AbstractNeutralProfile
    profile::P
    n_unit::UN
    l_unit::UL
end

function _strip_center(center::NTuple{3, AbstractQuantity}, l_unit::Units)
    return (ustrip(l_unit, center[1]), ustrip(l_unit, center[2]), ustrip(l_unit, center[3]))
end

function _strip_center(center::NTuple{3, Real}, ::Units)
    return center
end

function Exosphere.ExponentialProfile(
        n0::AbstractQuantity, r0::AbstractQuantity, H::AbstractQuantity;
        center::Union{NTuple{3, Real}, NTuple{3, AbstractQuantity}} = (0.0, 0.0, 0.0),
        FT::Type{<:AbstractFloat} = Float64,
    )
    dimension(n0) == dimension(u"m^-3") ||
        throw(ArgumentError("ExponentialProfile: n0 must have number-density dimensions"))
    dimension(r0) == dimension(u"m") ||
        throw(ArgumentError("ExponentialProfile: r0 must have length dimensions"))
    dimension(H) == dimension(u"m") ||
        throw(ArgumentError("ExponentialProfile: H must have length dimensions"))
    n_u = unit(n0)
    l_u = unit(r0)
    p = ExponentialProfile(
        ustrip(n_u, n0), ustrip(l_u, r0), ustrip(l_u, H);
        center = _strip_center(center, l_u), FT = FT,
    )
    return UnitfulProfile(p, n_u, l_u)
end

function Exosphere.PowerLawProfile(
        n0::AbstractQuantity, r0::AbstractQuantity, k::Real;
        center::Union{NTuple{3, Real}, NTuple{3, AbstractQuantity}} = (0.0, 0.0, 0.0),
        FT::Type{<:AbstractFloat} = Float64,
    )
    dimension(n0) == dimension(u"m^-3") ||
        throw(ArgumentError("PowerLawProfile: n0 must have number-density dimensions"))
    dimension(r0) == dimension(u"m") ||
        throw(ArgumentError("PowerLawProfile: r0 must have length dimensions"))
    n_u = unit(n0)
    l_u = unit(r0)
    p = PowerLawProfile(
        ustrip(n_u, n0), ustrip(l_u, r0), k;
        center = _strip_center(center, l_u), FT = FT,
    )
    return UnitfulProfile(p, n_u, l_u)
end

function Exosphere.ChamberlainProfile(
        n0::AbstractQuantity, r0::AbstractQuantity, H::AbstractQuantity;
        center::Union{NTuple{3, Real}, NTuple{3, AbstractQuantity}} = (0.0, 0.0, 0.0),
        FT::Type{<:AbstractFloat} = Float64,
    )
    dimension(n0) == dimension(u"m^-3") ||
        throw(ArgumentError("ChamberlainProfile: n0 must have number-density dimensions"))
    dimension(r0) == dimension(u"m") ||
        throw(ArgumentError("ChamberlainProfile: r0 must have length dimensions"))
    dimension(H) == dimension(u"m") ||
        throw(ArgumentError("ChamberlainProfile: H must have length dimensions"))
    n_u = unit(n0)
    l_u = unit(r0)
    p = ChamberlainProfile(
        ustrip(n_u, n0), ustrip(l_u, r0), ustrip(l_u, H);
        center = _strip_center(center, l_u), FT = FT,
    )
    return UnitfulProfile(p, n_u, l_u)
end

@inline function Exosphere.neutral_density(up::UnitfulProfile, r::AbstractQuantity)
    return Exosphere.neutral_density(up.profile, ustrip(up.l_unit, r)) * up.n_unit
end

@inline function Exosphere.neutral_density(
        up::UnitfulProfile, x::AbstractVector{<:AbstractQuantity}; t = nothing,
    )
    length(x) == 3 || throw(ArgumentError("Position vector must have length 3"))
    xv = SVector{3}(ustrip(up.l_unit, x[1]), ustrip(up.l_unit, x[2]), ustrip(up.l_unit, x[3]))
    return Exosphere.neutral_density(up.profile, xv; t = t) * up.n_unit
end

@inline function Exosphere.local_scale_height(up::UnitfulProfile, r::AbstractQuantity)
    return Exosphere.local_scale_height(up.profile, ustrip(up.l_unit, r)) * up.l_unit
end

function Exosphere.strip_units(
        up::UnitfulProfile{<:ExponentialProfile}, scale::UnitScale{FT},
    ) where {FT <: AbstractFloat}
    p = up.profile
    n0 = Exosphere.to_code_units(scale, p.n0 * up.n_unit)
    r0 = Exosphere.to_code_units(scale, p.r0 * up.l_unit)
    H = p.inv_H > 0 ? Exosphere.to_code_units(scale, inv(p.inv_H) * up.l_unit) : FT(Inf)
    c = (
        Exosphere.to_code_units(scale, p.center[1] * up.l_unit),
        Exosphere.to_code_units(scale, p.center[2] * up.l_unit),
        Exosphere.to_code_units(scale, p.center[3] * up.l_unit),
    )
    return ExponentialProfile(n0, r0, H; center = c, FT = FT)
end

function Exosphere.strip_units(
        up::UnitfulProfile{<:PowerLawProfile}, scale::UnitScale{FT},
    ) where {FT <: AbstractFloat}
    p = up.profile
    n0 = Exosphere.to_code_units(scale, p.n0 * up.n_unit)
    r0 = Exosphere.to_code_units(scale, p.r0 * up.l_unit)
    c = (
        Exosphere.to_code_units(scale, p.center[1] * up.l_unit),
        Exosphere.to_code_units(scale, p.center[2] * up.l_unit),
        Exosphere.to_code_units(scale, p.center[3] * up.l_unit),
    )
    return PowerLawProfile(n0, r0, p.k; center = c, FT = FT)
end

function Exosphere.strip_units(
        up::UnitfulProfile{<:ChamberlainProfile}, scale::UnitScale{FT},
    ) where {FT <: AbstractFloat}
    p = up.profile
    n0 = Exosphere.to_code_units(scale, p.n0 * up.n_unit)
    r0 = Exosphere.to_code_units(scale, p.r0 * up.l_unit)
    H = Exosphere.to_code_units(scale, p.H * up.l_unit)
    c = (
        Exosphere.to_code_units(scale, p.center[1] * up.l_unit),
        Exosphere.to_code_units(scale, p.center[2] * up.l_unit),
        Exosphere.to_code_units(scale, p.center[3] * up.l_unit),
    )
    return ChamberlainProfile(n0, r0, H; center = c, FT = FT)
end

end # module ExosphereUnitfulExt
