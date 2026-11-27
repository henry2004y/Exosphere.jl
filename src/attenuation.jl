# ==============================================================================
# Shadow Cylinder, EUV Optical Depth, Smith & Smith Chapman Function & ExosphereModel
# ==============================================================================

"""
    ShadowCylinder(; solar_dir = (1.0, 0.0, 0.0), radius = 1.0, half_height = Inf)

Geometric cylindrical planetary EUV/optical shadow along unit vector `dir` (normalized from
`solar_dir`, pointing toward the Sun) of cross-sectional `radius` and downstream
`half_height` (defaults to `Inf`).
"""
struct ShadowCylinder
    dir::SVector{3, Float64}
    radius::Float64
    half_height::Float64
end

function ShadowCylinder(;
        solar_dir::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (1.0, 0.0, 0.0),
        radius::Real = 1.0,
        half_height::Real = Inf,
    )
    length(solar_dir) == 3 || throw(ArgumentError("ShadowCylinder: solar_dir must have length 3"))
    d = SVector{3, Float64}(Float64(solar_dir[1]), Float64(solar_dir[2]), Float64(solar_dir[3]))
    dn = norm(d)
    dn > 0.0 || throw(ArgumentError("ShadowCylinder: solar_dir must be non-zero"))
    return ShadowCylinder(d / dn, Float64(radius), Float64(half_height))
end

const _SQRT_PI = sqrt(Float64(π))
const _TWO_SQRT_PI = 2.0 * _SQRT_PI

"""
    in_shadow(sc::ShadowCylinder, x::SVector{3, Float64}) -> Bool

Return `true` when relative position `x` lies inside the shadow cylinder behind the body.
"""
@inline @muladd function in_shadow(sc::ShadowCylinder, x::SVector{3, Float64})::Bool
    proj = x[1] * sc.dir[1] + x[2] * sc.dir[2] + x[3] * sc.dir[3]
    proj >= 0.0 && return false
    -proj > sc.half_height && return false
    x2 = x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
    return x2 - proj * proj <= sc.radius * sc.radius
end

"""
    shadow_factor(sc::ShadowCylinder, x::SVector{3, Float64}) -> Float64
    shadow_factor(sc::ShadowCylinder, x::Real, y::Real, z::Real) -> Float64

Return `0.0` inside the shadow cylinder and `1.0` outside.
"""
@inline shadow_factor(sc::ShadowCylinder, x::SVector{3, Float64})::Float64 =
    in_shadow(sc, x) ? 0.0 : 1.0
@inline shadow_factor(sc::ShadowCylinder, x::Real, y::Real, z::Real)::Float64 =
    shadow_factor(sc, SVector{3, Float64}(Float64(x), Float64(y), Float64(z)))

"""
    OpticalDepth(; cross_sections, solar_dir = (1.0, 0.0, 0.0), scale_heights = nothing,
                 chapman = false, min_production = nothing, tau_floor = 6.0e-3,
                 tau_cutoff = 11.5, cos_sza_floor = 5.0e-4, chapman_tau_max = 13.8,
                 chapman_component = nothing)

Code-unit EUV attenuation model supporting multi-component vertical optical depth
``\\tau = \\sum_c n_c \\sigma_c H_c``, grazing-ray terminator transmission, and the
curved-atmosphere Smith & Smith (1972) Chapman function (`chapman = true`).

When `scale_heights` is `nothing` (or has `nothing` entries), per-component local scale
heights are inferred from the neutral component profile. When `chapman = true` and
`chapman_component` is `nothing`, [`build_exosphere_model`](@ref) selects the component
with the largest surface optical depth ``n_c(R_p) \\sigma_c H_c``.
"""
struct OpticalDepth
    sigma::Vector{Float64}
    H::Vector{Float64}
    dir::SVector{3, Float64}
    chapman::Bool
    min_production::Float64
    tau_floor::Float64
    tau_cutoff::Float64
    cos_sza_floor::Float64
    chapman_tau_max::Float64
    chapman_component::Int
end

function OpticalDepth(;
        cross_sections::AbstractVector{<:Real},
        solar_dir::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (1.0, 0.0, 0.0),
        scale_heights = nothing,
        chapman::Bool = false,
        min_production = nothing,
        tau_floor::Real = 6.0e-3,
        tau_cutoff::Real = 11.5,
        cos_sza_floor::Real = 5.0e-4,
        chapman_tau_max::Real = 13.8,
        chapman_component = nothing,
    )
    σ = Float64[Float64(s) for s in cross_sections]
    any(<(0), σ) && throw(ArgumentError("OpticalDepth: cross sections must be >= 0"))
    any(>(0), σ) || throw(ArgumentError("OpticalDepth: needs at least one positive cross section"))
    H = scale_heights === nothing ? fill(-1.0, length(σ)) :
        Float64[h === nothing ? -1.0 : Float64(h) for h in scale_heights]
    length(H) == length(σ) || throw(ArgumentError("OpticalDepth: scale_heights must match cross_sections"))
    length(solar_dir) == 3 || throw(ArgumentError("OpticalDepth: solar_dir must have length 3"))
    d = SVector{3, Float64}(Float64(solar_dir[1]), Float64(solar_dir[2]), Float64(solar_dir[3]))
    dn = norm(d)
    dn > 0.0 || throw(ArgumentError("OpticalDepth: solar_dir must be non-zero"))
    (0.0 < cos_sza_floor < 1.0) || throw(ArgumentError("OpticalDepth: cos_sza_floor must be in (0, 1)"))
    tau_floor >= 0.0 || throw(ArgumentError("OpticalDepth: tau_floor must be >= 0"))
    tau_cutoff > 0.0 || throw(ArgumentError("OpticalDepth: tau_cutoff must be > 0"))
    mp = min_production === nothing ? (chapman ? 1.0e-6 : 1.0e-5) : Float64(min_production)
    (0.0 <= mp <= 1.0) || throw(ArgumentError("OpticalDepth: min_production must be in [0, 1]"))
    return OpticalDepth(
        σ, H, d / dn, chapman, mp, Float64(tau_floor), Float64(tau_cutoff),
        Float64(cos_sza_floor), Float64(chapman_tau_max),
        chapman_component === nothing ? 0 : Int(chapman_component),
    )
end

"""
    ExosphereModel

Multi-component code-unit planetary neutral exosphere with optional [`ShadowCylinder`](@ref)
or [`OpticalDepth`](@ref) EUV attenuation. Construct via [`build_exosphere_model`](@ref).
"""
struct ExosphereModel
    Rp::Float64
    inv_Rp::Float64
    center::SVector{3, Float64}
    comps::Vector{NeutralComponentCode}
    shadow::Union{Nothing, ShadowCylinder}
    optical::Union{Nothing, OpticalDepth}
    inner_radius::Float64
    photochem_radius::Float64
end

ExosphereModel(
    Rp::Float64,
    center::SVector{3, Float64},
    comps::Vector{NeutralComponentCode},
    shadow::Union{Nothing, ShadowCylinder},
    optical::Union{Nothing, OpticalDepth},
    inner_radius::Float64,
    photochem_radius::Float64,
) = ExosphereModel(Rp, Rp > 0.0 ? inv(Rp) : 0.0, center, comps, shadow, optical, inner_radius, photochem_radius)

@inline @muladd function _component_density(
        p::NeutralComponentCode, Rp::Float64, inv_Rp::Float64, r::Float64, inv_r::Float64,
    )
    if p.kind == _EXO_EXPONENTIAL
        return p.H > 0.0 ? p.n0 * exp(-(r - Rp) * p.inv_H) : 0.0
    elseif p.kind == _EXO_POWERLAW
        return r > 0.0 ? p.n0 * _fast_pow(Rp * inv_r, p.k) : 0.0
    else
        return (Rp > 0.0 && r > 0.0) ? p.n0 * exp(-p.H * (inv_Rp - inv_r)) : 0.0
    end
end

@inline function _component_scale_height(p::NeutralComponentCode, r::Float64)
    p.kind == _EXO_EXPONENTIAL && return max(p.H, 0.0)
    p.kind == _EXO_CHAMBERLAIN && return p.H > 0.0 ? r * r * p.inv_H : 0.0
    return p.k > 0.0 ? r / p.k : 0.0
end

@inline function _scale_height_at(comps::Vector{NeutralComponentCode}, ::Nothing, r::Float64, c::Int)
    r <= 0.0 && return 0.0
    return @inbounds _component_scale_height(comps[c], r)
end

@inline function _scale_height_at(comps::Vector{NeutralComponentCode}, od::OpticalDepth, r::Float64, c::Int)
    @inbounds oh = od.H[c]
    oh > 0.0 && return oh
    r <= 0.0 && return 0.0
    return @inbounds _component_scale_height(comps[c], r)
end

"""
    build_exosphere_model(; planet_radius, components, center = (0.0, 0.0, 0.0),
                          shadow = nothing, optical_depth = nothing,
                          inner_radius = 0.0, photochem_radius = 0.0) -> ExosphereModel

Construct and validate a multi-component [`ExosphereModel`](@ref) from a non-empty vector
of [`NeutralComponentCode`](@ref) species. `shadow` ([`ShadowCylinder`](@ref)) and
`optical_depth` ([`OpticalDepth`](@ref)) are mutually exclusive; when neither is provided,
[`photo_attenuation`](@ref) returns `1.0` everywhere. For `optical_depth` with
`chapman = true` and `chapman_component == 0`, the component with the largest surface
vertical optical depth at `planet_radius` is chosen automatically.
"""
function build_exosphere_model(;
        planet_radius::Real,
        components::AbstractVector{NeutralComponentCode},
        center::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (0.0, 0.0, 0.0),
        shadow::Union{Nothing, ShadowCylinder} = nothing,
        optical_depth::Union{Nothing, OpticalDepth} = nothing,
        inner_radius::Real = 0.0,
        photochem_radius::Real = 0.0,
    )
    (shadow !== nothing && optical_depth !== nothing) && throw(
        ArgumentError("ExosphereModel: shadow cylinder and optical-depth attenuation cannot both be set")
    )
    isempty(components) && throw(ArgumentError("ExosphereModel: needs at least one neutral component"))
    comps = NeutralComponentCode[c for c in components]
    Rp_f = Float64(planet_radius)
    inv_Rp = Rp_f > 0.0 ? inv(Rp_f) : 0.0
    od_eff = optical_depth
    if od_eff !== nothing
        length(od_eff.sigma) == length(comps) ||
            throw(ArgumentError("OpticalDepth: one cross section per neutral component required"))
        for i in eachindex(comps)
            od_eff.sigma[i] > 0.0 && _scale_height_at(comps, od_eff, Rp_f, i) <= 0.0 &&
                throw(ArgumentError("OpticalDepth: component $i has a cross section but no usable scale height"))
        end
        if od_eff.chapman && od_eff.chapman_component == 0
            best, bi = -1.0, 1
            for i in eachindex(comps)
                h = _scale_height_at(comps, od_eff, Rp_f, i)
                (od_eff.sigma[i] <= 0.0 || h <= 0.0) && continue
                τ = _component_density(comps[i], Rp_f, inv_Rp, Rp_f, inv_Rp) * od_eff.sigma[i] * h
                τ > best && ((best, bi) = (τ, i))
            end
            od_eff = OpticalDepth(
                od_eff.sigma, od_eff.H, od_eff.dir, true,
                od_eff.min_production, od_eff.tau_floor,
                od_eff.tau_cutoff, od_eff.cos_sza_floor,
                od_eff.chapman_tau_max, bi,
            )
        end
    end
    ctr = SVector{3, Float64}(Float64(center[1]), Float64(center[2]), Float64(center[3]))
    return ExosphereModel(
        Rp_f, inv_Rp, ctr, comps, shadow, od_eff,
        Float64(inner_radius), Float64(photochem_radius),
    )
end

"""
    n_components(exo::ExosphereModel) -> Int

Return the number of neutral components in `exo`.
"""
n_components(exo::ExosphereModel) = length(exo.comps)

"""
    exo_component_density(exo::ExosphereModel, r::Float64, c::Int) -> Float64

Number density of neutral component `c` at radius `r` (zero inside `exo.Rp`).
"""
@inline function exo_component_density(exo::ExosphereModel, r::Float64, c::Int)
    r < exo.Rp && return 0.0
    p = exo.comps[c]
    inv_r = r > 0.0 ? inv(r) : 0.0
    return _component_density(p, exo.Rp, exo.inv_Rp, r, inv_r)
end

"""
    exo_density(exo::ExosphereModel, r::Float64) -> Float64

Total neutral number density (sum over all components) at radius `r`.
"""
@inline function exo_density(exo::ExosphereModel, r::Float64)
    Rp = exo.Rp
    r < Rp && return 0.0
    inv_Rp = exo.inv_Rp
    inv_r = r > 0.0 ? inv(r) : 0.0
    comps = exo.comps
    tot = 0.0
    @inbounds for c in eachindex(comps)
        tot += _component_density(comps[c], Rp, inv_Rp, r, inv_r)
    end
    return tot
end

"""
    exo_scale_height(exo::ExosphereModel, r::Float64, c::Int) -> Float64

Local scale height ``-1 / (d\\ln n / dr)`` of component `c` (or explicit optical-depth
scale height); returns `0.0` when undefined.
"""
@inline function exo_scale_height(exo::ExosphereModel, r::Float64, c::Int)
    1 <= c <= length(exo.comps) || throw(BoundsError(exo.comps, c))
    od = exo.optical
    return od === nothing ? _scale_height_at(exo.comps, nothing, r, c) : _scale_height_at(exo.comps, od, r, c)
end

@inline @muladd function _vertical_optical_depth(exo::ExosphereModel, od::OpticalDepth, r::Float64, only::Int)
    Rp = exo.Rp
    r < Rp && return 0.0
    inv_Rp = exo.inv_Rp
    inv_r = r > 0.0 ? inv(r) : 0.0
    comps = exo.comps
    if only > 0
        @inbounds begin
            σ = od.sigma[only]
            σ <= 0.0 && return 0.0
            h = _scale_height_at(comps, od, r, only)
            h <= 0.0 && return 0.0
            return _component_density(comps[only], Rp, inv_Rp, r, inv_r) * σ * h
        end
    end
    τ = 0.0
    @inbounds for c in eachindex(comps)
        σ = od.sigma[c]
        σ <= 0.0 && continue
        h = _scale_height_at(comps, od, r, c)
        h <= 0.0 && continue
        τ += (_component_density(comps[c], Rp, inv_Rp, r, inv_r) * σ) * h
    end
    return τ
end

@inline @muladd function _grazing_optical_depth(exo::ExosphereModel, od::OpticalDepth, p::Float64)
    Rp = exo.Rp
    p < Rp && return 0.0
    inv_Rp = exo.inv_Rp
    inv_p = p > 0.0 ? inv(p) : 0.0
    two_pi_p = 2π * p
    comps = exo.comps
    τ = 0.0
    @inbounds for c in eachindex(comps)
        σ = od.sigma[c]
        σ <= 0.0 && continue
        h = _scale_height_at(comps, od, p, c)
        h <= 0.0 && continue
        τ += (_component_density(comps[c], Rp, inv_Rp, p, inv_p) * σ) * sqrt(two_pi_p * h)
    end
    return τ
end

"""
    chapman_function(Xp::Float64, cos_sza::Float64) -> Float64

Smith & Smith (1972) Chapman grazing-incidence function for a curved atmosphere with
``X_p = r / H``.
"""
@inline @muladd function chapman_function(Xp::Float64, cosSZA::Float64)
    s = sqrt(0.5 * Xp)
    y = s * abs(cosSZA)
    if cosSZA > 0.0
        y < 8.0 && return (_SQRT_PI * s) * (1.0606963 + 0.5564383 * y) / (1.0619896 + y * (1.7245609 + y))
        y < 100.0 && return (_SQRT_PI * s) * 0.56498823 / (0.6651874 + y)
        return 0.0
    end
    sinSZA = sqrt(max(1.0 - cosSZA * cosSZA, 0.0))
    if y < 8.0
        return (_TWO_SQRT_PI * s) * (
            sqrt(sinSZA) * exp(Xp * (1.0 - sinSZA)) -
                0.5 * (1.0606963 + 0.5564383 * y) / (1.0619896 + y * (1.7245609 + y))
        )
    end
    y < 100.0 && return (_TWO_SQRT_PI * s) * (
        sqrt(sinSZA) * exp(min(100.0, Xp * (1.0 - sinSZA))) -
            0.5 * 0.56498823 / (0.6651874 + y)
    )
    return 0.0
end

"""
    photo_attenuation(exo::ExosphereModel, x::SVector{3, Float64}, r::Float64) -> Float64

EUV ionizing flux attenuation factor in `[0, 1]` at position `x` relative to the planet
center (`r = norm(x)`).
"""
@inline @muladd function photo_attenuation(exo::ExosphereModel, x::SVector{3, Float64}, r::Float64)
    sc = exo.shadow
    od = exo.optical
    if od === nothing
        return (sc !== nothing && in_shadow(sc, x)) ? 0.0 : 1.0
    end
    rs = max(r, 1.0e-3 * max(exo.Rp, 1.0e-300))
    proj = x[1] * od.dir[1] + x[2] * od.dir[2] + x[3] * od.dir[3]
    cosSZA = proj / rs
    mu = max(cosSZA, od.cos_sza_floor)
    if od.chapman
        c = od.chapman_component > 0 ? od.chapman_component : 1
        τv = _vertical_optical_depth(exo, od, r, c)
        τv > od.chapman_tau_max && return od.min_production * max(mu, od.min_production)
        h = _scale_height_at(exo.comps, od, r, c)
        h <= 0.0 && return od.min_production
        ch = chapman_function(r / h, cosSZA)
        ch <= 0.0 && return od.min_production * max(mu, od.min_production)
        return max(exp(-τv * ch), od.min_production)
    end
    τ = if proj <= 0.0
        p = sqrt(max(r * r - proj * proj, 0.0))
        p <= exo.Rp && return od.min_production
        max(_grazing_optical_depth(exo, od, p), od.tau_floor)
    else
        max(_vertical_optical_depth(exo, od, r, 0), od.tau_floor) / mu
    end
    τ >= od.tau_cutoff && return od.min_production
    return max(exp(-τ), od.min_production)
end
