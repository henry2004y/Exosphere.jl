# ==============================================================================
# Shadow Cylinder, EUV Optical Depth, Smith & Smith Chapman Function & ExosphereModel
# ==============================================================================

"""
    ShadowCylinder(dir, radius, half_height)

Geometric cylindrical planetary EUV/optical shadow along unit `dir` (pointing toward the Sun)
of `radius` and downstream `half_height` (defaults to `Inf`).
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

"""
    in_shadow(sc::ShadowCylinder, x::SVector{3, Float64}) -> Bool

Return `true` when relative position `x` lies inside the shadow cylinder behind the body.
"""
@inline function in_shadow(sc::ShadowCylinder, x::SVector{3, Float64})::Bool
    proj = dot(x, sc.dir)
    proj >= 0.0 && return false
    -proj > sc.half_height && return false
    return dot(x, x) - proj * proj <= sc.radius * sc.radius
end

"""
    shadow_factor(sc::ShadowCylinder, x::SVector{3, Float64}) -> Float64

Return `0.0` inside the shadow cylinder and `1.0` outside.
"""
@inline shadow_factor(sc::ShadowCylinder, x::SVector{3, Float64})::Float64 =
    in_shadow(sc, x) ? 0.0 : 1.0
@inline shadow_factor(sc::ShadowCylinder, x::Real, y::Real, z::Real)::Float64 =
    shadow_factor(sc, SVector{3, Float64}(Float64(x), Float64(y), Float64(z)))

"""
    OpticalDepth(sigma, H, dir, chapman, min_production, tau_floor, tau_cutoff,
                 cos_sza_floor, chapman_tau_max, chapman_component)

Code-unit EUV attenuation model supporting multi-component vertical optical depth
``\\tau = \\sum_c n_c \\sigma_c H_c``, grazing-ray terminator transmission, and the
curved-atmosphere Smith & Smith (1972) Chapman function (`chapman = true`).
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
    ExosphereModel(Rp, center, comps, shadow, optical, inner_radius, photochem_radius)

Multi-component code-unit planetary neutral exosphere with optional [`ShadowCylinder`](@ref)
or [`OpticalDepth`](@ref) attenuation.
"""
struct ExosphereModel
    Rp::Float64
    center::SVector{3, Float64}
    comps::Vector{NeutralComponentCode}
    shadow::Union{Nothing, ShadowCylinder}
    optical::Union{Nothing, OpticalDepth}
    inner_radius::Float64
    photochem_radius::Float64
end

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
    if optical_depth !== nothing
        length(optical_depth.sigma) == length(comps) ||
            throw(ArgumentError("OpticalDepth: one cross section per neutral component required"))
    end
    ctr = SVector{3, Float64}(Float64(center[1]), Float64(center[2]), Float64(center[3]))
    exo = ExosphereModel(
        Float64(planet_radius), ctr, comps, shadow, optical_depth,
        Float64(inner_radius), Float64(photochem_radius),
    )
    if optical_depth !== nothing
        for i in eachindex(comps)
            optical_depth.sigma[i] > 0.0 && exo_scale_height(exo, exo.Rp, i) <= 0.0 &&
                throw(ArgumentError("OpticalDepth: component $i has a cross section but no usable scale height"))
        end
        if optical_depth.chapman && optical_depth.chapman_component == 0
            best, bi = -1.0, 1
            for i in eachindex(comps)
                h = exo_scale_height(exo, exo.Rp, i)
                (optical_depth.sigma[i] <= 0.0 || h <= 0.0) && continue
                τ = exo_component_density(exo, exo.Rp, i) * optical_depth.sigma[i] * h
                τ > best && ((best, bi) = (τ, i))
            end
            exo = ExosphereModel(
                exo.Rp, exo.center, comps, shadow,
                OpticalDepth(
                    optical_depth.sigma, optical_depth.H, optical_depth.dir, true,
                    optical_depth.min_production, optical_depth.tau_floor,
                    optical_depth.tau_cutoff, optical_depth.cos_sza_floor,
                    optical_depth.chapman_tau_max, bi,
                ),
                exo.inner_radius, exo.photochem_radius,
            )
        end
    end
    return exo
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
function exo_component_density(exo::ExosphereModel, r::Float64, c::Int)
    r < exo.Rp && return 0.0
    p = exo.comps[c]
    if p.kind == _EXO_EXPONENTIAL
        return p.H > 0.0 ? p.n0 * exp(-(r - exo.Rp) / p.H) : 0.0
    elseif p.kind == _EXO_POWERLAW
        return r > 0.0 ? p.n0 * (exo.Rp / r)^p.k : 0.0
    else
        return (exo.Rp > 0.0 && r > 0.0) ? p.n0 * exp(-p.H * (1.0 / exo.Rp - 1.0 / r)) : 0.0
    end
end

"""
    exo_density(exo::ExosphereModel, r::Float64) -> Float64

Total neutral number density (sum over all components) at radius `r`.
"""
exo_density(exo::ExosphereModel, r::Float64) =
    sum(c -> exo_component_density(exo, r, c), eachindex(exo.comps))

"""
    exo_scale_height(exo::ExosphereModel, r::Float64, c::Int) -> Float64

Local scale height ``-1 / (d\\ln n / dr)`` of component `c` (or explicit optical-depth
scale height); returns `0.0` when undefined.
"""
function exo_scale_height(exo::ExosphereModel, r::Float64, c::Int)
    od = exo.optical
    if od !== nothing && od.H[c] > 0.0
        return od.H[c]
    end
    r <= 0.0 && return 0.0
    p = exo.comps[c]
    p.kind == _EXO_EXPONENTIAL && return max(p.H, 0.0)
    p.kind == _EXO_CHAMBERLAIN && return p.H > 0.0 ? r * r / p.H : 0.0
    return p.k > 0.0 ? r / p.k : 0.0
end

function _vertical_optical_depth(exo::ExosphereModel, od::OpticalDepth, r::Float64, only::Int)
    τ = 0.0
    for c in eachindex(exo.comps)
        (only > 0 && c != only) && continue
        σ = od.sigma[c]
        σ <= 0.0 && continue
        h = exo_scale_height(exo, r, c)
        h <= 0.0 && continue
        τ += exo_component_density(exo, r, c) * σ * h
    end
    return τ
end

function _grazing_optical_depth(exo::ExosphereModel, od::OpticalDepth, p::Float64)
    τ = 0.0
    for c in eachindex(exo.comps)
        σ = od.sigma[c]
        σ <= 0.0 && continue
        h = exo_scale_height(exo, p, c)
        h <= 0.0 && continue
        τ += exo_component_density(exo, p, c) * σ * h * sqrt(2π * p / h)
    end
    return τ
end

"""
    chapman_function(Xp::Float64, cos_sza::Float64) -> Float64

Smith & Smith (1972) Chapman grazing-incidence function for a curved atmosphere with
``X_p = r / H``.
"""
function chapman_function(Xp::Float64, cosSZA::Float64)
    y = sqrt(0.5 * Xp) * abs(cosSZA)
    if cosSZA > 0.0
        y < 8.0 && return sqrt(0.5π * Xp) * (1.0606963 + 0.5564383 * y) / (1.0619896 + 1.7245609 * y + y * y)
        y < 100.0 && return sqrt(0.5π * Xp) * 0.56498823 / (0.6651874 + y)
        return 0.0
    end
    sinSZA = sqrt(max(1.0 - cosSZA^2, 0.0))
    if y < 8.0
        return sqrt(2π * Xp) * (
            sqrt(sinSZA) * exp(Xp * (1.0 - sinSZA)) -
                0.5 * (1.0606963 + 0.5564383 * y) / (1.0619896 + 1.7245609 * y + y * y)
        )
    end
    y < 100.0 && return sqrt(2π * Xp) * (
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
function photo_attenuation(exo::ExosphereModel, x::SVector{3, Float64}, r::Float64)
    sc = exo.shadow
    od = exo.optical
    if od === nothing
        return (sc !== nothing && in_shadow(sc, x)) ? 0.0 : 1.0
    end
    rs = max(r, 1.0e-3 * max(exo.Rp, 1.0e-300))
    proj = dot(x, od.dir)
    cosSZA = proj / rs
    mu = max(cosSZA, od.cos_sza_floor)
    if od.chapman
        c = od.chapman_component > 0 ? od.chapman_component : 1
        τv = _vertical_optical_depth(exo, od, r, c)
        τv > od.chapman_tau_max && return od.min_production * max(mu, od.min_production)
        h = exo_scale_height(exo, r, c)
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
