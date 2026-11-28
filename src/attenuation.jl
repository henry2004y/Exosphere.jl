# ==============================================================================
# EUV Attenuation (ShadowCylinder, OpticalDepth, Chapman, RayOpticalDepth)
# & Unified ExosphereModel (GPU/isbits-Ready, AD-Safe, v0.1.0 Compatible)
# ==============================================================================

Base.broadcastable(a::AbstractAttenuation) = Ref(a)
Base.broadcastable(m::AbstractNeutralModel) = Ref(m)

# ------------------------------------------------------------------------------
# NoAttenuation
# ------------------------------------------------------------------------------

"""
    NoAttenuation() <: AbstractAttenuation

Unattenuated EUV transmission model returning `1` everywhere with zero branching.
"""
struct NoAttenuation <: AbstractAttenuation end

# ------------------------------------------------------------------------------
# ShadowCylinder
# ------------------------------------------------------------------------------

"""
    ShadowCylinder{FT<:AbstractFloat} <: AbstractAttenuation
    ShadowCylinder(; solar_dir = (1.0, 0.0, 0.0), radius = 1.0, half_height = Inf,
                   FT = Float64)

Geometric cylindrical planetary EUV/optical shadow along unit vector `dir` (normalized from
`solar_dir`, pointing toward the Sun) of cross-sectional `radius` and downstream
`half_height` (defaults to `Inf`).
"""
struct ShadowCylinder{FT <: AbstractFloat} <: AbstractAttenuation
    dir::SVector{3, FT}
    radius::FT
    half_height::FT
end

function ShadowCylinder(;
        solar_dir::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            1.0, 0.0, 0.0,
        ),
        radius::Real = 1.0,
        half_height::Real = Inf,
        FT::Type{<:AbstractFloat} = Float64,
    )
    length(solar_dir) == 3 ||
        throw(ArgumentError("ShadowCylinder: solar_dir must have length 3"))
    d = SVector{3, FT}(FT(solar_dir[1]), FT(solar_dir[2]), FT(solar_dir[3]))
    dn = norm(d)
    dn > zero(FT) || throw(ArgumentError("ShadowCylinder: solar_dir must be non-zero"))
    return ShadowCylinder{FT}(d / dn, FT(radius), FT(half_height))
end

"""
    in_shadow(sc::ShadowCylinder, x::SVector{3}) -> Bool

Return `true` when relative position `x` lies inside the shadow cylinder behind the body.
"""
@inline @muladd function in_shadow(sc::ShadowCylinder, x::SVector{3, <:Real})::Bool
    proj = x[1] * sc.dir[1] + x[2] * sc.dir[2] + x[3] * sc.dir[3]
    proj >= zero(proj) && return false
    -proj > sc.half_height && return false
    x2 = x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
    return x2 - proj * proj <= sc.radius * sc.radius
end

@inline function in_shadow(sc::ShadowCylinder, x::AbstractVector{<:Real})::Bool
    return in_shadow(sc, _as_svec3(x))
end

"""
    shadow_factor(sc::ShadowCylinder, x::SVector{3})
    shadow_factor(sc::ShadowCylinder, x::Real, y::Real, z::Real)

Return `0` inside the shadow cylinder and `1` outside.
"""
@inline function shadow_factor(sc::ShadowCylinder{FT}, x::SVector{3, <:Real}) where {FT}
    T = float(promote_type(FT, eltype(x)))
    return in_shadow(sc, x) ? zero(T) : one(T)
end

@inline function shadow_factor(sc::ShadowCylinder{FT}, x::Real, y::Real, z::Real) where {FT}
    T = float(promote_type(FT, typeof(x), typeof(y), typeof(z)))
    return shadow_factor(sc, SVector{3, T}(T(x), T(y), T(z)))
end

# ------------------------------------------------------------------------------
# OpticalDepth & Smith & Smith (1972) Chapman Function
# ------------------------------------------------------------------------------

"""
    OpticalDepth{FT<:AbstractFloat, V<:AbstractVector{FT}} <: AbstractAttenuation
    OpticalDepth(; cross_sections, solar_dir = (1.0, 0.0, 0.0), scale_heights = nothing,
                 chapman = false, min_production = nothing, tau_floor = 6.0e-3,
                 tau_cutoff = 11.5, cos_sza_floor = 5.0e-4, chapman_tau_max = 13.8,
                 chapman_component = nothing, FT = Float64)

Code-unit EUV attenuation model supporting multi-component vertical optical depth
``\\tau = \\sum_c n_c \\sigma_c H_c``, grazing-ray terminator transmission, and the
curved-atmosphere Smith & Smith (1972) Chapman function (`chapman = true`).

Passing a `Tuple` or `SVector` for `cross_sections` produces an `isbits`, GPU-ready
`OpticalDepth{FT, SVector{N, FT}}`; passing a `Vector` preserves dynamic length.
"""
struct OpticalDepth{FT <: AbstractFloat, V <: AbstractVector{FT}} <: AbstractAttenuation
    sigma::V
    H::V
    dir::SVector{3, FT}
    chapman::Bool
    min_production::FT
    tau_floor::FT
    tau_cutoff::FT
    cos_sza_floor::FT
    chapman_tau_max::FT
    chapman_component::Int
end

function OpticalDepth(;
        cross_sections::Union{AbstractVector{<:Real}, Tuple{Vararg{Real}}},
        solar_dir::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            1.0, 0.0, 0.0,
        ),
        scale_heights = nothing,
        chapman::Bool = false,
        min_production = nothing,
        tau_floor::Real = 6.0e-3,
        tau_cutoff::Real = 11.5,
        cos_sza_floor::Real = 5.0e-4,
        chapman_tau_max::Real = 13.8,
        chapman_component = nothing,
        FT::Type{<:AbstractFloat} = Float64,
    )
    N = length(cross_sections)
    is_static = cross_sections isa Tuple || cross_sections isa SVector
    σ = is_static ? SVector{N, FT}(ntuple(i -> FT(cross_sections[i]), Val(N))) :
        FT[FT(s) for s in cross_sections]
    any(<(zero(FT)), σ) &&
        throw(ArgumentError("OpticalDepth: cross sections must be >= 0"))
    any(>(zero(FT)), σ) ||
        throw(ArgumentError("OpticalDepth: needs at least one positive cross section"))
    H = if scale_heights === nothing
        is_static ? SVector{N, FT}(ntuple(_ -> -one(FT), Val(N))) : fill(-one(FT), N)
    else
        length(scale_heights) == N ||
            throw(ArgumentError("OpticalDepth: scale_heights must match cross_sections"))
        is_static ?
            SVector{N, FT}(
                ntuple(
                    i -> scale_heights[i] === nothing ? -one(FT) : FT(scale_heights[i]),
                    Val(N),
                ),
            ) :
            FT[h === nothing ? -one(FT) : FT(h) for h in scale_heights]
    end
    length(solar_dir) == 3 ||
        throw(ArgumentError("OpticalDepth: solar_dir must have length 3"))
    d = SVector{3, FT}(FT(solar_dir[1]), FT(solar_dir[2]), FT(solar_dir[3]))
    dn = norm(d)
    dn > zero(FT) || throw(ArgumentError("OpticalDepth: solar_dir must be non-zero"))
    (zero(FT) < FT(cos_sza_floor) < one(FT)) ||
        throw(ArgumentError("OpticalDepth: cos_sza_floor must be in (0, 1)"))
    tau_floor >= 0 || throw(ArgumentError("OpticalDepth: tau_floor must be >= 0"))
    tau_cutoff > 0 || throw(ArgumentError("OpticalDepth: tau_cutoff must be > 0"))
    mp = min_production === nothing ? (chapman ? FT(1.0e-6) : FT(1.0e-5)) :
        FT(min_production)
    (zero(FT) <= mp <= one(FT)) ||
        throw(ArgumentError("OpticalDepth: min_production must be in [0, 1]"))
    return OpticalDepth{FT, typeof(σ)}(
        σ, H, d / dn, chapman, mp, FT(tau_floor), FT(tau_cutoff),
        FT(cos_sza_floor), FT(chapman_tau_max),
        chapman_component === nothing ? 0 : Int(chapman_component),
    )
end

@inline function _static_optical(od::OpticalDepth{FT, <:SVector{N, FT}}, ::Val{N}, bi::Int) where {FT, N}
    return OpticalDepth{FT, SVector{N, FT}}(
        od.sigma, od.H, od.dir, od.chapman, od.min_production, od.tau_floor,
        od.tau_cutoff, od.cos_sza_floor, od.chapman_tau_max, bi,
    )
end

@inline function _static_optical(od::OpticalDepth{FT}, ::Val{N}, bi::Int) where {FT, N}
    σ = SVector{N, FT}(ntuple(i -> @inbounds(od.sigma[i]), Val(N)))
    H = SVector{N, FT}(ntuple(i -> @inbounds(od.H[i]), Val(N)))
    return OpticalDepth{FT, SVector{N, FT}}(
        σ, H, od.dir, od.chapman, od.min_production, od.tau_floor,
        od.tau_cutoff, od.cos_sza_floor, od.chapman_tau_max, bi,
    )
end

@inline function _with_chapman_component(od::OpticalDepth{FT, V}, bi::Int) where {FT, V}
    return OpticalDepth{FT, V}(
        od.sigma, od.H, od.dir, od.chapman, od.min_production, od.tau_floor,
        od.tau_cutoff, od.cos_sza_floor, od.chapman_tau_max, bi,
    )
end

# ------------------------------------------------------------------------------
# RayOpticalDepth (3D Asymmetric & Dynamic Neutral Column Integration)
# ------------------------------------------------------------------------------

# Gauss-Legendre nodes & weights on [0, 1] (mapped from [-1, 1] by u = (t + 1) / 2).
const _GL8_U = SVector{8, Float64}(
    0.019855071751231884, 0.10166676129318663, 0.2372337950418355, 0.4082826787521751,
    0.5917173212478249, 0.7627662049581645, 0.8983332387068134, 0.9801449282487681,
)
const _GL8_W = SVector{8, Float64}(
    0.05061426814518813, 0.11119051722668723, 0.15685332293894364, 0.18134189168918099,
    0.18134189168918099, 0.15685332293894364, 0.11119051722668723, 0.05061426814518813,
)

const _GL16_U = SVector{16, Float64}(
    0.005299532504175034, 0.027712488463383712, 0.06718439880608413, 0.12229779582249848,
    0.19106187779867814, 0.2709916111713863, 0.35919822461037054, 0.4524937450811813,
    0.5475062549188187, 0.6408017753896295, 0.7290083888286137, 0.8089381222013219,
    0.8777022041775015, 0.9328156011939159, 0.9722875115366163, 0.994700467495825,
)
const _GL16_W = SVector{16, Float64}(
    0.013576229705877047, 0.031126761969323946, 0.04757925584124639, 0.062314485627766936,
    0.07479799440828837, 0.08457825969750127, 0.09130170752246179, 0.09472530522753425,
    0.09472530522753425, 0.09130170752246179, 0.08457825969750127, 0.07479799440828837,
    0.062314485627766936, 0.04757925584124639, 0.031126761969323946, 0.013576229705877047,
)

@inline _gl_nodes(::Val{8}, ::Type{FT}) where {FT} = (SVector{8, FT}(_GL8_U), SVector{8, FT}(_GL8_W))
@inline _gl_nodes(::Val{16}, ::Type{FT}) where {FT} = (SVector{16, FT}(_GL16_U), SVector{16, FT}(_GL16_W))

"""
    RayOpticalDepth{FT<:AbstractFloat, V<:AbstractVector{FT}, NQ} <: AbstractAttenuation
    RayOpticalDepth(; cross_sections, solar_dir = (1.0, 0.0, 0.0), s_max = 10.0,
                    planet_radius = 1.0, center = (0.0, 0.0, 0.0), min_production = 1.0e-6,
                    tau_cutoff = 13.8, n_quad = 16, FT = Float64)

Line-of-sight EUV column-density attenuation along the Sunward ray
``\\mathbf{x} + s\\hat{\\mathbf{s}}`` for `s ∈ [0, s_max]`, using quadratic-stretched
`n_quad`-point (`8` or `16`) Gauss–Legendre quadrature with solid-body shadowing at
`planet_radius`. Works with arbitrary 3D asymmetric profiles and dynamic two-way
[`GriddedNeutralField`](@ref) models with zero heap allocations.
"""
struct RayOpticalDepth{FT <: AbstractFloat, V <: AbstractVector{FT}, NQ} <: AbstractAttenuation
    sigma::V
    dir::SVector{3, FT}
    s_max::FT
    planet_radius::FT
    center::SVector{3, FT}
    min_production::FT
    tau_cutoff::FT
end

function RayOpticalDepth(;
        cross_sections::Union{AbstractVector{<:Real}, Tuple{Vararg{Real}}},
        solar_dir::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            1.0, 0.0, 0.0,
        ),
        s_max::Real = 10.0,
        planet_radius::Real = 1.0,
        center::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            0.0, 0.0, 0.0,
        ),
        min_production::Real = 1.0e-6,
        tau_cutoff::Real = 13.8,
        n_quad::Integer = 16,
        FT::Type{<:AbstractFloat} = Float64,
    )
    (n_quad == 8 || n_quad == 16) ||
        throw(ArgumentError("RayOpticalDepth: n_quad must be 8 or 16 (got $n_quad)"))
    N = length(cross_sections)
    is_static = cross_sections isa Tuple || cross_sections isa SVector
    σ = is_static ? SVector{N, FT}(ntuple(i -> FT(cross_sections[i]), Val(N))) :
        FT[FT(s) for s in cross_sections]
    any(<(zero(FT)), σ) &&
        throw(ArgumentError("RayOpticalDepth: cross sections must be >= 0"))
    any(>(zero(FT)), σ) ||
        throw(ArgumentError("RayOpticalDepth: needs at least one positive cross section"))
    length(solar_dir) == 3 ||
        throw(ArgumentError("RayOpticalDepth: solar_dir must have length 3"))
    d = SVector{3, FT}(FT(solar_dir[1]), FT(solar_dir[2]), FT(solar_dir[3]))
    dn = norm(d)
    dn > zero(FT) || throw(ArgumentError("RayOpticalDepth: solar_dir must be non-zero"))
    s_max > 0 || throw(ArgumentError("RayOpticalDepth: s_max must be > 0"))
    planet_radius >= 0 || throw(ArgumentError("RayOpticalDepth: planet_radius must be >= 0"))
    (0 <= min_production <= 1) ||
        throw(ArgumentError("RayOpticalDepth: min_production must be in [0, 1]"))
    tau_cutoff > 0 || throw(ArgumentError("RayOpticalDepth: tau_cutoff must be > 0"))
    ctr = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    NQ = Int(n_quad)
    return RayOpticalDepth{FT, typeof(σ), NQ}(
        σ, d / dn, FT(s_max), FT(planet_radius), ctr, FT(min_production), FT(tau_cutoff),
    )
end

@inline function _ray_attenuation(
        model::AbstractNeutralModel,
        rod::RayOpticalDepth{FT, V, NQ},
        x::SVector{3, <:Real},
        t,
    ) where {FT, V, NQ}
    T = float(promote_type(FT, eltype(x)))
    # Check solid-body occultation along Sunward ray
    d = SVector{3, T}(x[1] - rod.center[1], x[2] - rod.center[2], x[3] - rod.center[3])
    proj = d[1] * rod.dir[1] + d[2] * rod.dir[2] + d[3] * rod.dir[3]
    if proj < zero(T) && rod.planet_radius > zero(FT)
        r2 = d[1] * d[1] + d[2] * d[2] + d[3] * d[3]
        p2 = max(r2 - proj * proj, zero(T))
        p2 <= rod.planet_radius * rod.planet_radius && return T(rod.min_production)
    end
    nodes, weights = _gl_nodes(Val(NQ), FT)
    tau = zero(T)
    two_smax = T(2) * rod.s_max
    @inbounds for q in 1:NQ
        u = nodes[q]
        s = rod.s_max * (u * u)
        ds = two_smax * u * weights[q]
        xq = SVector{3, T}(
            x[1] + s * rod.dir[1],
            x[2] + s * rod.dir[2],
            x[3] + s * rod.dir[3],
        )
        col_rate = _ray_extinction_at(model, rod.sigma, xq, t)
        tau += col_rate * ds
    end
    tau >= rod.tau_cutoff && return T(rod.min_production)
    return max(exp(-tau), T(rod.min_production))
end

@inline function _ray_extinction_at(
        model::AbstractNeutralModel, sigma::AbstractVector, xq::SVector{3}, t,
    )
    T = float(promote_type(eltype(sigma), eltype(xq)))
    ext = zero(T)
    @inbounds for c in eachindex(sigma)
        σ = sigma[c]
        σ <= zero(σ) && continue
        ext += σ * neutral_density(model, xq; t = t, species = c)
    end
    return ext
end

# ------------------------------------------------------------------------------
# ExosphereModel (Unified Tuple-isbits & Legacy Vector-Compatible)
# ------------------------------------------------------------------------------

"""
    ExosphereModel{FT<:AbstractFloat, Comps, Atten<:AbstractAttenuation} <: AbstractNeutralModel
    ExosphereModel(populations::NeutralPopulation...; planet_radius = 1.0,
                   center = (0.0, 0.0, 0.0), attenuation = NoAttenuation(),
                   shadow = nothing, optical_depth = nothing,
                   inner_radius = 0.0, photochem_radius = 0.0, FT = Float64)

Multi-component planetary neutral exosphere with configurable EUV attenuation
([`NoAttenuation`](@ref), [`ShadowCylinder`](@ref), [`OpticalDepth`](@ref), or
[`RayOpticalDepth`](@ref)). Passing a tuple or varargs of [`NeutralPopulation`](@ref)
instances constructs an `isbits`, GPU-ready model; [`build_exosphere_model`](@ref) also
supports legacy `Vector{NeutralComponentCode}` specifications.
"""
struct ExosphereModel{
        FT <: AbstractFloat,
        Comps,
        Atten <: AbstractAttenuation,
    } <: AbstractNeutralModel
    Rp::FT
    inv_Rp::FT
    center::SVector{3, FT}
    comps::Comps
    attenuation::Atten
    inner_radius::FT
    photochem_radius::FT
end

# Constant-folded backwards-compatible property access for `.shadow` and `.optical`
@inline function Base.getproperty(exo::ExosphereModel, s::Symbol)
    if s === :shadow
        a = getfield(exo, :attenuation)
        return a isa ShadowCylinder ? a : nothing
    elseif s === :optical
        a = getfield(exo, :attenuation)
        return a isa OpticalDepth ? a : nothing
    end
    return getfield(exo, s)
end

Base.propertynames(::ExosphereModel, ::Bool = false) = (
    :Rp, :inv_Rp, :center, :comps, :attenuation, :shadow, :optical,
    :inner_radius, :photochem_radius,
)

# Legacy 7-argument positional constructor
function ExosphereModel(
        Rp::Real,
        center::SVector{3, <:Real},
        comps,
        shadow::Union{Nothing, ShadowCylinder},
        optical::Union{Nothing, OpticalDepth},
        inner_radius::Real,
        photochem_radius::Real,
    )
    (shadow !== nothing && optical !== nothing) && throw(
        ArgumentError("ExosphereModel: shadow cylinder and optical-depth attenuation cannot both be set")
    )
    FT = float(promote_type(typeof(Rp), eltype(center), typeof(inner_radius), typeof(photochem_radius)))
    Rp_f = FT(Rp)
    inv_Rp = Rp_f > zero(FT) ? inv(Rp_f) : zero(FT)
    atten = optical !== nothing ? optical : (shadow !== nothing ? shadow : NoAttenuation())
    ctr = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    return ExosphereModel{FT, typeof(comps), typeof(atten)}(
        Rp_f, inv_Rp, ctr, comps, atten, FT(inner_radius), FT(photochem_radius),
    )
end

function ExosphereModel(
        pop::NeutralPopulation,
        pops::NeutralPopulation...;
        planet_radius::Real = 1.0,
        center::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            0.0, 0.0, 0.0,
        ),
        attenuation::Union{Nothing, AbstractAttenuation} = nothing,
        shadow::Union{Nothing, ShadowCylinder} = nothing,
        optical_depth::Union{Nothing, OpticalDepth} = nothing,
        inner_radius::Real = 0.0,
        photochem_radius::Real = 0.0,
        FT::Type{<:AbstractFloat} = Float64,
    )
    return build_exosphere_model(;
        planet_radius = planet_radius,
        components = (pop, pops...),
        center = center,
        attenuation = attenuation,
        shadow = shadow,
        optical_depth = optical_depth,
        inner_radius = inner_radius,
        photochem_radius = photochem_radius,
        FT = FT,
    )
end

# ------------------------------------------------------------------------------
# Component Density, Scale Height & Tuple-Unrolled Traversal Helpers
# ------------------------------------------------------------------------------

@inline @muladd function _component_density(
        p::NeutralComponentCode, Rp::Real, inv_Rp::Real, r::Real, inv_r::Real,
    )
    T = float(promote_type(typeof(Rp), typeof(r)))
    if p.kind == _EXO_EXPONENTIAL
        return p.H > 0.0 ? T(p.n0) * exp(-(T(r) - T(Rp)) * p.inv_H) : zero(T)
    elseif p.kind == _EXO_POWERLAW
        return r > zero(r) ? T(p.n0) * _fast_pow(T(Rp) * inv_r, p.k) : zero(T)
    else
        return (Rp > zero(Rp) && r > zero(r)) ?
            T(p.n0) * exp(-p.H * (T(inv_Rp) - inv_r)) : zero(T)
    end
end

@inline function _component_density(
        p::NeutralPopulation, ::Real, ::Real, r::Real, ::Real,
    )
    return neutral_density(p.profile, r)
end

@inline function _component_density_3d(
        p::NeutralComponentCode, Rp::Real, inv_Rp::Real, ::SVector{3}, r::Real, inv_r::Real, t,
    )
    return _component_density(p, Rp, inv_Rp, r, inv_r)
end

@inline function _component_density_3d(
        p::NeutralPopulation, ::Real, ::Real, x::SVector{3}, ::Real, ::Real, t,
    )
    return neutral_density(p.profile, x; t = t)
end

@inline function _component_scale_height(p::NeutralComponentCode, r::Real)
    T = float(typeof(r))
    p.kind == _EXO_EXPONENTIAL && return T(max(p.H, 0.0))
    p.kind == _EXO_CHAMBERLAIN && return p.H > 0.0 ? (T(r) * T(r)) * p.inv_H : zero(T)
    return p.k > 0.0 ? T(r) / p.k : zero(T)
end

@inline _component_scale_height(p::NeutralPopulation, r::Real) =
    local_scale_height(p.profile, r)

@inline _component_scale_height_3d(p::NeutralComponentCode, ::SVector{3}, r::Real, t) =
    _component_scale_height(p, r)

@inline _component_scale_height_3d(p::NeutralPopulation, x::SVector{3}, ::Real, t) =
    local_scale_height(p.profile, x; t = t)

@inline _component_mass(::NeutralComponentCode, ::Type{FT}) where {FT} = one(FT)
@inline _component_mass(p::NeutralPopulation, ::Type{FT}) where {FT} = FT(p.species.m)

@inline function _component_state(
        p::NeutralComponentCode, Rp::Real, inv_Rp::Real, x::SVector{3}, r::Real, inv_r::Real,
        t, kB::Number,
    )
    T = float(promote_type(typeof(Rp), eltype(x)))
    n = r < Rp ? zero(T) : _component_density(p, Rp, inv_Rp, r, inv_r)
    return NeutralState(n, SVector{3, T}(zero(T), zero(T), zero(T)), T(p.T0))
end

@inline function _component_state(
        p::NeutralPopulation, Rp::Real, ::Real, x::SVector{3}, r::Real, ::Real, t, kB::Number,
    )
    st = neutral_state(p, x; t = t, kB = kB)
    r < Rp && return NeutralState(zero(st.n), st.u, st.T)
    return st
end

@inline function _component_vdf(p::NeutralComponentCode, ::SVector{3}, t, kB::Number)
    return Maxwellian(NeutralSpecies(; m = 1.0); T = p.T0, kB = kB)
end

@inline function _component_vdf(p::NeutralPopulation, x::SVector{3}, t, kB::Number)
    return local_vdf(p, x; t = t, kB = kB)
end

# Tuple recursive dispatch helpers (0 heap allocations, GPU-safe, @inferred)
@inline function _at_component(f::F, comps::Tuple{C, Vararg}, c::Int, idx::Int) where {F, C}
    idx == c && return f(comps[1])
    return _at_component(f, Base.tail(comps), c, idx + 1)
end

@inline _at_component(::F, ::Tuple{}, c::Int, ::Int) where {F} = throw(BoundsError((), c))
@inline _at_component(f::F, comps::Tuple, c::Int) where {F} = _at_component(f, comps, c, 1)

@inline function _at_component_vec(f::F, comps::AbstractVector, c::Int) where {F}
    1 <= c <= length(comps) || throw(BoundsError(comps, c))
    return @inbounds f(comps[c])
end

@inline function _scale_height_at(
        comps::AbstractVector, ::Union{Nothing, NoAttenuation, ShadowCylinder, RayOpticalDepth},
        r::Real, c::Int,
    )
    T = float(typeof(r))
    r <= zero(r) && return zero(T)
    return @inbounds _component_scale_height(comps[c], r)
end

@inline function _scale_height_at(comps::AbstractVector, od::OpticalDepth, r::Real, c::Int)
    T = float(promote_type(eltype(od.H), typeof(r)))
    @inbounds oh = od.H[c]
    oh > zero(oh) && return T(oh)
    r <= zero(r) && return zero(T)
    return @inbounds T(_component_scale_height(comps[c], r))
end

@inline function _scale_height_at(
        comps::Tuple, ::Union{Nothing, NoAttenuation, ShadowCylinder, RayOpticalDepth},
        r::Real, c::Int,
    )
    T = float(typeof(r))
    r <= zero(r) && return zero(T)
    return T(_at_component(comp -> _component_scale_height(comp, r), comps, c))
end

@inline function _scale_height_at(comps::Tuple, od::OpticalDepth, r::Real, c::Int)
    T = float(promote_type(eltype(od.H), typeof(r)))
    1 <= c <= length(comps) || throw(BoundsError(comps, c))
    @inbounds oh = od.H[c]
    oh > zero(oh) && return T(oh)
    r <= zero(r) && return zero(T)
    return T(_at_component(comp -> _component_scale_height(comp, r), comps, c))
end

@inline _scale_height_comp(comp, ::Union{Nothing, NoAttenuation, ShadowCylinder, RayOpticalDepth}, r::Real, ::Int) =
    r <= zero(r) ? zero(float(typeof(r))) : _component_scale_height(comp, r)

@inline function _scale_height_comp(comp, od::OpticalDepth, r::Real, c::Int)
    T = float(promote_type(eltype(od.H), typeof(r)))
    @inbounds oh = od.H[c]
    oh > zero(oh) && return T(oh)
    r <= zero(r) && return zero(T)
    return T(_component_scale_height(comp, r))
end

# Normalize one entry in `components` tuple/vector into a NeutralPopulation or NeutralComponentCode
function _normalize_component(c::NeutralComponentCode, ::Real, ::NTuple{3, Real}, ::Type{FT}) where {FT}
    return c
end

function _normalize_component(c::NeutralPopulation, ::Real, ::NTuple{3, Real}, ::Type{FT}) where {FT}
    return c
end

function _normalize_component(
        p::AbstractNeutralProfile, ::Real, ::NTuple{3, Real}, ::Type{FT},
    ) where {FT}
    return NeutralPopulation(NeutralSpecies(; m = one(FT)), p; temperature = ConstantTemperature(one(FT)))
end

"""
    build_exosphere_model(; planet_radius, components, center = (0.0, 0.0, 0.0),
                          attenuation = nothing, shadow = nothing, optical_depth = nothing,
                          inner_radius = 0.0, photochem_radius = 0.0,
                          FT = Float64) -> ExosphereModel

Construct and validate a multi-component [`ExosphereModel`](@ref) from either a non-empty
vector of [`NeutralComponentCode`](@ref) (legacy `v0.1.0` path) or a tuple/vector of
[`NeutralPopulation`](@ref) / [`AbstractNeutralProfile`](@ref) entries.

At most one of `attenuation`, `shadow` ([`ShadowCylinder`](@ref)), or `optical_depth`
([`OpticalDepth`](@ref)) may be provided; when none is given, [`NoAttenuation`](@ref) is
used and [`photo_attenuation`](@ref) returns `1` everywhere. Passing a `Tuple` for
`components` produces a 100% `isbits`, GPU-ready `ExosphereModel` (promoting `OpticalDepth`
cross-section vectors to `SVector` automatically).
"""
function build_exosphere_model(;
        planet_radius::Real,
        components::Union{AbstractVector, Tuple},
        center::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            0.0, 0.0, 0.0,
        ),
        attenuation::Union{Nothing, AbstractAttenuation} = nothing,
        shadow::Union{Nothing, ShadowCylinder} = nothing,
        optical_depth::Union{Nothing, OpticalDepth} = nothing,
        inner_radius::Real = 0.0,
        photochem_radius::Real = 0.0,
        FT::Type{<:AbstractFloat} = Float64,
    )
    n_atten = (attenuation !== nothing) + (shadow !== nothing) + (optical_depth !== nothing)
    n_atten > 1 && throw(
        ArgumentError("ExosphereModel: shadow cylinder and optical-depth attenuation cannot both be set")
    )
    isempty(components) &&
        throw(ArgumentError("ExosphereModel: needs at least one neutral component"))
    ctr_t = (FT(center[1]), FT(center[2]), FT(center[3]))
    Rp_f = FT(planet_radius)
    inv_Rp = Rp_f > zero(FT) ? inv(Rp_f) : zero(FT)
    comps = if components isa Tuple
        map(c -> _normalize_component(c, Rp_f, ctr_t, FT), components)
    elseif components isa AbstractVector{NeutralComponentCode}
        NeutralComponentCode[c for c in components]
    else
        [_normalize_component(c, Rp_f, ctr_t, FT) for c in components]
    end
    raw_atten = attenuation !== nothing ? attenuation :
        optical_depth !== nothing ? optical_depth :
        shadow !== nothing ? shadow : NoAttenuation()
    atten_eff = if raw_atten isa OpticalDepth
        od_eff = raw_atten
        length(od_eff.sigma) == length(comps) ||
            throw(ArgumentError("OpticalDepth: one cross section per neutral component required"))
        for i in 1:length(comps)
            od_eff.sigma[i] > 0 && _scale_height_at(comps, od_eff, Rp_f, i) <= 0 &&
                throw(
                ArgumentError(
                    "OpticalDepth: component $i has a cross section but no usable scale height",
                )
            )
        end
        bi = od_eff.chapman_component
        if od_eff.chapman && bi == 0
            best = -one(FT)
            bi = 1
            for i in 1:length(comps)
                h = _scale_height_at(comps, od_eff, Rp_f, i)
                (od_eff.sigma[i] <= 0 || h <= 0) && continue
                n_i = comps isa Tuple ?
                    _at_component(c -> _component_density(c, Rp_f, inv_Rp, Rp_f, inv_Rp), comps, i) :
                    _component_density(comps[i], Rp_f, inv_Rp, Rp_f, inv_Rp)
                τ = n_i * od_eff.sigma[i] * h
                τ > best && ((best, bi) = (τ, i))
            end
        end
        comps isa Tuple ? _static_optical(od_eff, Val(length(comps)), bi) :
            _with_chapman_component(od_eff, bi)
    elseif raw_atten isa RayOpticalDepth
        length(raw_atten.sigma) == length(comps) ||
            throw(ArgumentError("RayOpticalDepth: one cross section per neutral component required"))
        raw_atten
    else
        raw_atten
    end
    ctr = SVector{3, FT}(ctr_t[1], ctr_t[2], ctr_t[3])
    return ExosphereModel{FT, typeof(comps), typeof(atten_eff)}(
        Rp_f, inv_Rp, ctr, comps, atten_eff, FT(inner_radius), FT(photochem_radius),
    )
end

# ------------------------------------------------------------------------------
# ExosphereModel Query Methods
# ------------------------------------------------------------------------------

"""
    n_components(exo::AbstractNeutralModel) -> Int

Return the number of neutral components (species/populations) in `exo`.
"""
@inline n_components(exo::ExosphereModel) = length(exo.comps)

@inline function species_mass(exo::ExosphereModel{FT, <:Tuple}, c::Integer = 1) where {FT}
    return _at_component(comp -> _component_mass(comp, FT), exo.comps, Int(c))
end

@inline function species_mass(exo::ExosphereModel{FT, <:AbstractVector}, c::Integer = 1) where {FT}
    return _at_component_vec(comp -> _component_mass(comp, FT), exo.comps, Int(c))
end

"""
    exo_component_density(exo::ExosphereModel, r::Real, c::Integer)

Number density of neutral component `c` at radius `r` (zero inside `exo.Rp`).
"""
@inline function exo_component_density(exo::ExosphereModel{FT, <:Tuple}, r::Real, c::Integer) where {FT}
    T = float(promote_type(FT, typeof(r)))
    r < exo.Rp && return zero(T)
    inv_r = r > zero(r) ? inv(T(r)) : zero(T)
    return T(_at_component(p -> _component_density(p, exo.Rp, exo.inv_Rp, r, inv_r), exo.comps, Int(c)))
end

@inline function exo_component_density(exo::ExosphereModel{FT, <:AbstractVector}, r::Real, c::Integer) where {FT}
    T = float(promote_type(FT, typeof(r)))
    r < exo.Rp && return zero(T)
    p = exo.comps[Int(c)]
    inv_r = r > zero(r) ? inv(T(r)) : zero(T)
    return T(_component_density(p, exo.Rp, exo.inv_Rp, r, inv_r))
end

# Tuple-unrolled radial and 3D density sums
@inline _sum_exo_r(comps::Tuple{C}, Rp, inv_Rp, r, inv_r) where {C} =
    _component_density(comps[1], Rp, inv_Rp, r, inv_r)
@inline _sum_exo_r(comps::Tuple{C, Vararg}, Rp, inv_Rp, r, inv_r) where {C} =
    _component_density(comps[1], Rp, inv_Rp, r, inv_r) +
    _sum_exo_r(Base.tail(comps), Rp, inv_Rp, r, inv_r)

@inline _sum_exo_3d(comps::Tuple{C}, Rp, inv_Rp, x, r, inv_r, t) where {C} =
    _component_density_3d(comps[1], Rp, inv_Rp, x, r, inv_r, t)
@inline _sum_exo_3d(comps::Tuple{C, Vararg}, Rp, inv_Rp, x, r, inv_r, t) where {C} =
    _component_density_3d(comps[1], Rp, inv_Rp, x, r, inv_r, t) +
    _sum_exo_3d(Base.tail(comps), Rp, inv_Rp, x, r, inv_r, t)

"""
    exo_density(exo::ExosphereModel, r::Real)

Total neutral number density (sum over all components) at radius `r`.
"""
@inline function exo_density(exo::ExosphereModel{FT, <:Tuple}, r::Real) where {FT}
    T = float(promote_type(FT, typeof(r)))
    Rp = exo.Rp
    r < Rp && return zero(T)
    inv_r = r > zero(r) ? inv(T(r)) : zero(T)
    return T(_sum_exo_r(exo.comps, Rp, exo.inv_Rp, T(r), inv_r))
end

@inline function exo_density(exo::ExosphereModel{FT, <:AbstractVector}, r::Real) where {FT}
    T = float(promote_type(FT, typeof(r)))
    Rp = exo.Rp
    r < Rp && return zero(T)
    inv_Rp = exo.inv_Rp
    inv_r = r > zero(r) ? inv(T(r)) : zero(T)
    comps = exo.comps
    tot = zero(T)
    @inbounds for c in eachindex(comps)
        tot += _component_density(comps[c], Rp, inv_Rp, T(r), inv_r)
    end
    return tot
end

"""
    exo_scale_height(exo::ExosphereModel, r::Real, c::Integer)

Local scale height ``-1 / (d\\ln n / dr)`` of component `c` (or explicit optical-depth
scale height); returns `0` when undefined.
"""
@inline function exo_scale_height(exo::ExosphereModel, r::Real, c::Integer)
    ci = Int(c)
    1 <= ci <= length(exo.comps) || throw(BoundsError(exo.comps, ci))
    return _scale_height_at(exo.comps, exo.attenuation, r, ci)
end

# AbstractNeutralModel query methods for ExosphereModel
@inline function neutral_density(
        exo::ExosphereModel, r::Real; t = nothing, species::Union{Nothing, Integer} = nothing,
    )
    return species === nothing ? exo_density(exo, r) : exo_component_density(exo, r, species)
end

@inline function neutral_density(
        exo::ExosphereModel{FT, <:Tuple}, x::AbstractVector{<:Real};
        t = nothing, species::Union{Nothing, Integer} = nothing,
    ) where {FT}
    xv = _as_svec3(x)
    d = _relative_pos(exo.center, xv)
    r = _safe_norm(d)
    T = float(promote_type(FT, typeof(r)))
    r < exo.Rp && return zero(T)
    inv_r = r > zero(r) ? inv(T(r)) : zero(T)
    if species === nothing
        return T(_sum_exo_3d(exo.comps, exo.Rp, exo.inv_Rp, xv, T(r), inv_r, t))
    end
    return T(
        _at_component(
            p -> _component_density_3d(p, exo.Rp, exo.inv_Rp, xv, T(r), inv_r, t),
            exo.comps,
            Int(species),
        ),
    )
end

@inline function neutral_density(
        exo::ExosphereModel{FT, <:AbstractVector}, x::AbstractVector{<:Real};
        t = nothing, species::Union{Nothing, Integer} = nothing,
    ) where {FT}
    xv = _as_svec3(x)
    d = _relative_pos(exo.center, xv)
    r = _safe_norm(d)
    T = float(promote_type(FT, typeof(r)))
    r < exo.Rp && return zero(T)
    inv_r = r > zero(r) ? inv(T(r)) : zero(T)
    if species !== nothing
        p = exo.comps[Int(species)]
        return T(_component_density_3d(p, exo.Rp, exo.inv_Rp, xv, T(r), inv_r, t))
    end
    tot = zero(T)
    @inbounds for c in eachindex(exo.comps)
        tot += _component_density_3d(exo.comps[c], exo.Rp, exo.inv_Rp, xv, T(r), inv_r, t)
    end
    return tot
end

@inline function local_scale_height(
        exo::ExosphereModel, r::Real; t = nothing, species::Integer = 1,
    )
    return exo_scale_height(exo, r, species)
end

@inline function local_scale_height(
        exo::ExosphereModel, x::AbstractVector{<:Real}; t = nothing, species::Integer = 1,
    )
    xv = _as_svec3(x)
    r = _safe_norm(_relative_pos(exo.center, xv))
    return exo_scale_height(exo, r, species)
end

@inline function neutral_state(
        exo::ExosphereModel{FT, <:Tuple}, x::AbstractVector{<:Real};
        t = nothing, species::Integer = 1, kB::Number = 1,
    ) where {FT}
    xv = _as_svec3(x)
    d = _relative_pos(exo.center, xv)
    r = _safe_norm(d)
    T = float(promote_type(FT, typeof(r)))
    inv_r = r > zero(r) ? inv(T(r)) : zero(T)
    return _at_component(
        p -> _component_state(p, exo.Rp, exo.inv_Rp, xv, T(r), inv_r, t, kB),
        exo.comps,
        Int(species),
    )
end

@inline function neutral_state(
        exo::ExosphereModel{FT, <:AbstractVector}, x::AbstractVector{<:Real};
        t = nothing, species::Integer = 1, kB::Number = 1,
    ) where {FT}
    xv = _as_svec3(x)
    d = _relative_pos(exo.center, xv)
    r = _safe_norm(d)
    T = float(promote_type(FT, typeof(r)))
    inv_r = r > zero(r) ? inv(T(r)) : zero(T)
    p = exo.comps[Int(species)]
    return _component_state(p, exo.Rp, exo.inv_Rp, xv, T(r), inv_r, t, kB)
end

@inline function local_vdf(
        exo::ExosphereModel{FT, <:Tuple}, x::AbstractVector{<:Real};
        t = nothing, species::Integer = 1, kB::Number = 1,
    ) where {FT}
    xv = _as_svec3(x)
    return _at_component(p -> _component_vdf(p, xv, t, kB), exo.comps, Int(species))
end

@inline function local_vdf(
        exo::ExosphereModel{FT, <:AbstractVector}, x::AbstractVector{<:Real};
        t = nothing, species::Integer = 1, kB::Number = 1,
    ) where {FT}
    xv = _as_svec3(x)
    return _component_vdf(exo.comps[Int(species)], xv, t, kB)
end

# ------------------------------------------------------------------------------
# Optical Depth, Chapman Function & Photo-Attenuation
# ------------------------------------------------------------------------------

@inline function _sum_vert_tau(
        comps::Tuple{C}, od::OpticalDepth, Rp, inv_Rp, r, inv_r, idx::Int,
    ) where {C}
    @inbounds σ = od.sigma[idx]
    T = float(promote_type(typeof(σ), typeof(r)))
    σ <= zero(σ) && return zero(T)
    h = _scale_height_comp(comps[1], od, r, idx)
    h <= zero(h) && return zero(T)
    return T((_component_density(comps[1], Rp, inv_Rp, r, inv_r) * σ) * h)
end

@inline function _sum_vert_tau(
        comps::Tuple{C, Vararg}, od::OpticalDepth, Rp, inv_Rp, r, inv_r, idx::Int,
    ) where {C}
    return _sum_vert_tau((comps[1],), od, Rp, inv_Rp, r, inv_r, idx) +
        _sum_vert_tau(Base.tail(comps), od, Rp, inv_Rp, r, inv_r, idx + 1)
end

@inline function _sum_graz_tau(
        comps::Tuple{C}, od::OpticalDepth, Rp, inv_Rp, p, inv_p, two_pi_p, idx::Int,
    ) where {C}
    @inbounds σ = od.sigma[idx]
    T = float(promote_type(typeof(σ), typeof(p)))
    σ <= zero(σ) && return zero(T)
    h = _scale_height_comp(comps[1], od, p, idx)
    h <= zero(h) && return zero(T)
    return T((_component_density(comps[1], Rp, inv_Rp, p, inv_p) * σ) * sqrt(two_pi_p * h))
end

@inline function _sum_graz_tau(
        comps::Tuple{C, Vararg}, od::OpticalDepth, Rp, inv_Rp, p, inv_p, two_pi_p, idx::Int,
    ) where {C}
    return _sum_graz_tau((comps[1],), od, Rp, inv_Rp, p, inv_p, two_pi_p, idx) +
        _sum_graz_tau(Base.tail(comps), od, Rp, inv_Rp, p, inv_p, two_pi_p, idx + 1)
end

@inline @muladd function _vertical_optical_depth(
        exo::ExosphereModel{FT, <:Tuple}, od::OpticalDepth, r::Real, only::Int,
    ) where {FT}
    T = float(promote_type(FT, typeof(r)))
    Rp = exo.Rp
    r < Rp && return zero(T)
    inv_Rp = exo.inv_Rp
    rf = T(r)
    inv_r = rf > zero(T) ? inv(rf) : zero(T)
    if only > 0
        @inbounds σ = od.sigma[only]
        σ <= zero(σ) && return zero(T)
        h = _scale_height_at(exo.comps, od, rf, only)
        h <= zero(h) && return zero(T)
        n = _at_component(c -> _component_density(c, Rp, inv_Rp, rf, inv_r), exo.comps, only)
        return T(n * σ * h)
    end
    return _sum_vert_tau(exo.comps, od, Rp, inv_Rp, rf, inv_r, 1)
end

@inline @muladd function _vertical_optical_depth(
        exo::ExosphereModel{FT, <:AbstractVector}, od::OpticalDepth, r::Real, only::Int,
    ) where {FT}
    T = float(promote_type(FT, typeof(r)))
    Rp = exo.Rp
    r < Rp && return zero(T)
    inv_Rp = exo.inv_Rp
    rf = T(r)
    inv_r = rf > zero(T) ? inv(rf) : zero(T)
    comps = exo.comps
    if only > 0
        @inbounds begin
            σ = od.sigma[only]
            σ <= zero(σ) && return zero(T)
            h = _scale_height_at(comps, od, rf, only)
            h <= zero(h) && return zero(T)
            return T(_component_density(comps[only], Rp, inv_Rp, rf, inv_r) * σ * h)
        end
    end
    τ = zero(T)
    @inbounds for c in eachindex(comps)
        σ = od.sigma[c]
        σ <= zero(σ) && continue
        h = _scale_height_at(comps, od, rf, c)
        h <= zero(h) && continue
        τ += (_component_density(comps[c], Rp, inv_Rp, rf, inv_r) * σ) * h
    end
    return τ
end

@inline @muladd function _grazing_optical_depth(
        exo::ExosphereModel{FT, <:Tuple}, od::OpticalDepth, p::Real,
    ) where {FT}
    T = float(promote_type(FT, typeof(p)))
    Rp = exo.Rp
    p < Rp && return zero(T)
    pf = T(p)
    inv_p = pf > zero(T) ? inv(pf) : zero(T)
    two_pi_p = (T(2) * T(π)) * pf
    return _sum_graz_tau(exo.comps, od, Rp, exo.inv_Rp, pf, inv_p, two_pi_p, 1)
end

@inline @muladd function _grazing_optical_depth(
        exo::ExosphereModel{FT, <:AbstractVector}, od::OpticalDepth, p::Real,
    ) where {FT}
    T = float(promote_type(FT, typeof(p)))
    Rp = exo.Rp
    p < Rp && return zero(T)
    inv_Rp = exo.inv_Rp
    pf = T(p)
    inv_p = pf > zero(T) ? inv(pf) : zero(T)
    two_pi_p = (T(2) * T(π)) * pf
    comps = exo.comps
    τ = zero(T)
    @inbounds for c in eachindex(comps)
        σ = od.sigma[c]
        σ <= zero(σ) && continue
        h = _scale_height_at(comps, od, pf, c)
        h <= zero(h) && continue
        τ += (_component_density(comps[c], Rp, inv_Rp, pf, inv_p) * σ) * sqrt(two_pi_p * h)
    end
    return τ
end

"""
    chapman_function(Xp::Real, cos_sza::Real)

Smith & Smith (1972) Chapman grazing-incidence function for a curved atmosphere with
``X_p = r / H``. Supports `Float32`, `Float64`, and `ForwardDiff.Dual` inputs.
"""
@inline @muladd function chapman_function(Xp::Real, cosSZA::Real)
    T = float(promote_type(typeof(Xp), typeof(cosSZA)))
    xp = T(Xp)
    c = T(cosSZA)
    sqrt_pi = sqrt(T(π))
    s = sqrt(T(0.5) * xp)
    y = s * abs(c)
    if c > zero(T)
        y < T(8) && return (sqrt_pi * s) *
            (T(1.0606963) + T(0.5564383) * y) / (T(1.0619896) + y * (T(1.7245609) + y))
        y < T(100) && return (sqrt_pi * s) * T(0.56498823) / (T(0.6651874) + y)
        return zero(T)
    end
    one_m_c2 = one(T) - c * c
    sinSZA = one_m_c2 > zero(T) ? sqrt(one_m_c2) : zero(T)
    two_sqrt_pi_s = (T(2) * sqrt_pi) * s
    if y < T(8)
        return two_sqrt_pi_s * (
            sqrt(sinSZA) * exp(xp * (one(T) - sinSZA)) -
                T(0.5) * (T(1.0606963) + T(0.5564383) * y) /
                (T(1.0619896) + y * (T(1.7245609) + y))
        )
    end
    y < T(100) && return two_sqrt_pi_s * (
        sqrt(sinSZA) * exp(min(T(100), xp * (one(T) - sinSZA))) -
            T(0.5) * T(0.56498823) / (T(0.6651874) + y)
    )
    return zero(T)
end

@inline function _photo_attenuation_dispatch(
        ::ExosphereModel{FT}, ::NoAttenuation, x::SVector{3}, r::Real, t,
    ) where {FT}
    T = float(promote_type(FT, eltype(x), typeof(r)))
    return one(T)
end

@inline function _photo_attenuation_dispatch(
        ::ExosphereModel{FT}, sc::ShadowCylinder, x::SVector{3}, r::Real, t,
    ) where {FT}
    T = float(promote_type(FT, eltype(x), typeof(r)))
    return in_shadow(sc, x) ? zero(T) : one(T)
end

@inline function _photo_attenuation_dispatch(
        exo::ExosphereModel, rod::RayOpticalDepth, x::SVector{3}, ::Real, t,
    )
    return _ray_attenuation(exo, rod, x, t)
end

@inline @muladd function _photo_attenuation_dispatch(
        exo::ExosphereModel{FT}, od::OpticalDepth, x::SVector{3}, r::Real, t,
    ) where {FT}
    T = float(promote_type(FT, eltype(x), typeof(r)))
    rf = T(r)
    rs = max(rf, T(1.0e-3) * max(T(exo.Rp), T(1.0e-30)))
    proj = x[1] * od.dir[1] + x[2] * od.dir[2] + x[3] * od.dir[3]
    cosSZA = proj / rs
    mu = max(cosSZA, T(od.cos_sza_floor))
    if od.chapman
        c = od.chapman_component > 0 ? od.chapman_component : 1
        τv = _vertical_optical_depth(exo, od, rf, c)
        τv > od.chapman_tau_max &&
            return T(od.min_production) * max(mu, T(od.min_production))
        h = _scale_height_at(exo.comps, od, rf, c)
        h <= zero(h) && return T(od.min_production)
        ch = chapman_function(rf / h, cosSZA)
        ch <= zero(ch) && return T(od.min_production) * max(mu, T(od.min_production))
        return max(exp(-τv * ch), T(od.min_production))
    end
    τ = if proj <= zero(proj)
        p2 = max(rf * rf - proj * proj, zero(rf))
        p = iszero(p2) ? zero(rf) : sqrt(p2)
        p <= exo.Rp && return T(od.min_production)
        max(_grazing_optical_depth(exo, od, p), T(od.tau_floor))
    else
        max(_vertical_optical_depth(exo, od, rf, 0), T(od.tau_floor)) / mu
    end
    τ >= od.tau_cutoff && return T(od.min_production)
    return max(exp(-τ), T(od.min_production))
end

"""
    photo_attenuation(exo::ExosphereModel, x::SVector{3}, r::Real; t = nothing)
    photo_attenuation(model::AbstractNeutralModel, x::AbstractVector; t = nothing)

EUV ionizing flux attenuation factor in `[0, 1]` at position `x` relative to the planet
center (`r = norm(x)`). The 3-argument form avoids recomputing `norm(x)` in hot loops.
"""
@inline function photo_attenuation(
        exo::ExosphereModel, x::SVector{3, <:Real}, r::Real; t = nothing,
    )
    return _photo_attenuation_dispatch(exo, exo.attenuation, x, r, t)
end

@inline function photo_attenuation(
        exo::ExosphereModel, x::AbstractVector{<:Real}; t = nothing,
    )
    xv = _as_svec3(x)
    return _photo_attenuation_dispatch(exo, exo.attenuation, xv, _safe_norm(xv), t)
end

@inline function photo_attenuation(
        ::AbstractNeutralModel, x::AbstractVector{<:Real}; t = nothing,
    )
    return one(float(eltype(x)))
end
