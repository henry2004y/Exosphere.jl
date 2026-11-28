# ==============================================================================
# Structured Neutral Grids, Two-Way Feedback Buffers & GriddedNeutralField
# ==============================================================================

# ------------------------------------------------------------------------------
# Structured Grid Geometries (isbits / GPU-Friendly)
# ------------------------------------------------------------------------------

"""
    CartesianNeutralGrid{D, FT<:AbstractFloat} <: AbstractNeutralGrid
    CartesianNeutralGrid(origin, dx, dims; FT = Float64)

Uniform rectilinear `D`-dimensional (`D ∈ {1, 2, 3}`) neutral grid parameterized by the
center of cell `(1, ..., 1)` (`origin`), cell spacing `dx`, and grid size `dims`.
"""
struct CartesianNeutralGrid{D, FT <: AbstractFloat} <: AbstractNeutralGrid
    origin::SVector{D, FT}
    dx::SVector{D, FT}
    inv_dx::SVector{D, FT}
    dims::NTuple{D, Int}
end

function CartesianNeutralGrid(
        origin::Union{NTuple{D, Real}, SVector{D, <:Real}},
        dx::Union{NTuple{D, Real}, SVector{D, <:Real}},
        dims::NTuple{D, Integer};
        FT::Type{<:AbstractFloat} = Float64,
    ) where {D}
    (1 <= D <= 3) || throw(ArgumentError("CartesianNeutralGrid supports D ∈ {1, 2, 3}"))
    orig = SVector{D, FT}(ntuple(d -> FT(origin[d]), Val(D)))
    sp = SVector{D, FT}(ntuple(d -> FT(dx[d]), Val(D)))
    all(>(zero(FT)), sp) || throw(ArgumentError("CartesianNeutralGrid requires dx > 0"))
    dm = ntuple(d -> Int(dims[d]), Val(D))
    all(>=(1), dm) || throw(ArgumentError("CartesianNeutralGrid requires dims >= 1"))
    return CartesianNeutralGrid{D, FT}(orig, sp, inv.(sp), dm)
end

"""
    cell_center(grid::AbstractNeutralGrid, idx::CartesianIndex) -> SVector{3}

Return the 3D spatial coordinates of the center of cell `idx` in `grid`.
"""
@inline function cell_center(g::CartesianNeutralGrid{1, FT}, idx::CartesianIndex{1}) where {FT}
    x1 = g.origin[1] + FT(idx[1] - 1) * g.dx[1]
    return SVector{3, FT}(x1, zero(FT), zero(FT))
end

@inline function cell_center(g::CartesianNeutralGrid{2, FT}, idx::CartesianIndex{2}) where {FT}
    x1 = g.origin[1] + FT(idx[1] - 1) * g.dx[1]
    x2 = g.origin[2] + FT(idx[2] - 1) * g.dx[2]
    return SVector{3, FT}(x1, x2, zero(FT))
end

@inline function cell_center(g::CartesianNeutralGrid{3, FT}, idx::CartesianIndex{3}) where {FT}
    x1 = g.origin[1] + FT(idx[1] - 1) * g.dx[1]
    x2 = g.origin[2] + FT(idx[2] - 1) * g.dx[2]
    x3 = g.origin[3] + FT(idx[3] - 1) * g.dx[3]
    return SVector{3, FT}(x1, x2, x3)
end

"""
    cell_volume(grid::AbstractNeutralGrid[, idx])

Return the cell volume (or 1D length / 2D area for lower-dimensional Cartesian grids) at
`idx`.
"""
@inline cell_volume(g::CartesianNeutralGrid{1, FT}) where {FT} = g.dx[1]
@inline cell_volume(g::CartesianNeutralGrid{2, FT}) where {FT} = g.dx[1] * g.dx[2]
@inline cell_volume(g::CartesianNeutralGrid{3, FT}) where {FT} = g.dx[1] * g.dx[2] * g.dx[3]
@inline cell_volume(g::CartesianNeutralGrid{D, FT}, ::CartesianIndex{D}) where {D, FT} =
    cell_volume(g)
@inline cell_volume(g::CartesianNeutralGrid, ::Nothing) = cell_volume(g)

@inline _grid_dims(g::CartesianNeutralGrid) = g.dims

"""
    SphericalNeutralGrid{FT<:AbstractFloat, VR<:AbstractVector{FT}} <: AbstractNeutralGrid
    SphericalNeutralGrid(r; ntheta = 1, nphi = 1, center = (0.0, 0.0, 0.0),
                         polar_axis = (1.0, 0.0, 0.0))

Structured spherical planetary neutral grid with sorted (uniform or stretched) radial
centers `r`, polar count `ntheta`, and azimuthal count `nphi` about `polar_axis` (defaults
to `+x` so polar angle ``\\theta`` equals solar zenith angle in 2D axisymmetric models):
- `ntheta == 1 && nphi == 1`: 1D radial grid `(nr,)`
- `ntheta > 1 && nphi == 1`: 2D axisymmetric `(nr, ntheta)` grid
- `ntheta > 1 && nphi > 1`: 3D spherical `(nr, ntheta, nphi)` grid
"""
struct SphericalNeutralGrid{FT <: AbstractFloat, VR <: AbstractVector{FT}} <:
    AbstractNeutralGrid
    r::VR
    ntheta::Int
    nphi::Int
    center::SVector{3, FT}
    e_polar::SVector{3, FT}
    e_phi0::SVector{3, FT}
    e_phi90::SVector{3, FT}
end

function SphericalNeutralGrid(
        r::AbstractVector{<:Real};
        ntheta::Integer = 1,
        nphi::Integer = 1,
        center::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            0.0, 0.0, 0.0,
        ),
        polar_axis::Union{NTuple{3, Real}, SVector{3, <:Real}, AbstractVector{<:Real}} = (
            1.0, 0.0, 0.0,
        ),
    )
    length(r) >= 2 || throw(ArgumentError("SphericalNeutralGrid requires at least 2 radial nodes"))
    for i in (firstindex(r) + 1):lastindex(r)
        r[i] > r[i - 1] ||
            throw(ArgumentError("SphericalNeutralGrid: r must be strictly increasing"))
    end
    ntheta >= 1 || throw(ArgumentError("SphericalNeutralGrid requires ntheta >= 1"))
    nphi >= 1 || throw(ArgumentError("SphericalNeutralGrid requires nphi >= 1"))
    (nphi > 1 && ntheta == 1) &&
        throw(ArgumentError("SphericalNeutralGrid: nphi > 1 requires ntheta > 1"))
    FT = float(eltype(r))
    rv = r isa SVector ? SVector{length(r), FT}(r) : FT[FT(x) for x in r]
    ctr = SVector{3, FT}(FT(center[1]), FT(center[2]), FT(center[3]))
    pa = SVector{3, FT}(FT(polar_axis[1]), FT(polar_axis[2]), FT(polar_axis[3]))
    pn = norm(pa)
    pn > zero(FT) || throw(ArgumentError("SphericalNeutralGrid: polar_axis must be non-zero"))
    e_p = pa / pn
    e1, e2 = _orthonormal_perp_pair(e_p)
    return SphericalNeutralGrid{FT, typeof(rv)}(rv, Int(ntheta), Int(nphi), ctr, e_p, e1, e2)
end

@inline function _grid_dims(g::SphericalNeutralGrid)
    nr = length(g.r)
    g.ntheta == 1 && return (nr,)
    g.nphi == 1 && return (nr, g.ntheta)
    return (nr, g.ntheta, g.nphi)
end

@inline function cell_center(g::SphericalNeutralGrid{FT}, idx::CartesianIndex{1}) where {FT}
    @inbounds ri = g.r[idx[1]]
    return g.center + ri * g.e_polar
end

@inline function cell_center(g::SphericalNeutralGrid{FT}, idx::CartesianIndex{2}) where {FT}
    @inbounds ri = g.r[idx[1]]
    dth = FT(π) / FT(g.ntheta)
    th = (FT(idx[2]) - FT(0.5)) * dth
    s, c = sincos(th)
    return g.center + ri * (c * g.e_polar + s * g.e_phi0)
end

@inline function cell_center(g::SphericalNeutralGrid{FT}, idx::CartesianIndex{3}) where {FT}
    @inbounds ri = g.r[idx[1]]
    dth = FT(π) / FT(g.ntheta)
    dph = (FT(2) * FT(π)) / FT(g.nphi)
    th = (FT(idx[2]) - FT(0.5)) * dth
    ph = (FT(idx[3]) - FT(0.5)) * dph
    st, ct = sincos(th)
    sp, cp = sincos(ph)
    return g.center + ri * (ct * g.e_polar + (st * cp) * g.e_phi0 + (st * sp) * g.e_phi90)
end

# ------------------------------------------------------------------------------
# Two-Way Feedback Buffers (Eulerian Moments & Kinetic Particles)
# ------------------------------------------------------------------------------

"""
    NeutralSourceBuffer{AN, AM, AE}
    NeutralSourceBuffer(FT, dims, nspecies)

Mutable Eulerian source-rate accumulator storing `(dn, dmom, dE)` over array shape
`(dims..., nspecies)` for two-way charge–neutral coupling.
"""
struct NeutralSourceBuffer{
        AN <: AbstractArray, AM <: AbstractArray, AE <: AbstractArray,
    }
    dn::AN
    dmom::AM
    dE::AE
end

function NeutralSourceBuffer(
        ::Type{FT}, dims::Tuple{Vararg{Int}}, nspecies::Integer,
    ) where {FT <: AbstractFloat}
    ns = Int(nspecies)
    shape = (dims..., ns)
    dn = zeros(FT, shape)
    dmom = fill(SVector{3, FT}(zero(FT), zero(FT), zero(FT)), shape)
    dE = zeros(FT, shape)
    return NeutralSourceBuffer{typeof(dn), typeof(dmom), typeof(dE)}(dn, dmom, dE)
end

"""
    reset_sources!(buf::NeutralSourceBuffer)

Zero all accumulated Eulerian source arrays in place.
"""
function reset_sources!(buf::NeutralSourceBuffer)
    fill!(buf.dn, zero(eltype(buf.dn)))
    fill!(buf.dmom, zero(eltype(buf.dmom)))
    fill!(buf.dE, zero(eltype(buf.dE)))
    return buf
end

"""
    merge_sources!(dst::NeutralSourceBuffer, src::NeutralSourceBuffer)

Accumulate `src` into `dst` in place (for multi-threaded per-thread source reduction).
"""
function merge_sources!(dst::NeutralSourceBuffer, src::NeutralSourceBuffer)
    dst.dn .+= src.dn
    dst.dmom .+= src.dmom
    dst.dE .+= src.dE
    return dst
end

"""
    NeutralParticleBuffer{FT<:AbstractFloat, VP<:AbstractVector{NeutralParticle{FT}}}
    NeutralParticleBuffer(FT = Float64, capacity = 1024; resizable = true)

Pre-allocated buffer for discrete neutral macroparticles (ENAs, sputtered neutrals)
deposited during a kinetic/Hybrid-PIC coupling step.
"""
struct NeutralParticleBuffer{FT <: AbstractFloat, VP <: AbstractVector{NeutralParticle{FT}}}
    particles::VP
    count::Base.RefValue{Int}
    resizable::Bool
end

function NeutralParticleBuffer(
        ::Type{FT} = Float64, capacity::Integer = 1024; resizable::Bool = true,
    ) where {FT <: AbstractFloat}
    cap = max(Int(capacity), 0)
    z3 = SVector{3, FT}(zero(FT), zero(FT), zero(FT))
    buf = fill(NeutralParticle{FT}(z3, z3, zero(FT), zero(FT), Int32(0)), cap)
    return NeutralParticleBuffer{FT, typeof(buf)}(buf, Ref(0), resizable)
end

"""
    push_particle!(buf::NeutralParticleBuffer, p::NeutralParticle) -> Int

Append `p` to `buf` without allocating while below capacity (growing automatically when
`buf.resizable == true`) and return the updated active particle count.
"""
function push_particle!(buf::NeutralParticleBuffer{FT}, p::NeutralParticle) where {FT}
    p_ft = p isa NeutralParticle{FT} ? p :
        NeutralParticle{FT}(
            SVector{3, FT}(FT(p.pos[1]), FT(p.pos[2]), FT(p.pos[3])),
            SVector{3, FT}(FT(p.vel[1]), FT(p.vel[2]), FT(p.vel[3])),
            FT(p.mass), FT(p.weight), p.species,
        )
    idx = buf.count[] + 1
    if idx > length(buf.particles)
        if buf.resizable
            resize!(buf.particles, max(2 * length(buf.particles), 64))
        else
            throw(BoundsError(buf.particles, idx))
        end
    end
    @inbounds buf.particles[idx] = p_ft
    buf.count[] = idx
    return idx
end

function reset_sources!(buf::NeutralParticleBuffer)
    buf.count[] = 0
    return buf
end

"""
    active_particles(buf::NeutralParticleBuffer)

Return a zero-allocation `@view` of all neutral macroparticles deposited since the last
[`reset_sources!`](@ref).
"""
@inline active_particles(buf::NeutralParticleBuffer) = @view buf.particles[1:buf.count[]]

# ------------------------------------------------------------------------------
# GriddedNeutralField (Tabulated One-Way OR Mutable Two-Way Neutral Model)
# ------------------------------------------------------------------------------

"""
    GriddedNeutralField{G, AN, AU, AT, M, Src, PB, Atten, BG, FT} <: AbstractNeutralModel
    GriddedNeutralField(grid::AbstractNeutralGrid, nspecies::Integer;
                        masses = ones(FT, nspecies), two_way = false,
                        particle_capacity = 0, attenuation = NoAttenuation(),
                        background = nothing, planet_radius = 0.0, FT = Float64)

Structured-grid neutral atmosphere container supporting both read-only tabulated 3D
backgrounds (`two_way = false` -> [`OneWayCoupling`](@ref)) and live two-way coupled
simulations (`two_way = true` or `particle_capacity > 0` -> [`TwoWayCoupling`](@ref)).

Supports:
- Direct `@inbounds` cell indexing via `CartesianIndex` (matched plasma/neutral grids)
- Zero-allocation linear/bilinear/trilinear interpolation at continuous 3D positions `x`
- Conservative adjoint CIC deposition onto `sources` at continuous 3D positions `x`
- Optional far-field `background::AbstractNeutralModel` outside the neutral grid domain
- GPU array migration via `ExosphereAdaptExt` (`Adapt.adapt(CuArray, field)`).
"""
struct GriddedNeutralField{
        G <: AbstractNeutralGrid,
        AN <: AbstractArray,
        AU <: AbstractArray,
        AT <: AbstractArray,
        M <: AbstractVector,
        Src <: Union{Nothing, NeutralSourceBuffer},
        PB <: Union{Nothing, NeutralParticleBuffer},
        Atten <: AbstractAttenuation,
        BG <: Union{Nothing, AbstractNeutralModel},
        FT <: AbstractFloat,
    } <: AbstractNeutralModel
    grid::G
    n::AN
    u::AU
    T::AT
    masses::M
    sources::Src
    particles::PB
    attenuation::Atten
    background::BG
    Rp::FT
end

function GriddedNeutralField(
        grid::AbstractNeutralGrid,
        nspecies::Integer;
        masses::Union{Nothing, AbstractVector{<:Real}, Tuple{Vararg{Real}}} = nothing,
        two_way::Bool = false,
        particle_capacity::Integer = 0,
        attenuation::AbstractAttenuation = NoAttenuation(),
        background::Union{Nothing, AbstractNeutralModel} = nothing,
        planet_radius::Real = 0.0,
        FT::Type{<:AbstractFloat} = Float64,
    )
    ns = Int(nspecies)
    ns >= 1 || throw(ArgumentError("GriddedNeutralField requires nspecies >= 1"))
    dims = _grid_dims(grid)
    shape = (dims..., ns)
    n_arr = zeros(FT, shape)
    u_arr = fill(SVector{3, FT}(zero(FT), zero(FT), zero(FT)), shape)
    T_arr = zeros(FT, shape)
    m_vec = if masses === nothing
        ones(FT, ns)
    elseif masses isa Tuple
        length(masses) == ns ||
            throw(ArgumentError("GriddedNeutralField: masses length must equal nspecies"))
        SVector{length(masses), FT}(ntuple(i -> FT(masses[i]), Val(length(masses))))
    elseif masses isa SVector
        length(masses) == ns ||
            throw(ArgumentError("GriddedNeutralField: masses length must equal nspecies"))
        SVector{length(masses), FT}(masses)
    else
        length(masses) == ns ||
            throw(ArgumentError("GriddedNeutralField: masses length must equal nspecies"))
        FT[FT(m) for m in masses]
    end
    any(<=(zero(FT)), m_vec) &&
        throw(ArgumentError("GriddedNeutralField requires positive species masses"))
    src = two_way ? NeutralSourceBuffer(FT, dims, ns) : nothing
    pbuf = particle_capacity > 0 ? NeutralParticleBuffer(FT, particle_capacity) : nothing
    return GriddedNeutralField{
        typeof(grid), typeof(n_arr), typeof(u_arr), typeof(T_arr), typeof(m_vec),
        typeof(src), typeof(pbuf), typeof(attenuation), typeof(background), FT,
    }(
        grid, n_arr, u_arr, T_arr, m_vec, src, pbuf, attenuation, background, FT(planet_radius),
    )
end

@inline _coupling_mode_from(::Nothing, ::Nothing) = OneWayCoupling()
@inline _coupling_mode_from(
    ::Union{Nothing, NeutralSourceBuffer}, ::Union{Nothing, NeutralParticleBuffer},
) = TwoWayCoupling()

@inline coupling_mode(f::GriddedNeutralField) = _coupling_mode_from(f.sources, f.particles)

@inline n_components(f::GriddedNeutralField) = size(f.n, ndims(f.n))
@inline species_mass(f::GriddedNeutralField, c::Integer = 1) = @inbounds f.masses[Int(c)]
@inline neutral_sources(f::GriddedNeutralField) = f.sources
@inline neutral_particles(f::GriddedNeutralField) = f.particles

function _reset_sources!(::TwoWayCoupling, f::GriddedNeutralField)
    f.sources !== nothing && reset_sources!(f.sources)
    f.particles !== nothing && reset_sources!(f.particles)
    return f
end

# ------------------------------------------------------------------------------
# Grid Coordinate Mapping, Interpolation & Adjoint CIC Deposition
# ------------------------------------------------------------------------------

@inline function _axis_interp(coord::Real, orig::FT, inv_dx::FT, dim::Int) where {FT}
    T = float(promote_type(FT, typeof(coord)))
    s = (T(coord) - orig) * inv_dx + one(T)
    in_bounds = (s >= T(0.5)) && (s <= T(dim) + T(0.5))
    if dim == 1
        return (in_bounds, 1, 1, zero(T))
    end
    sc = clamp(s, one(T), T(dim))
    i0 = min(max(floor(Int, sc), 1), dim - 1)
    w = sc - T(i0)
    return (in_bounds, i0, i0 + 1, w)
end

# Direct cell-index queries (zero interpolation overhead on matched grids)
@inline function neutral_state(
        f::GriddedNeutralField, idx::CartesianIndex;
        t = nothing, species::Integer = 1, kB::Number = 1,
    )
    s = Int(species)
    @inbounds return NeutralState(f.n[idx, s], f.u[idx, s], f.T[idx, s])
end

@inline function neutral_density(
        f::GriddedNeutralField, idx::CartesianIndex;
        t = nothing, species::Union{Nothing, Integer} = nothing,
    )
    if species !== nothing
        @inbounds return f.n[idx, Int(species)]
    end
    tot = zero(eltype(f.n))
    @inbounds for s in 1:n_components(f)
        tot += f.n[idx, s]
    end
    return tot
end

@inline function _fallback_state_bg(
        ::Nothing, ::Type{FT}, x::SVector{3}, t, species::Int, kB::Number,
    ) where {FT}
    T = float(promote_type(FT, eltype(x)))
    return NeutralState(zero(T), SVector{3, T}(zero(T), zero(T), zero(T)), zero(T))
end

@inline function _fallback_state_bg(
        bg::AbstractNeutralModel, ::Type{FT}, x::SVector{3}, t, species::Int, kB::Number,
    ) where {FT}
    return neutral_state(bg, x; t = t, species = species, kB = kB)
end

@inline function _fallback_state(
        f::GriddedNeutralField, x::SVector{3}, t, species::Int, kB::Number,
    )
    return _fallback_state_bg(f.background, eltype(f.n), x, t, species, kB)
end

# 1D, 2D, 3D Cartesian interpolation
@inline function _interp_state(
        f::GriddedNeutralField{CartesianNeutralGrid{1, FT}}, x::SVector{3}, t, s::Int, kB::Number,
    ) where {FT}
    g = f.grid
    ok1, i0, i1, w1 = _axis_interp(x[1], g.origin[1], g.inv_dx[1], g.dims[1])
    ok1 || return _fallback_state(f, x, t, s, kB)
    a0 = one(w1) - w1
    @inbounds begin
        n = a0 * f.n[i0, s] + w1 * f.n[i1, s]
        u = a0 * f.u[i0, s] + w1 * f.u[i1, s]
        Tv = a0 * f.T[i0, s] + w1 * f.T[i1, s]
    end
    return NeutralState(n, u, Tv)
end

@inline function _interp_state(
        f::GriddedNeutralField{CartesianNeutralGrid{2, FT}}, x::SVector{3}, t, s::Int, kB::Number,
    ) where {FT}
    g = f.grid
    ok1, i0, i1, w1 = _axis_interp(x[1], g.origin[1], g.inv_dx[1], g.dims[1])
    ok2, j0, j1, w2 = _axis_interp(x[2], g.origin[2], g.inv_dx[2], g.dims[2])
    (ok1 && ok2) || return _fallback_state(f, x, t, s, kB)
    a1 = one(w1) - w1
    a2 = one(w2) - w2
    w00 = a1 * a2
    w10 = w1 * a2
    w01 = a1 * w2
    w11 = w1 * w2
    @inbounds begin
        n = w00 * f.n[i0, j0, s] + w10 * f.n[i1, j0, s] +
            w01 * f.n[i0, j1, s] + w11 * f.n[i1, j1, s]
        u = w00 * f.u[i0, j0, s] + w10 * f.u[i1, j0, s] +
            w01 * f.u[i0, j1, s] + w11 * f.u[i1, j1, s]
        Tv = w00 * f.T[i0, j0, s] + w10 * f.T[i1, j0, s] +
            w01 * f.T[i0, j1, s] + w11 * f.T[i1, j1, s]
    end
    return NeutralState(n, u, Tv)
end

@inline function _interp_state(
        f::GriddedNeutralField{CartesianNeutralGrid{3, FT}}, x::SVector{3}, t, s::Int, kB::Number,
    ) where {FT}
    g = f.grid
    ok1, i0, i1, w1 = _axis_interp(x[1], g.origin[1], g.inv_dx[1], g.dims[1])
    ok2, j0, j1, w2 = _axis_interp(x[2], g.origin[2], g.inv_dx[2], g.dims[2])
    ok3, k0, k1, w3 = _axis_interp(x[3], g.origin[3], g.inv_dx[3], g.dims[3])
    (ok1 && ok2 && ok3) || return _fallback_state(f, x, t, s, kB)
    a1 = one(w1) - w1
    a2 = one(w2) - w2
    a3 = one(w3) - w3
    w000 = a1 * a2 * a3
    w100 = w1 * a2 * a3
    w010 = a1 * w2 * a3
    w110 = w1 * w2 * a3
    w001 = a1 * a2 * w3
    w101 = w1 * a2 * w3
    w011 = a1 * w2 * w3
    w111 = w1 * w2 * w3
    @inbounds begin
        n = w000 * f.n[i0, j0, k0, s] + w100 * f.n[i1, j0, k0, s] +
            w010 * f.n[i0, j1, k0, s] + w110 * f.n[i1, j1, k0, s] +
            w001 * f.n[i0, j0, k1, s] + w101 * f.n[i1, j0, k1, s] +
            w011 * f.n[i0, j1, k1, s] + w111 * f.n[i1, j1, k1, s]
        u = w000 * f.u[i0, j0, k0, s] + w100 * f.u[i1, j0, k0, s] +
            w010 * f.u[i0, j1, k0, s] + w110 * f.u[i1, j1, k0, s] +
            w001 * f.u[i0, j0, k1, s] + w101 * f.u[i1, j0, k1, s] +
            w011 * f.u[i0, j1, k1, s] + w111 * f.u[i1, j1, k1, s]
        Tv = w000 * f.T[i0, j0, k0, s] + w100 * f.T[i1, j0, k0, s] +
            w010 * f.T[i0, j1, k0, s] + w110 * f.T[i1, j1, k0, s] +
            w001 * f.T[i0, j0, k1, s] + w101 * f.T[i1, j0, k1, s] +
            w011 * f.T[i0, j1, k1, s] + w111 * f.T[i1, j1, k1, s]
    end
    return NeutralState(n, u, Tv)
end

# Spherical grid (1D / 2D / 3D) interpolation
@inline function _spherical_coords(g::SphericalNeutralGrid{FT}, x::SVector{3}) where {FT}
    d = _relative_pos(g.center, x)
    r = _safe_norm(d)
    T = float(promote_type(FT, typeof(r)))
    if iszero(r)
        return (r, zero(T), zero(T))
    end
    zp = (d[1] * g.e_polar[1] + d[2] * g.e_polar[2] + d[3] * g.e_polar[3])
    xp = (d[1] * g.e_phi0[1] + d[2] * g.e_phi0[2] + d[3] * g.e_phi0[3])
    yp = (d[1] * g.e_phi90[1] + d[2] * g.e_phi90[2] + d[3] * g.e_phi90[3])
    rhop2 = xp * xp + yp * yp
    rhop = iszero(rhop2) ? zero(T) : sqrt(rhop2)
    th = atan(rhop, zp)
    ph = atan(yp, xp)
    ph < zero(T) && (ph += T(2) * T(π))
    return (r, th, ph)
end

@inline function _interp_state(
        f::GriddedNeutralField{<:SphericalNeutralGrid{FT}}, x::SVector{3}, t, s::Int, kB::Number,
    ) where {FT}
    g = f.grid
    r, th, ph = _spherical_coords(g, x)
    @inbounds begin
        (r < g.r[1] || r > g.r[end]) && return _fallback_state(f, x, t, s, kB)
        ir = _find_bracket(g.r, r)
        wr = (r - g.r[ir]) / (g.r[ir + 1] - g.r[ir])
        ar = one(wr) - wr
        if g.ntheta == 1
            n = ar * f.n[ir, s] + wr * f.n[ir + 1, s]
            u = ar * f.u[ir, s] + wr * f.u[ir + 1, s]
            Tv = ar * f.T[ir, s] + wr * f.T[ir + 1, s]
            return NeutralState(n, u, Tv)
        end
        dth = FT(π) / FT(g.ntheta)
        sth = th / dth + FT(0.5)
        sthc = clamp(sth, one(sth), oftype(sth, g.ntheta))
        it0 = min(max(floor(Int, sthc), 1), g.ntheta - 1)
        it1 = it0 + 1
        wt = sthc - oftype(sthc, it0)
        at = one(wt) - wt
        if g.nphi == 1
            w00 = ar * at
            w10 = wr * at
            w01 = ar * wt
            w11 = wr * wt
            n = w00 * f.n[ir, it0, s] + w10 * f.n[ir + 1, it0, s] +
                w01 * f.n[ir, it1, s] + w11 * f.n[ir + 1, it1, s]
            u = w00 * f.u[ir, it0, s] + w10 * f.u[ir + 1, it0, s] +
                w01 * f.u[ir, it1, s] + w11 * f.u[ir + 1, it1, s]
            Tv = w00 * f.T[ir, it0, s] + w10 * f.T[ir + 1, it0, s] +
                w01 * f.T[ir, it1, s] + w11 * f.T[ir + 1, it1, s]
            return NeutralState(n, u, Tv)
        end
        dph = (FT(2) * FT(π)) / FT(g.nphi)
        sph = ph / dph + FT(0.5)
        ip0_raw = floor(Int, sph)
        wp = sph - oftype(sph, ip0_raw)
        ap = one(wp) - wp
        ip0 = mod1(ip0_raw, g.nphi)
        ip1 = mod1(ip0_raw + 1, g.nphi)
        w000 = ar * at * ap
        w100 = wr * at * ap
        w010 = ar * wt * ap
        w110 = wr * wt * ap
        w001 = ar * at * wp
        w101 = wr * at * wp
        w011 = ar * wt * wp
        w111 = wr * wt * wp
        n = w000 * f.n[ir, it0, ip0, s] + w100 * f.n[ir + 1, it0, ip0, s] +
            w010 * f.n[ir, it1, ip0, s] + w110 * f.n[ir + 1, it1, ip0, s] +
            w001 * f.n[ir, it0, ip1, s] + w101 * f.n[ir + 1, it0, ip1, s] +
            w011 * f.n[ir, it1, ip1, s] + w111 * f.n[ir + 1, it1, ip1, s]
        u = w000 * f.u[ir, it0, ip0, s] + w100 * f.u[ir + 1, it0, ip0, s] +
            w010 * f.u[ir, it1, ip0, s] + w110 * f.u[ir + 1, it1, ip0, s] +
            w001 * f.u[ir, it0, ip1, s] + w101 * f.u[ir + 1, it0, ip1, s] +
            w011 * f.u[ir, it1, ip1, s] + w111 * f.u[ir + 1, it1, ip1, s]
        Tv = w000 * f.T[ir, it0, ip0, s] + w100 * f.T[ir + 1, it0, ip0, s] +
            w010 * f.T[ir, it1, ip0, s] + w110 * f.T[ir + 1, it1, ip0, s] +
            w001 * f.T[ir, it0, ip1, s] + w101 * f.T[ir + 1, it0, ip1, s] +
            w011 * f.T[ir, it1, ip1, s] + w111 * f.T[ir + 1, it1, ip1, s]
        return NeutralState(n, u, Tv)
    end
end

@inline function neutral_state(
        f::GriddedNeutralField, x::AbstractVector{<:Real};
        t = nothing, species::Integer = 1, kB::Number = 1,
    )
    xv = _as_svec3(x)
    st = _interp_state(f, xv, t, Int(species), kB)
    if f.Rp > zero(f.Rp) && _safe_norm(xv) < f.Rp
        return NeutralState(zero(st.n), st.u, st.T)
    end
    return st
end

@inline function neutral_density(
        f::GriddedNeutralField, x::AbstractVector{<:Real};
        t = nothing, species::Union{Nothing, Integer} = nothing,
    )
    xv = _as_svec3(x)
    if species !== nothing
        return neutral_state(f, xv; t = t, species = Int(species)).n
    end
    tot = zero(float(promote_type(eltype(f.n), eltype(xv))))
    for s in 1:n_components(f)
        tot += neutral_state(f, xv; t = t, species = s).n
    end
    return tot
end

@inline function local_vdf(
        f::GriddedNeutralField, loc::Union{CartesianIndex, AbstractVector{<:Real}};
        t = nothing, species::Integer = 1, kB::Number = 1,
    )
    ns = neutral_state(f, loc; t = t, species = species, kB = kB)
    sp = NeutralSpecies(; m = species_mass(f, species))
    return _local_vdf_from_uT(sp, ns.u, ns.T, kB)
end

@inline function local_scale_height(
        f::GriddedNeutralField{CartesianNeutralGrid{D, FT}}, x::AbstractVector{<:Real};
        t = nothing, species::Integer = 1,
    ) where {D, FT}
    if f.background !== nothing
        return local_scale_height(f.background, x; t = t, species = species)
    end
    return zero(float(promote_type(FT, eltype(x))))
end

@inline function local_scale_height(
        f::GriddedNeutralField{<:SphericalNeutralGrid{FT}}, x::AbstractVector{<:Real};
        t = nothing, species::Integer = 1,
    ) where {FT}
    if f.background !== nothing
        return local_scale_height(f.background, x; t = t, species = species)
    end
    return zero(float(promote_type(FT, eltype(x))))
end

@inline _field_attenuation(::GriddedNeutralField, ::NoAttenuation, x::SVector{3}, t) =
    one(float(eltype(x)))
@inline _field_attenuation(::GriddedNeutralField, sc::ShadowCylinder, x::SVector{3}, t) =
    shadow_factor(sc, x)
@inline _field_attenuation(f::GriddedNeutralField, rod::RayOpticalDepth, x::SVector{3}, t) =
    _ray_attenuation(f, rod, x, t)

@inline function photo_attenuation(
        f::GriddedNeutralField, x::AbstractVector{<:Real}; t = nothing,
    )
    return _field_attenuation(f, f.attenuation, _as_svec3(x), t)
end

@inline function photo_attenuation(
        f::GriddedNeutralField, idx::CartesianIndex; t = nothing,
    )
    return photo_attenuation(f, cell_center(f.grid, idx); t = t)
end

# ------------------------------------------------------------------------------
# Two-Way Source & Particle Deposition on GriddedNeutralField
# ------------------------------------------------------------------------------

@inline function _add_cell_source!(
        buf::NeutralSourceBuffer, idx, s::Int, w::Real, src::MomentSource,
    )
    FT = eltype(buf.dn)
    wf = FT(w)
    @inbounds begin
        buf.dn[idx, s] += wf * FT(src.dn)
        buf.dmom[idx, s] += wf * SVector{3, FT}(FT(src.dmom[1]), FT(src.dmom[2]), FT(src.dmom[3]))
        buf.dE[idx, s] += wf * FT(src.dE)
    end
    return nothing
end

@inline function _deposit_neutral_source!(
        ::TwoWayCoupling,
        f::GriddedNeutralField,
        idx::CartesianIndex,
        src::MomentSource,
        species::Integer,
    )
    buf = f.sources
    buf === nothing && return nothing
    _add_cell_source!(buf, idx, Int(species), 1, src)
    return nothing
end

@inline function _deposit_cic!(
        buf::NeutralSourceBuffer, g::CartesianNeutralGrid{1, FT}, x::SVector{3},
        src::MomentSource, s::Int,
    ) where {FT}
    ok1, i0, i1, w1 = _axis_interp(x[1], g.origin[1], g.inv_dx[1], g.dims[1])
    ok1 || return nothing
    _add_cell_source!(buf, CartesianIndex(i0), s, one(w1) - w1, src)
    _add_cell_source!(buf, CartesianIndex(i1), s, w1, src)
    return nothing
end

@inline function _deposit_cic!(
        buf::NeutralSourceBuffer, g::CartesianNeutralGrid{2, FT}, x::SVector{3},
        src::MomentSource, s::Int,
    ) where {FT}
    ok1, i0, i1, w1 = _axis_interp(x[1], g.origin[1], g.inv_dx[1], g.dims[1])
    ok2, j0, j1, w2 = _axis_interp(x[2], g.origin[2], g.inv_dx[2], g.dims[2])
    (ok1 && ok2) || return nothing
    a1 = one(w1) - w1
    a2 = one(w2) - w2
    _add_cell_source!(buf, CartesianIndex(i0, j0), s, a1 * a2, src)
    _add_cell_source!(buf, CartesianIndex(i1, j0), s, w1 * a2, src)
    _add_cell_source!(buf, CartesianIndex(i0, j1), s, a1 * w2, src)
    _add_cell_source!(buf, CartesianIndex(i1, j1), s, w1 * w2, src)
    return nothing
end

@inline function _deposit_cic!(
        buf::NeutralSourceBuffer, g::CartesianNeutralGrid{3, FT}, x::SVector{3},
        src::MomentSource, s::Int,
    ) where {FT}
    ok1, i0, i1, w1 = _axis_interp(x[1], g.origin[1], g.inv_dx[1], g.dims[1])
    ok2, j0, j1, w2 = _axis_interp(x[2], g.origin[2], g.inv_dx[2], g.dims[2])
    ok3, k0, k1, w3 = _axis_interp(x[3], g.origin[3], g.inv_dx[3], g.dims[3])
    (ok1 && ok2 && ok3) || return nothing
    a1 = one(w1) - w1
    a2 = one(w2) - w2
    a3 = one(w3) - w3
    _add_cell_source!(buf, CartesianIndex(i0, j0, k0), s, a1 * a2 * a3, src)
    _add_cell_source!(buf, CartesianIndex(i1, j0, k0), s, w1 * a2 * a3, src)
    _add_cell_source!(buf, CartesianIndex(i0, j1, k0), s, a1 * w2 * a3, src)
    _add_cell_source!(buf, CartesianIndex(i1, j1, k0), s, w1 * w2 * a3, src)
    _add_cell_source!(buf, CartesianIndex(i0, j0, k1), s, a1 * a2 * w3, src)
    _add_cell_source!(buf, CartesianIndex(i1, j0, k1), s, w1 * a2 * w3, src)
    _add_cell_source!(buf, CartesianIndex(i0, j1, k1), s, a1 * w2 * w3, src)
    _add_cell_source!(buf, CartesianIndex(i1, j1, k1), s, w1 * w2 * w3, src)
    return nothing
end

@inline function _deposit_cic!(
        buf::NeutralSourceBuffer, g::SphericalNeutralGrid{FT}, x::SVector{3},
        src::MomentSource, s::Int,
    ) where {FT}
    r, th, ph = _spherical_coords(g, x)
    @inbounds (r < g.r[1] || r > g.r[end]) && return nothing
    ir = _find_bracket(g.r, r)
    @inbounds wr = (r - g.r[ir]) / (g.r[ir + 1] - g.r[ir])
    ar = one(wr) - wr
    if g.ntheta == 1
        _add_cell_source!(buf, CartesianIndex(ir), s, ar, src)
        _add_cell_source!(buf, CartesianIndex(ir + 1), s, wr, src)
        return nothing
    end
    dth = FT(π) / FT(g.ntheta)
    sth = th / dth + FT(0.5)
    sthc = clamp(sth, one(sth), oftype(sth, g.ntheta))
    it0 = min(max(floor(Int, sthc), 1), g.ntheta - 1)
    it1 = it0 + 1
    wt = sthc - oftype(sthc, it0)
    at = one(wt) - wt
    if g.nphi == 1
        _add_cell_source!(buf, CartesianIndex(ir, it0), s, ar * at, src)
        _add_cell_source!(buf, CartesianIndex(ir + 1, it0), s, wr * at, src)
        _add_cell_source!(buf, CartesianIndex(ir, it1), s, ar * wt, src)
        _add_cell_source!(buf, CartesianIndex(ir + 1, it1), s, wr * wt, src)
        return nothing
    end
    dph = (FT(2) * FT(π)) / FT(g.nphi)
    sph = ph / dph + FT(0.5)
    ip0_raw = floor(Int, sph)
    wp = sph - oftype(sph, ip0_raw)
    ap = one(wp) - wp
    ip0 = mod1(ip0_raw, g.nphi)
    ip1 = mod1(ip0_raw + 1, g.nphi)
    _add_cell_source!(buf, CartesianIndex(ir, it0, ip0), s, ar * at * ap, src)
    _add_cell_source!(buf, CartesianIndex(ir + 1, it0, ip0), s, wr * at * ap, src)
    _add_cell_source!(buf, CartesianIndex(ir, it1, ip0), s, ar * wt * ap, src)
    _add_cell_source!(buf, CartesianIndex(ir + 1, it1, ip0), s, wr * wt * ap, src)
    _add_cell_source!(buf, CartesianIndex(ir, it0, ip1), s, ar * at * wp, src)
    _add_cell_source!(buf, CartesianIndex(ir + 1, it0, ip1), s, wr * at * wp, src)
    _add_cell_source!(buf, CartesianIndex(ir, it1, ip1), s, ar * wt * wp, src)
    _add_cell_source!(buf, CartesianIndex(ir + 1, it1, ip1), s, wr * wt * wp, src)
    return nothing
end

@inline function _deposit_neutral_source!(
        ::TwoWayCoupling,
        f::GriddedNeutralField,
        x::AbstractVector{<:Real},
        src::MomentSource,
        species::Integer,
    )
    buf = f.sources
    buf === nothing && return nothing
    _deposit_cic!(buf, f.grid, _as_svec3(x), src, Int(species))
    return nothing
end

@inline function _deposit_neutral_particle!(
        ::TwoWayCoupling, f::GriddedNeutralField, p::NeutralParticle,
    )
    pbuf = f.particles
    pbuf === nothing && return nothing
    push_particle!(pbuf, p)
    return nothing
end

# ------------------------------------------------------------------------------
# Sampling a Static Model onto a GriddedNeutralField
# ------------------------------------------------------------------------------

"""
    sample_to_grid!(field::GriddedNeutralField, model::AbstractNeutralModel;
                    t = nothing, kB = 1) -> GriddedNeutralField

Populate `field.n`, `field.u`, and `field.T` at every cell center from `model`, enabling a
seamless transition from a static one-way exosphere to a two-way dynamic grid.
"""
function sample_to_grid!(
        field::GriddedNeutralField, model::AbstractNeutralModel; t = nothing, kB::Number = 1,
    )
    ns = n_components(field)
    ns == n_components(model) || throw(
        ArgumentError(
            "sample_to_grid!: species count mismatch ($(ns) vs $(n_components(model)))",
        ),
    )
    FT = eltype(field.n)
    dims = _grid_dims(field.grid)
    for s in 1:ns
        for idx in CartesianIndices(dims)
            x = cell_center(field.grid, idx)
            st = neutral_state(model, x; t = t, species = s, kB = kB)
            @inbounds begin
                field.n[idx, s] = FT(st.n)
                field.u[idx, s] = SVector{3, FT}(FT(st.u[1]), FT(st.u[2]), FT(st.u[3]))
                field.T[idx, s] = FT(st.T)
            end
        end
    end
    return field
end
