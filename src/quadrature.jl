# ==============================================================================
# Adaptive Octree Dual-Cell Exosphere Box Quadrature
# ==============================================================================

const _MAX_CUBE_DEPTH = 8

@inline _quad_off4(::Type{FT}) where {FT <: AbstractFloat} =
    SVector{4, FT}(FT(-0.375), FT(-0.125), FT(0.125), FT(0.375))

@inline function _steepest_comp(
        comp, od, Rp::FT, inv_Rp::FT, rref::FT, inv_rref::FT, idx::Int,
    ) where {FT}
    n = _component_density(comp, Rp, inv_Rp, rref, inv_rref)
    n0_ref = comp isa NeutralComponentCode ? FT(comp.n0) :
        _component_density(comp, Rp, inv_Rp, Rp, inv_Rp)
    n <= FT(1.0e-20) * max(n0_ref, FT(1.0e-30)) && return FT(1.0e30)
    h = _scale_height_comp(comp, od, rref, idx)
    return h > zero(FT) ? FT(h) : FT(1.0e30)
end

@inline function _steepest_tuple(
        comps::Tuple{C}, od, Rp::FT, inv_Rp::FT, rref::FT, inv_rref::FT, idx::Int,
    ) where {C, FT}
    return _steepest_comp(comps[1], od, Rp, inv_Rp, rref, inv_rref, idx)
end

@inline function _steepest_tuple(
        comps::Tuple{C, Vararg}, od, Rp::FT, inv_Rp::FT, rref::FT, inv_rref::FT, idx::Int,
    ) where {C, FT}
    h1 = _steepest_comp(comps[1], od, Rp, inv_Rp, rref, inv_rref, idx)
    h2 = _steepest_tuple(Base.tail(comps), od, Rp, inv_Rp, rref, inv_rref, idx + 1)
    return min(h1, h2)
end

@inline function _steepest_scale_height(exo::ExosphereModel{FT, <:Tuple}, rmin::FT) where {FT}
    Rp = exo.Rp
    rref = max(rmin, Rp)
    inv_rref = rref > zero(FT) ? inv(rref) : zero(FT)
    return _steepest_tuple(exo.comps, exo.attenuation, Rp, exo.inv_Rp, rref, inv_rref, 1)
end

@inline function _steepest_scale_height(
        exo::ExosphereModel{FT, <:AbstractVector}, rmin::FT,
    ) where {FT}
    hmin = FT(1.0e30)
    Rp = exo.Rp
    inv_Rp = exo.inv_Rp
    rref = max(rmin, Rp)
    inv_rref = rref > zero(FT) ? inv(rref) : zero(FT)
    comps = exo.comps
    atten = exo.attenuation
    @inbounds for c in eachindex(comps)
        h = _steepest_comp(comps[c], atten, Rp, inv_Rp, rref, inv_rref, c)
        h < hmin && (hmin = h)
    end
    return hmin
end

@inline function _steepest_scale_height(
        model::AbstractNeutralModel, ce::SVector{3, FT}, t,
    ) where {FT}
    hmin = FT(1.0e30)
    nc = n_components(model)
    for c in 1:nc
        h = FT(local_scale_height(model, ce; t = t, species = c))
        h > zero(FT) && (hmin = min(hmin, h))
    end
    return hmin
end

@inline function _accum_one_comp!(
        sumN, sumNA, comp, atten, Rp::FT, inv_Rp::FT, ce::SVector{3, FT}, re::FT, inv_re::FT,
        straddle::Bool, a1::FT, a2::FT, a3::FT, w::FT, w_att::FT, k::Int, t,
    ) where {FT}
    n = FT(_component_density_3d(comp, Rp, inv_Rp, ce, re, inv_re, t))
    n <= zero(FT) && return nothing
    if !straddle
        hC = FT(_scale_height_comp(comp, atten, re, k))
        if hC > zero(FT)
            inv_hC = inv(hC)
            b1 = a1 * inv_hC
            b2 = a2 * inv_hC
            b3 = a3 * inv_hC
            b1 > FT(1.0e-6) && (n *= sinh(b1) / b1)
            b2 > FT(1.0e-6) && (n *= sinh(b2) / b2)
            b3 > FT(1.0e-6) && (n *= sinh(b3) / b3)
        end
    end
    @inbounds begin
        sumN[k] += n * w
        sumNA[k] += n * w_att
    end
    return nothing
end

@inline function _accum_tuple_comps!(
        sumN, sumNA, comps::Tuple{C}, atten, Rp::FT, inv_Rp::FT, ce::SVector{3, FT}, re::FT,
        inv_re::FT, straddle::Bool, a1::FT, a2::FT, a3::FT, w::FT, w_att::FT, k::Int, t,
    ) where {C, FT}
    _accum_one_comp!(
        sumN, sumNA, comps[1], atten, Rp, inv_Rp, ce, re, inv_re, straddle, a1, a2, a3, w,
        w_att, k, t,
    )
    return nothing
end

@inline function _accum_tuple_comps!(
        sumN, sumNA, comps::Tuple{C, Vararg}, atten, Rp::FT, inv_Rp::FT, ce::SVector{3, FT},
        re::FT, inv_re::FT, straddle::Bool, a1::FT, a2::FT, a3::FT, w::FT, w_att::FT, k::Int,
        t,
    ) where {C, FT}
    _accum_one_comp!(
        sumN, sumNA, comps[1], atten, Rp, inv_Rp, ce, re, inv_re, straddle, a1, a2, a3, w,
        w_att, k, t,
    )
    _accum_tuple_comps!(
        sumN, sumNA, Base.tail(comps), atten, Rp, inv_Rp, ce, re, inv_re, straddle, a1, a2,
        a3, w, w_att, k + 1, t,
    )
    return nothing
end

@inline @muladd function _straddle_fraction(
        c::SVector{3, FT}, h::SVector{3, FT}, rcut::FT,
    ) where {FT}
    rcut2 = rcut * rcut
    offs = _quad_off4(FT)
    qx = c[1] .+ offs .* h[1]
    qy = c[2] .+ offs .* h[2]
    qx2 = qx .* qx
    qy2 = qy .* qy
    nin = 0
    if h[3] <= zero(FT)
        qz2_scalar = c[3] * c[3]
        @inbounds for a in 1:4
            qxz2 = qx2[a] + qz2_scalar
            nin += count(>(rcut2), qxz2 .+ qy2)
        end
        return FT(nin) * FT(1.0 / 16.0)
    else
        qz = c[3] .+ offs .* h[3]
        qz2 = qz .* qz
        @inbounds for a in 1:4, b in 1:4
            qxy2 = qx2[a] + qy2[b]
            nin += count(>(rcut2), qxy2 .+ qz2)
        end
        return FT(nin) * FT(1.0 / 64.0)
    end
end

"""
    integrate_exosphere_box!(sumN, sumNA, exo::AbstractNeutralModel, c::SVector{3, FT},
                             h::SVector{3, FT}, depth::Int, rcut::FT;
                             t = nothing) -> FT

Adaptively accumulate per-component neutral density (`sumN`) and EUV-attenuated neutral
density (`sumNA`) in place over an axis-aligned box centered at `c` (relative to the planet
center) with side widths `h`, excluding the interior sphere `r < rcut`. Pass `depth = 0` at
the top level; set `h[d] <= 0` for collapsed 2D axes. Non-straddling cells apply the
analytic ``\\sinh(b)/b`` finite-cell correction along each axis. Returns the integrated
exterior volume (or area when one axis is collapsed).
"""
@muladd function integrate_exosphere_box!(
        sumN::AbstractVector,
        sumNA::AbstractVector,
        exo::ExosphereModel{FT},
        c::SVector{3, FT},
        h::SVector{3, FT},
        depth::Int,
        rcut::FT;
        t = nothing,
    ) where {FT <: AbstractFloat}
    r = sqrt(c[1] * c[1] + c[2] * c[2] + c[3] * c[3])
    half = FT(0.5) * sqrt(h[1] * h[1] + h[2] * h[2] + h[3] * h[3])
    vol = (h[1] > zero(FT) ? h[1] : one(FT)) *
        (h[2] > zero(FT) ? h[2] : one(FT)) *
        (h[3] > zero(FT) ? h[3] : one(FT))
    r + half <= rcut && return zero(FT)
    straddle = r - half < rcut
    hmax = max(h[1], h[2], h[3])
    hs = _steepest_scale_height(exo, r - half)
    split = straddle ? (hmax > FT(0.5) * hs && hs < FT(1.0e29)) : hmax > FT(2.0) * hs
    if split && depth < _MAX_CUBE_DEPTH
        vout = zero(FT)
        hh = SVector{3, FT}(
            h[1] > zero(FT) ? FT(0.5) * h[1] : zero(FT),
            h[2] > zero(FT) ? FT(0.5) * h[2] : zero(FT),
            h[3] > zero(FT) ? FT(0.5) * h[3] : zero(FT),
        )
        for ic in 0:7
            s1 = (ic & 1) == 1 ? 1 : -1
            s2 = ((ic >> 1) & 1) == 1 ? 1 : -1
            s3 = ((ic >> 2) & 1) == 1 ? 1 : -1
            if (h[1] <= zero(FT) && s1 > 0) ||
                    (h[2] <= zero(FT) && s2 > 0) ||
                    (h[3] <= zero(FT) && s3 > 0)
                continue
            end
            cc = SVector{3, FT}(
                h[1] > zero(FT) ? c[1] + FT(s1) * FT(0.25) * h[1] : c[1],
                h[2] > zero(FT) ? c[2] + FT(s2) * FT(0.25) * h[2] : c[2],
                h[3] > zero(FT) ? c[3] + FT(s3) * FT(0.25) * h[3] : c[3],
            )
            vout += integrate_exosphere_box!(sumN, sumNA, exo, cc, hh, depth + 1, rcut; t = t)
        end
        return vout
    end
    fout = straddle ? _straddle_fraction(c, h, rcut) : one(FT)
    fout <= zero(FT) && return zero(FT)
    ce = c
    re = r
    if straddle && r > zero(FT)
        rE = max(r, rcut + FT(0.25) * half)
        ce = c * (rE / r)
        re = rE
    end
    w = fout * vol
    re <= zero(FT) && return w
    att = FT(photo_attenuation(exo, ce, re; t = t))
    w_att = att * w
    Rp = exo.Rp
    re < Rp && return w
    inv_Rp = exo.inv_Rp
    inv_re = inv(re)
    a1 = FT(0.5) * h[1] * abs(ce[1]) * inv_re
    a2 = FT(0.5) * h[2] * abs(ce[2]) * inv_re
    a3 = FT(0.5) * h[3] * abs(ce[3]) * inv_re
    comps = exo.comps
    atten = exo.attenuation
    if comps isa Tuple
        _accum_tuple_comps!(
            sumN, sumNA, comps, atten, Rp, inv_Rp, ce, re, inv_re, straddle, a1, a2, a3, w,
            w_att, 1, t,
        )
    else
        @inbounds for k in eachindex(comps)
            _accum_one_comp!(
                sumN, sumNA, comps[k], atten, Rp, inv_Rp, ce, re, inv_re, straddle, a1, a2,
                a3, w, w_att, k, t,
            )
        end
    end
    return w
end

@muladd function integrate_exosphere_box!(
        sumN::AbstractVector,
        sumNA::AbstractVector,
        model::AbstractNeutralModel,
        c::SVector{3, FT},
        h::SVector{3, FT},
        depth::Int,
        rcut::FT;
        t = nothing,
    ) where {FT <: AbstractFloat}
    r = sqrt(c[1] * c[1] + c[2] * c[2] + c[3] * c[3])
    half = FT(0.5) * sqrt(h[1] * h[1] + h[2] * h[2] + h[3] * h[3])
    vol = (h[1] > zero(FT) ? h[1] : one(FT)) *
        (h[2] > zero(FT) ? h[2] : one(FT)) *
        (h[3] > zero(FT) ? h[3] : one(FT))
    r + half <= rcut && return zero(FT)
    straddle = r - half < rcut
    hmax = max(h[1], h[2], h[3])
    hs = _steepest_scale_height(model, c, t)
    split = straddle ? (hmax > FT(0.5) * hs && hs < FT(1.0e29)) : hmax > FT(2.0) * hs
    if split && depth < _MAX_CUBE_DEPTH
        vout = zero(FT)
        hh = SVector{3, FT}(
            h[1] > zero(FT) ? FT(0.5) * h[1] : zero(FT),
            h[2] > zero(FT) ? FT(0.5) * h[2] : zero(FT),
            h[3] > zero(FT) ? FT(0.5) * h[3] : zero(FT),
        )
        for ic in 0:7
            s1 = (ic & 1) == 1 ? 1 : -1
            s2 = ((ic >> 1) & 1) == 1 ? 1 : -1
            s3 = ((ic >> 2) & 1) == 1 ? 1 : -1
            if (h[1] <= zero(FT) && s1 > 0) ||
                    (h[2] <= zero(FT) && s2 > 0) ||
                    (h[3] <= zero(FT) && s3 > 0)
                continue
            end
            cc = SVector{3, FT}(
                h[1] > zero(FT) ? c[1] + FT(s1) * FT(0.25) * h[1] : c[1],
                h[2] > zero(FT) ? c[2] + FT(s2) * FT(0.25) * h[2] : c[2],
                h[3] > zero(FT) ? c[3] + FT(s3) * FT(0.25) * h[3] : c[3],
            )
            vout += integrate_exosphere_box!(sumN, sumNA, model, cc, hh, depth + 1, rcut; t = t)
        end
        return vout
    end
    fout = straddle ? _straddle_fraction(c, h, rcut) : one(FT)
    fout <= zero(FT) && return zero(FT)
    ce = c
    re = r
    if straddle && r > zero(FT)
        rE = max(r, rcut + FT(0.25) * half)
        ce = c * (rE / r)
        re = rE
    end
    w = fout * vol
    re <= zero(FT) && return w
    att = FT(photo_attenuation(model, ce; t = t))
    w_att = att * w
    nc = n_components(model)
    @inbounds for k in 1:nc
        n = FT(neutral_density(model, ce; t = t, species = k))
        n <= zero(FT) && continue
        sumN[k] += n * w
        sumNA[k] += n * w_att
    end
    return w
end
