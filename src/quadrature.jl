# ==============================================================================
# Adaptive Octree Dual-Cell Exosphere Box Quadrature
# ==============================================================================

const _MAX_CUBE_DEPTH = 8
const _QUAD_OFF4 = SVector{4, Float64}(-0.375, -0.125, 0.125, 0.375)

@inline function _steepest_scale_height(exo::ExosphereModel, rmin::Float64)
    hmin = 1.0e300
    Rp = exo.Rp
    inv_Rp = exo.inv_Rp
    rref = max(rmin, Rp)
    inv_rref = rref > 0.0 ? inv(rref) : 0.0
    comps = exo.comps
    od = exo.optical
    @inbounds for c in eachindex(comps)
        p = comps[c]
        n = _component_density(p, Rp, inv_Rp, rref, inv_rref)
        n <= 1.0e-20 * max(p.n0, 1.0e-300) && continue
        h = od === nothing ? _scale_height_at(comps, nothing, rref, c) : _scale_height_at(comps, od, rref, c)
        h > 0.0 && (hmin = min(hmin, h))
    end
    return hmin
end

"""
    integrate_exosphere_box!(sumN, sumNA, exo::ExosphereModel, c::SVector{3, Float64},
                             h::SVector{3, Float64}, depth::Int, rcut::Float64) -> Float64

Adaptively integrate per-component neutral density (`sumN`) and EUV-attenuated neutral
density (`sumNA`) over a box centered at `c` (relative to `exo.center`) with side widths
`h` (set `h[d] == 0` for collapsed 2D axes), excluding the sphere `r < rcut`. Returns the
integrated exterior volume.
"""
@muladd function integrate_exosphere_box!(
        sumN, sumNA, exo::ExosphereModel, c::SVector{3, Float64}, h::SVector{3, Float64},
        depth::Int, rcut::Float64,
    )
    r = sqrt(c[1] * c[1] + c[2] * c[2] + c[3] * c[3])
    half = 0.5 * sqrt(h[1] * h[1] + h[2] * h[2] + h[3] * h[3])
    vol = (h[1] > 0.0 ? h[1] : 1.0) * (h[2] > 0.0 ? h[2] : 1.0) * (h[3] > 0.0 ? h[3] : 1.0)
    r + half <= rcut && return 0.0
    straddle = r - half < rcut
    hmax = max(h[1], h[2], h[3])
    hs = _steepest_scale_height(exo, r - half)
    split = straddle ? (hmax > 0.5 * hs && hs < 1.0e299) : hmax > 2.0 * hs
    if split && depth < _MAX_CUBE_DEPTH
        vout = 0.0
        hh = SVector{3, Float64}(
            h[1] > 0.0 ? 0.5 * h[1] : 0.0,
            h[2] > 0.0 ? 0.5 * h[2] : 0.0,
            h[3] > 0.0 ? 0.5 * h[3] : 0.0,
        )
        for ic in 0:7
            s1 = (ic & 1) == 1 ? 1 : -1
            s2 = ((ic >> 1) & 1) == 1 ? 1 : -1
            s3 = ((ic >> 2) & 1) == 1 ? 1 : -1
            if (h[1] <= 0.0 && s1 > 0) || (h[2] <= 0.0 && s2 > 0) || (h[3] <= 0.0 && s3 > 0)
                continue
            end
            cc = SVector{3, Float64}(
                h[1] > 0.0 ? c[1] + s1 * 0.25 * h[1] : c[1],
                h[2] > 0.0 ? c[2] + s2 * 0.25 * h[2] : c[2],
                h[3] > 0.0 ? c[3] + s3 * 0.25 * h[3] : c[3],
            )
            vout += integrate_exosphere_box!(sumN, sumNA, exo, cc, hh, depth + 1, rcut)
        end
        return vout
    end
    fout = 1.0
    if straddle
        rcut2 = rcut * rcut
        qx = c[1] .+ _QUAD_OFF4 .* h[1]
        qy = c[2] .+ _QUAD_OFF4 .* h[2]
        qx2 = qx .* qx
        qy2 = qy .* qy
        nin = 0
        if h[3] <= 0.0
            qz2_scalar = c[3] * c[3]
            @inbounds for a in 1:4
                qxz2 = qx2[a] + qz2_scalar
                nin += count(>(rcut2), qxz2 .+ qy2)
            end
            fout = nin * (1.0 / 16.0)
        else
            qz = c[3] .+ _QUAD_OFF4 .* h[3]
            qz2 = qz .* qz
            @inbounds for a in 1:4, b in 1:4
                qxy2 = qx2[a] + qy2[b]
                nin += count(>(rcut2), qxy2 .+ qz2)
            end
            fout = nin * (1.0 / 64.0)
        end
    end
    fout <= 0.0 && return 0.0
    ce = c
    re = r
    if straddle && r > 0.0
        rE = max(r, rcut + 0.25 * half)
        ce = c * (rE / r)
        re = rE
    end
    w = fout * vol
    re <= 0.0 && return w
    att = photo_attenuation(exo, ce, re)
    w_att = att * w
    Rp = exo.Rp
    inv_Rp = exo.inv_Rp
    inv_re = inv(re)
    re < Rp && return w
    comps = exo.comps
    od = exo.optical
    a1 = 0.5 * h[1] * abs(ce[1]) * inv_re
    a2 = 0.5 * h[2] * abs(ce[2]) * inv_re
    a3 = 0.5 * h[3] * abs(ce[3]) * inv_re
    @inbounds for k in eachindex(comps)
        n = _component_density(comps[k], Rp, inv_Rp, re, inv_re)
        n <= 0.0 && continue
        if !straddle
            hC = od === nothing ? _scale_height_at(comps, nothing, re, k) : _scale_height_at(comps, od, re, k)
            if hC > 0.0
                inv_hC = inv(hC)
                b1 = a1 * inv_hC
                b2 = a2 * inv_hC
                b3 = a3 * inv_hC
                b1 > 1.0e-6 && (n *= sinh(b1) / b1)
                b2 > 1.0e-6 && (n *= sinh(b2) / b2)
                b3 > 1.0e-6 && (n *= sinh(b3) / b3)
            end
        end
        sumN[k] += n * w
        sumNA[k] += n * w_att
    end
    return w
end
