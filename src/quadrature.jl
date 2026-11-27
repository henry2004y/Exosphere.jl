# ==============================================================================
# Adaptive Octree Dual-Cell Exosphere Box Quadrature
# ==============================================================================

const _MAX_CUBE_DEPTH = 8

function _steepest_scale_height(exo::ExosphereModel, rmin::Float64)
    hmin = 1.0e300
    rref = max(rmin, exo.Rp)
    for c in eachindex(exo.comps)
        n = exo_component_density(exo, rref, c)
        n <= 1.0e-20 * max(exo.comps[c].n0, 1.0e-300) && continue
        h = exo_scale_height(exo, rref, c)
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
function integrate_exosphere_box!(
        sumN, sumNA, exo::ExosphereModel, c::SVector{3, Float64}, h::SVector{3, Float64},
        depth::Int, rcut::Float64,
    )
    r = norm(c)
    half = 0.5 * norm(h)
    vol = prod(hd -> hd > 0.0 ? hd : 1.0, h)
    r + half <= rcut && return 0.0
    straddle = r - half < rcut
    hmax = maximum(h)
    hs = _steepest_scale_height(exo, r - half)
    split = straddle ? (hmax > 0.5 * hs && hs < 1.0e299) : hmax > 2.0 * hs
    if split && depth < _MAX_CUBE_DEPTH
        vout = 0.0
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
            hh = SVector{3, Float64}(
                h[1] > 0.0 ? 0.5 * h[1] : 0.0,
                h[2] > 0.0 ? 0.5 * h[2] : 0.0,
                h[3] > 0.0 ? 0.5 * h[3] : 0.0,
            )
            vout += integrate_exosphere_box!(sumN, sumNA, exo, cc, hh, depth + 1, rcut)
        end
        return vout
    end
    fout = 1.0
    if straddle
        nin = 0
        for a in 0:3, b in 0:3, e in 0:3
            off = ((a + 0.5) / 4 - 0.5, (b + 0.5) / 4 - 0.5, (e + 0.5) / 4 - 0.5)
            q2 = 0.0
            for d in 1:3
                q2 += (c[d] + off[d] * h[d])^2
            end
            q2 > rcut^2 && (nin += 1)
        end
        fout = nin / 64
    end
    fout <= 0.0 && return 0.0
    ce = c
    re = r
    if straddle && r > 0.0
        rE = max(r, rcut + 0.25 * half)
        ce = c * (rE / r)
        re = rE
    end
    re <= 0.0 && return fout * vol
    att = photo_attenuation(exo, ce, re)
    for k in eachindex(exo.comps)
        n = exo_component_density(exo, re, k)
        n <= 0.0 && continue
        hC = exo_scale_height(exo, re, k)
        if hC > 0.0 && !straddle
            for d in 1:3
                a = 0.5 * h[d] * abs(ce[d] / re) / hC
                a > 1.0e-6 && (n *= sinh(a) / a)
            end
        end
        sumN[k] += n * fout * vol
        sumNA[k] += n * att * fout * vol
    end
    return fout * vol
end
