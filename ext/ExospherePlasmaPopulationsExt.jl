module ExospherePlasmaPopulationsExt

import Exosphere
using Exosphere: AbstractNeutralProfile, BiMaxwellianNeutral, ColdNeutral, E_SI, KB_SI,
    Maxwellian, NeutralPopulation, NeutralSpecies, PlasmaState, UnitScale
import PlasmaPopulations as PP
using StaticArrays: SVector
using Statistics: mean

# ------------------------------------------------------------------------------
# Particle -> NeutralSpecies
# ------------------------------------------------------------------------------

function Exosphere.NeutralSpecies(p::PP.Particle)
    iszero(p.q) || throw(
        ArgumentError("NeutralSpecies(p::PlasmaPopulations.Particle) requires q == 0 (got q = $(p.q))")
    )
    return NeutralSpecies(; m = p.m)
end

function Exosphere.NeutralSpecies(scale::UnitScale{FT}, p::PP.Particle) where {FT}
    iszero(p.q) || throw(
        ArgumentError("NeutralSpecies(scale, p::PlasmaPopulations.Particle) requires q == 0 (got q = $(p.q))")
    )
    return NeutralSpecies(; m = FT(p.m / scale.m0))
end

# ------------------------------------------------------------------------------
# PlasmaPopulations.AbstractVDF -> Exosphere AbstractNeutralVDF
# ------------------------------------------------------------------------------

function Exosphere.ColdNeutral(d::PP.Cold)
    u = mean(d)
    return ColdNeutral(; u = SVector{3}(u[1], u[2], u[3]))
end

function Exosphere.Maxwellian(d::PP.BiMaxwellian)
    d.vth_perp == d.vth_para || throw(
        ArgumentError("Maxwellian(d::BiMaxwellian) requires isotropic vth_perp == vth_para; use BiMaxwellianNeutral")
    )
    u = mean(d)
    return Maxwellian(; vth = d.vth_para, u = SVector{3}(u[1], u[2], u[3]))
end

function Exosphere.BiMaxwellianNeutral(
        d::PP.BiMaxwellian; axis::Union{NTuple{3, Real}, SVector{3, <:Real}} = (0.0, 0.0, 1.0),
    )
    u = mean(d)
    return BiMaxwellianNeutral(;
        vth_perp = d.vth_perp, vth_para = d.vth_para,
        u = SVector{3}(u[1], u[2], u[3]), axis = axis,
    )
end

function Exosphere.NeutralPopulation(
        p::PP.Particle, profile::AbstractNeutralProfile; kw...,
    )
    return NeutralPopulation(NeutralSpecies(p), profile; kw...)
end

function Exosphere.NeutralPopulation(
        p::PP.Particle, profile::AbstractNeutralProfile, vdf::PP.Cold,
    )
    return NeutralPopulation(NeutralSpecies(p), profile, ColdNeutral(vdf))
end

function Exosphere.NeutralPopulation(
        p::PP.Particle, profile::AbstractNeutralProfile, vdf::PP.BiMaxwellian,
    )
    nvdf = vdf.vth_perp == vdf.vth_para ? Maxwellian(vdf) : BiMaxwellianNeutral(vdf)
    return NeutralPopulation(NeutralSpecies(p), profile, nvdf)
end

# ------------------------------------------------------------------------------
# PlasmaPopulations.Population -> Exosphere.PlasmaState
# ------------------------------------------------------------------------------

function Exosphere.PlasmaState(pop::PP.Population; energy_units::Bool = false)
    u_raw = mean(pop.vdf)
    u = SVector{3}(u_raw[1], u_raw[2], u_raw[3])
    t_perp_eV, t_para_eV = PP.temperature_perp_para(pop)
    kBT_J = ((2 * t_perp_eV + t_para_eV) / 3) * E_SI
    T_out = energy_units ? kBT_J : kBT_J / KB_SI
    return PlasmaState(pop.n, u, T_out, pop.particle.m)
end

function Exosphere.PlasmaState(scale::UnitScale{FT}, pop::PP.Population) where {FT}
    u_raw = mean(pop.vdf)
    u = SVector{3, FT}(FT(u_raw[1]) / scale.v0, FT(u_raw[2]) / scale.v0, FT(u_raw[3]) / scale.v0)
    t_perp_eV, t_para_eV = PP.temperature_perp_para(pop)
    kBT_J = ((2 * t_perp_eV + t_para_eV) / 3) * E_SI
    return PlasmaState(
        FT(pop.n) / scale.n0,
        u,
        FT(kBT_J) / scale.T0,
        FT(pop.particle.m) / scale.m0,
    )
end

end # module ExospherePlasmaPopulationsExt
