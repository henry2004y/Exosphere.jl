"""
    Exosphere

Standalone, zero-allocation planetary neutral exosphere models, `PlasmaPopulations.jl`-style
neutral species and VDFs, EUV shadow and optical-depth/Chapman/ray attenuation, conservative
one-way and two-way charge–neutral interaction kernels, structured neutral field containers,
and adaptive dual-cell box quadrature.
"""
module Exosphere

using LinearAlgebra: cross, norm
using MuladdMacro: @muladd
using PrecompileTools: @compile_workload, @setup_workload
import Random
using Random: AbstractRNG, randn
using StaticArrays: SMatrix, SVector
import Statistics
using Statistics: mean

include("interface.jl")
include("species.jl")
include("vdf.jl")
include("profiles.jl")
include("attenuation.jl")
include("sources.jl")
include("interactions.jl")
include("fields.jl")
include("quadrature.jl")

# Core interfaces, traits, states & scaling
export AbstractNeutralModel, AbstractNeutralProfile, AbstractNeutralVDF
export AbstractVelocityModel, AbstractTemperatureModel, AbstractAttenuation
export AbstractInteractionChannel, AbstractNeutralGrid
export AbstractCouplingMode, OneWayCoupling, TwoWayCoupling, coupling_mode
export NeutralState, PlasmaState, MomentSource, CoupledExchange, NeutralParticle
export neutral_state, local_vdf, species_mass
export deposit_neutral_source!, deposit_neutral_particle!, reset_sources!
export neutral_sources, neutral_particles
export UnitScale, to_code_units, from_code_units, strip_units

# Neutral species & VDFs (PlasmaPopulations.jl style)
export NeutralSpecies, atomic_mass
export ColdNeutral, Maxwellian, BiMaxwellianNeutral
export pdf, thermal_speed, vdf_temperature, temperature_perp_para

# Profiles, kinematics & populations
export ExponentialProfile, PowerLawProfile, ChamberlainProfile
export ExosphereProfile, exosphere_profile, neutral_density, local_scale_height
export DayNightProfile, PlumeProfile, CompositeProfile
export TabulatedRadialProfile, TimeDependentProfile
export ZeroVelocity, ConstantVelocity, CorotationVelocity, RadialOutflowVelocity
export ConstantTemperature, DayNightTemperature
export neutral_velocity, neutral_temperature
export NeutralPopulation, NeutralComponentCode

# Attenuation & ExosphereModel
export NoAttenuation, ShadowCylinder, OpticalDepth, RayOpticalDepth
export ExosphereModel, build_exosphere_model
export in_shadow, shadow_factor
export n_components, exo_component_density, exo_density, exo_scale_height
export chapman_function, photo_attenuation

# Sources & conservative charge-neutral interactions
export photoionization_rate, charge_exchange_rate
export effective_relative_speed
export Photoionization, ElectronImpactIonization
export ResonantChargeExchange, NonResonantChargeExchange, NonResonantExchange
export ElasticCollision, Recombination
export compute_exchange, compute_nonresonant_exchange, couple_point!
export collision_probability, particle_charge_exchange_rate
export photoionization_creation_weight, charge_exchange_collision, sputtering_flux

# Structured grids & two-way dynamic neutral fields
export CartesianNeutralGrid, SphericalNeutralGrid, cell_center, cell_volume
export NeutralSourceBuffer, NeutralParticleBuffer, merge_sources!, active_particles
export GriddedNeutralField, sample_to_grid!

# Adaptive quadrature
export integrate_exosphere_box!

public U_SI, MP_SI, ME_SI, KB_SI, E_SI
public si_mass, si_length, si_number_density, si_velocity
public si_temperature, si_energy, si_cross_section, si_frequency
public push_particle!

@setup_workload begin
    @compile_workload begin
        ep = exosphere_profile(; n0 = 1.0, r0 = 1.0, H = 0.5)
        _ = ep(1.5, 0.2, 0.0)
        sc = ShadowCylinder(; solar_dir = (1.0, 0.0, 0.0), radius = 1.0)
        _ = shadow_factor(sc, -1.5, 0.0, 0.0)
        c = NeutralComponentCode(:exponential, 1.0, 0.5, 0.0, 0.1, "H")
        od = OpticalDepth(; cross_sections = [1.0e-2], chapman = true)
        exo = build_exosphere_model(; planet_radius = 1.0, components = [c], optical_depth = od)
        n = exo_density(exo, 1.5)
        att = photo_attenuation(exo, SVector(1.5, 0.0, 0.0), 1.5)
        _ = photoionization_rate(1.0e-6, n, att)
        _ = charge_exchange_rate(1.0e-18, 1.0, n, 1.0e5)
        sumN = zeros(1)
        sumNA = zeros(1)
        _ = integrate_exosphere_box!(
            sumN, sumNA, exo, SVector(1.5, 0.0, 0.0), SVector(0.2, 0.2, 0.2), 0, 1.0,
        )

        # Modern isbits tuple ExosphereModel & one-way couple_point!
        sp_o = NeutralSpecies(; m = 16.0)
        pop_o = NeutralPopulation(
            sp_o,
            ExponentialProfile(100.0, 1.0, 0.2);
            temperature = ConstantTemperature(0.01),
        )
        od_s = OpticalDepth(; cross_sections = (0.05,), chapman = true)
        exo_s = ExosphereModel(pop_o; planet_radius = 1.0, optical_depth = od_s)
        ion = PlasmaState(5.0, SVector(-2.0, 0.0, 0.0), 0.1, 16.0)
        ch = (Photoionization(1.0e-6), ResonantChargeExchange(2.0e-3))
        _ = couple_point!(exo_s, SVector(1.5, 0.0, 0.0), ch, ion; species = 1)
    end
end

end # module Exosphere
