# API Reference

## Package Overview

```@docs
Exosphere
```

## Core Interfaces, Coupling Traits & State Containers

```@docs
AbstractNeutralModel
AbstractNeutralProfile
AbstractNeutralVDF
AbstractVelocityModel
AbstractTemperatureModel
AbstractAttenuation
AbstractInteractionChannel
AbstractNeutralGrid
AbstractCouplingMode
OneWayCoupling
TwoWayCoupling
coupling_mode
NeutralState
PlasmaState
MomentSource
CoupledExchange
NeutralParticle
neutral_state
local_vdf
species_mass
deposit_neutral_source!
deposit_neutral_particle!
reset_sources!
neutral_sources
neutral_particles
UnitScale
to_code_units
from_code_units
strip_units
```

## Neutral Species & Velocity Distribution Functions

```@docs
NeutralSpecies
atomic_mass
ColdNeutral
Maxwellian
BiMaxwellianNeutral
pdf
thermal_speed
vdf_temperature
temperature_perp_para
```

## Neutral Density Profiles, Kinematics & Populations

```@docs
ExponentialProfile
PowerLawProfile
ChamberlainProfile
ExosphereProfile
exosphere_profile
neutral_density
local_scale_height
DayNightProfile
PlumeProfile
CompositeProfile
TabulatedRadialProfile
TimeDependentProfile
ZeroVelocity
ConstantVelocity
CorotationVelocity
RadialOutflowVelocity
ConstantTemperature
DayNightTemperature
neutral_velocity
neutral_temperature
NeutralPopulation
NeutralComponentCode
```

## Multi-Component Exosphere & EUV Attenuation

```@docs
NoAttenuation
ShadowCylinder
OpticalDepth
RayOpticalDepth
ExosphereModel
build_exosphere_model
in_shadow
shadow_factor
n_components
exo_component_density
exo_density
exo_scale_height
chapman_function
photo_attenuation
```

## Ion–Neutral Source Rates & Conservative Interactions

```@docs
photoionization_rate
charge_exchange_rate
effective_relative_speed
Photoionization
ElectronImpactIonization
ResonantChargeExchange
NonResonantChargeExchange
NonResonantExchange
ElasticCollision
Recombination
compute_exchange
compute_nonresonant_exchange
couple_point!
collision_probability
particle_charge_exchange_rate
photoionization_creation_weight
charge_exchange_collision
sputtering_flux
```

## Structured Grids & Two-Way Dynamic Neutral Fields

```@docs
CartesianNeutralGrid
SphericalNeutralGrid
cell_center
cell_volume
NeutralSourceBuffer
NeutralParticleBuffer
merge_sources!
active_particles
GriddedNeutralField
sample_to_grid!
```

## Adaptive Box Quadrature

```@docs
integrate_exosphere_box!
```

## Public Constants & Low-Level Hooks

```@docs
Exosphere.U_SI
Exosphere.MP_SI
Exosphere.ME_SI
Exosphere.KB_SI
Exosphere.E_SI
Exosphere.si_mass
Exosphere.si_length
Exosphere.si_number_density
Exosphere.si_velocity
Exosphere.si_temperature
Exosphere.si_energy
Exosphere.si_cross_section
Exosphere.si_frequency
Exosphere.push_particle!
```
