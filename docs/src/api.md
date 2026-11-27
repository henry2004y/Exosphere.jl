# API Reference

## Package Overview

```@docs
Exosphere
```

## Neutral Density Profiles

```@docs
AbstractNeutralProfile
ExponentialProfile
PowerLawProfile
ChamberlainProfile
ExosphereProfile
exosphere_profile
neutral_density
local_scale_height
```

## Multi-Component Exosphere & EUV Attenuation

```@docs
NeutralComponentCode
ShadowCylinder
OpticalDepth
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

## Ion–Neutral Source Rates

```@docs
photoionization_rate
charge_exchange_rate
```

## Adaptive Box Quadrature

```@docs
integrate_exosphere_box!
```

