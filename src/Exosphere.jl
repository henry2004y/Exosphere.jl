"""
    Exosphere

Standalone, zero-allocation planetary neutral exosphere profiles, EUV shadow and
optical-depth/Chapman attenuation, ion–neutral source rates, and adaptive dual-cell
box quadrature.
"""
module Exosphere

using LinearAlgebra: dot, norm
using MuladdMacro: @muladd
using PrecompileTools: @compile_workload, @setup_workload
using StaticArrays: SVector

include("profiles.jl")
include("attenuation.jl")
include("sources.jl")
include("quadrature.jl")

export AbstractNeutralProfile, ExponentialProfile, PowerLawProfile, ChamberlainProfile
export ExosphereProfile, exosphere_profile, neutral_density, local_scale_height
export NeutralComponentCode, ShadowCylinder, OpticalDepth, ExosphereModel, build_exosphere_model
export in_shadow, shadow_factor
export n_components, exo_component_density, exo_density, exo_scale_height
export chapman_function, photo_attenuation
export photoionization_rate, charge_exchange_rate
export integrate_exosphere_box!

@setup_workload begin
    @compile_workload begin
        ep = exosphere_profile(; n0 = 1.0, r0 = 1.0, H = 0.5)
        _ = ep(1.5, 0.2, 0.0)
        c = NeutralComponentCode(:exponential, 1.0, 0.5, 0.0, 0.1, "H")
        od = OpticalDepth(; cross_sections = [1.0e-2], chapman = true)
        exo = build_exosphere_model(; planet_radius = 1.0, components = [c], optical_depth = od)
        _ = exo_density(exo, 1.5)
        _ = photo_attenuation(exo, SVector(1.5, 0.0, 0.0), 1.5)
    end
end

end # module Exosphere
