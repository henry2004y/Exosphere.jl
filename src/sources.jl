# ==============================================================================
# Pointwise Ion-Neutral Photoionization & Charge-Exchange Source Rates
# ==============================================================================

"""
    photoionization_rate(nu_photo::Number, neutral_n::Number, atten::Number = 1)

Local volumetric photoionization production rate `nu_photo * neutral_n * atten`.
Supports `Float32`, `Float64`, `ForwardDiff.Dual`, and `Unitful.Quantity` inputs.
"""
@inline function photoionization_rate(
        nu_photo::Number, neutral_n::Number, atten::Number = 1,
    )
    return nu_photo * neutral_n * atten
end

"""
    charge_exchange_rate(sigma_cx::Number, ion_n::Number, neutral_n::Number, v_rel::Number)

Local resonant charge-exchange collision rate `sigma_cx * ion_n * neutral_n * v_rel`.
Supports `Float32`, `Float64`, `ForwardDiff.Dual`, and `Unitful.Quantity` inputs.
"""
@inline function charge_exchange_rate(
        sigma_cx::Number, ion_n::Number, neutral_n::Number, v_rel::Number,
    )
    return sigma_cx * ion_n * neutral_n * v_rel
end
