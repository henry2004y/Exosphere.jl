# ==============================================================================
# Pointwise Ion-Neutral Photoionization & Charge-Exchange Source Rates
# ==============================================================================

"""
    photoionization_rate(nu_photo::Real, neutral_n::Real, atten::Real = 1.0)

Local volumetric photoionization production rate `nu_photo * neutral_n * atten`.
"""
@inline function photoionization_rate(nu_photo::Real, neutral_n::Real, atten::Real = 1.0)
    return nu_photo * neutral_n * atten
end

"""
    charge_exchange_rate(sigma_cx::Real, ion_n::Real, neutral_n::Real, v_rel::Real)

Local resonant charge-exchange collision rate `sigma_cx * ion_n * neutral_n * v_rel`.
"""
@inline function charge_exchange_rate(
        sigma_cx::Real, ion_n::Real, neutral_n::Real, v_rel::Real,
    )
    return sigma_cx * ion_n * neutral_n * v_rel
end
