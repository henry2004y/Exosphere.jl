# ==============================================================================
# Neutral Particle Species (PlasmaPopulations.jl Style)
# ==============================================================================

# Standard atomic weights [u] (IUPAC 2021) for space, planetary & cometary atmospheres.
const _ELEMENTS = Dict{Symbol, Float64}(
    :H => 1.008,
    :He => 4.002602,
    :Li => 6.94,
    :Be => 9.0121831,
    :B => 10.81,
    :C => 12.011,
    :N => 14.007,
    :O => 15.999,
    :F => 18.998403163,
    :Ne => 20.1797,
    :Na => 22.98976928,
    :Mg => 24.305,
    :Al => 26.9815385,
    :Si => 28.085,
    :P => 30.973761998,
    :S => 32.06,
    :Cl => 35.45,
    :Ar => 39.948,
    :K => 39.0983,
    :Ca => 40.078,
    :Ti => 47.867,
    :Cr => 51.9961,
    :Mn => 54.938044,
    :Fe => 55.845,
    :Co => 58.933194,
    :Ni => 58.6934,
    :Cu => 63.546,
    :Zn => 65.38,
    :Kr => 83.798,
    :Xe => 131.293,
)

# Single-letter isotope shorthand aliases.
const _ISOTOPE_SYMBOLS = Dict{Symbol, Tuple{Symbol, Int}}(
    :D => (:H, 2),
    :T => (:H, 3),
)

# Exact nuclide masses [u] (AME2020) for common planetary/cometary isotopes.
const _ISOTOPES = Dict{Tuple{Symbol, Int}, Float64}(
    (:H, 1) => 1.007825031898,
    (:H, 2) => 2.014101777844,
    (:H, 3) => 3.01604928132,
    (:He, 3) => 3.016029321967,
    (:He, 4) => 4.00260325413,
    (:C, 12) => 12.0,
    (:C, 13) => 13.00335483507,
    (:N, 14) => 14.00307400443,
    (:N, 15) => 15.00010889888,
    (:O, 16) => 15.99491461957,
    (:O, 17) => 16.9991317565,
    (:O, 18) => 17.99915961286,
    (:Ne, 20) => 19.9924401762,
    (:Ne, 22) => 21.991385114,
    (:Na, 23) => 22.989769282,
    (:Mg, 24) => 23.985041697,
    (:Si, 28) => 27.97692653465,
    (:S, 32) => 31.9720711744,
    (:S, 34) => 33.967867004,
    (:Ar, 36) => 35.967545105,
    (:Ar, 40) => 39.9623831237,
    (:K, 39) => 38.9637064864,
    (:Ca, 40) => 39.962590863,
    (:Fe, 56) => 55.93493633,
)

"""
    NeutralSpecies{T<:Real}
    NeutralSpecies(; m)
    NeutralSpecies(formula::Union{Symbol, AbstractString})
    NeutralSpecies(scale::UnitScale, formula::Union{Symbol, AbstractString})

Lightweight `isbits` neutral atom or molecule species with mass `m`.

- `NeutralSpecies(; m)` stores bare numeric `m` as given (code units or SI kg) and
  converts `Unitful.Quantity` inputs to kilograms when `Unitful` is loaded.
- `NeutralSpecies(formula)` parses chemical and nuclide formulas (such as `:H`, `:O`,
  `"CO2"`, `"H2O"`, `"SO2"`, `"D"`, `"{18}O"`) and stores the standard molecular mass in
  SI kilograms (`weight * U_SI`), matching `PlasmaPopulations.Particle(formula, 0)`.
- `NeutralSpecies(scale, formula)` converts the parsed molecular mass directly into code
  units `m / scale.m0`.
- `NeutralSpecies{T}(s)` converts precision (e.g., `NeutralSpecies{Float32}(s)`).
"""
struct NeutralSpecies{T <: Real}
    m::T
    function NeutralSpecies(; m)
        ms = si_mass(m)
        (isfinite(ms) && ms > zero(ms)) ||
            throw(ArgumentError("NeutralSpecies requires finite mass m > 0 (got $m)"))
        return new{typeof(ms)}(ms)
    end
end

NeutralSpecies{T}(args...; kw...) where {T <: Real} =
    NeutralSpecies{T}(NeutralSpecies(args...; kw...))
NeutralSpecies{T}(s::NeutralSpecies) where {T <: Real} = NeutralSpecies(; m = T(s.m))
Base.convert(::Type{NeutralSpecies{T}}, s::NeutralSpecies) where {T <: Real} =
    NeutralSpecies{T}(s)

Base.broadcastable(s::NeutralSpecies) = Ref(s)

function Base.show(io::IO, s::NeutralSpecies)
    return print(io, "NeutralSpecies(m = ", s.m, ")")
end

function _parse_neutral_formula(str::String)
    mass_u = 0.0
    pos = 1
    for m in eachmatch(r"(?:\{(\d+)\})?([A-Z][a-z]?)(\d*)", str)
        m.offset == pos || break
        el = Symbol(m[2])
        el, A = isnothing(m[1]) ? get(_ISOTOPE_SYMBOLS, el, (el, nothing)) :
            (el, parse(Int, m[1]))
        haskey(_ELEMENTS, el) ||
            throw(ArgumentError("NeutralSpecies: unknown element $el in $str"))
        w = isnothing(A) ? _ELEMENTS[el] : get(_ISOTOPES, (el, A)) do
                known = sort!([k[2] for k in keys(_ISOTOPES) if k[1] == el])
                throw(
                    ArgumentError(
                        "NeutralSpecies: no mass for {$A}$el; known: $known; use NeutralSpecies(; m)",
                    )
                )
        end
        count = isempty(m[3]) ? 1 : parse(Int, m[3])
        count > 0 || throw(ArgumentError("NeutralSpecies: invalid stoichiometry in $str"))
        mass_u += count * w
        pos += ncodeunits(m.match)
    end
    (pos == ncodeunits(str) + 1 && mass_u > 0.0) ||
        throw(ArgumentError("NeutralSpecies: cannot parse formula $str"))
    return mass_u
end

function NeutralSpecies(formula::Union{Symbol, AbstractString})
    mass_u = _parse_neutral_formula(String(formula))
    return NeutralSpecies(; m = mass_u * U_SI)
end

function NeutralSpecies(scale::UnitScale{FT}, formula::Union{Symbol, AbstractString}) where {FT}
    mass_si = _parse_neutral_formula(String(formula)) * U_SI
    return NeutralSpecies(; m = FT(mass_si / scale.m0))
end

"""
    atomic_mass(s::NeutralSpecies)

Return the mass of `s` in unified atomic mass units (`s.m / U_SI`), assuming `s.m` is
stored in SI kilograms.
"""
@inline atomic_mass(s::NeutralSpecies) = s.m / U_SI

@inline to_code_units(scale::UnitScale{FT}, s::NeutralSpecies) where {FT} =
    NeutralSpecies(; m = FT(s.m / scale.m0))
