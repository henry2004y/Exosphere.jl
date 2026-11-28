module ExosphereAdaptExt

using Adapt: Adapt
using Exosphere: GriddedNeutralField, NeutralSourceBuffer, SphericalNeutralGrid,
    TabulatedRadialProfile

Adapt.@adapt_structure TabulatedRadialProfile
Adapt.@adapt_structure SphericalNeutralGrid
Adapt.@adapt_structure NeutralSourceBuffer
Adapt.@adapt_structure GriddedNeutralField

end # module ExosphereAdaptExt
