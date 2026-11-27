using BenchmarkTools
using Exosphere
using StaticArrays

const SUITE = BenchmarkGroup()

let
    exp_p = ExponentialProfile(10.0, 1.0, 0.5)
    pl_p = PowerLawProfile(10.0, 1.0, 2.0)
    ch_p = ChamberlainProfile(10.0, 1.0, 2.0)
    ep = exosphere_profile(; n0 = 4.0, r0 = 1.0, H = 0.5, center = (1.0, 0.0, 0.0))
    haser = exosphere_profile(; n0 = 9.0, r0 = 1.0, H = Inf, power = 2.0)

    c1 = NeutralComponentCode(:exponential, 100.0, 0.2, 0.0, 0.01, "O")
    c2 = NeutralComponentCode(:powerlaw, 10.0, 0.0, 2.0, 0.01, "H")
    c3 = NeutralComponentCode(:chamberlain, 50.0, 1.5, 0.0, 0.01, "He")
    comps = [c1, c2, c3]

    sc = ShadowCylinder(; solar_dir = (1.0, 0.0, 0.0), radius = 1.0, half_height = 5.0)
    od_plain = OpticalDepth(;
        cross_sections = [0.05, 0.01, 0.02],
        solar_dir = (1.0, 0.0, 0.0),
        chapman = false,
    )
    od_chapman = OpticalDepth(;
        cross_sections = [0.05, 0.01, 0.02],
        solar_dir = (1.0, 0.0, 0.0),
        chapman = true,
    )

    exo_shadow = build_exosphere_model(;
        planet_radius = 1.0, components = comps, shadow = sc, inner_radius = 1.0,
    )
    exo_optical = build_exosphere_model(;
        planet_radius = 1.0, components = comps, optical_depth = od_plain, inner_radius = 1.0,
    )
    exo_chapman = build_exosphere_model(;
        planet_radius = 1.0, components = comps, optical_depth = od_chapman, inner_radius = 1.0,
    )

    x_day = SVector{3, Float64}(1.5, 0.0, 0.0)
    r_day = hypot(x_day[1], x_day[2], x_day[3])
    x_night = SVector{3, Float64}(-1.5, 0.5, 0.0)
    r_night = hypot(x_night[1], x_night[2], x_night[3])
    x_grazing = SVector{3, Float64}(-1.5, 1.2, 0.0)
    r_grazing = hypot(x_grazing[1], x_grazing[2], x_grazing[3])
    x_term = SVector{3, Float64}(-0.15, 1.5, 0.0)
    r_term = hypot(x_term[1], x_term[2], x_term[3])

    # 1. Neutral density profiles
    SUITE["profiles"] = BenchmarkGroup(["profiles"])
    SUITE["profiles"]["exponential_density"] = @benchmarkable neutral_density($exp_p, 1.5)
    SUITE["profiles"]["exponential_scale_height"] = @benchmarkable local_scale_height($exp_p, 1.5)
    SUITE["profiles"]["powerlaw_density"] = @benchmarkable neutral_density($pl_p, 2.0)
    SUITE["profiles"]["chamberlain_density"] = @benchmarkable neutral_density($ch_p, 2.0)
    SUITE["profiles"]["exosphere_profile_3d"] = @benchmarkable $ep(1.5, 0.2, -0.1)
    SUITE["profiles"]["haser_profile_3d"] = @benchmarkable $haser(3.0, 0.5, -0.2)
    SUITE["profiles"]["exo_density_3comp"] = @benchmarkable exo_density($exo_chapman, 1.5)

    # 2. Shadow & EUV optical-depth / Chapman attenuation
    SUITE["attenuation"] = BenchmarkGroup(["attenuation"])
    SUITE["attenuation"]["shadow_factor_day"] = @benchmarkable shadow_factor($sc, $x_day)
    SUITE["attenuation"]["shadow_factor_night"] = @benchmarkable shadow_factor($sc, $x_night)
    SUITE["attenuation"]["chapman_day"] = @benchmarkable chapman_function(20.0, 0.7)
    SUITE["attenuation"]["chapman_terminator"] = @benchmarkable chapman_function(20.0, -0.1)
    SUITE["attenuation"]["photo_attenuation_shadow"] = @benchmarkable photo_attenuation(
        $exo_shadow, $x_night, $r_night,
    )
    SUITE["attenuation"]["photo_attenuation_optical_day"] = @benchmarkable photo_attenuation(
        $exo_optical, $x_day, $r_day,
    )
    SUITE["attenuation"]["photo_attenuation_optical_grazing"] = @benchmarkable photo_attenuation(
        $exo_optical, $x_grazing, $r_grazing,
    )
    SUITE["attenuation"]["photo_attenuation_chapman_day"] = @benchmarkable photo_attenuation(
        $exo_chapman, $x_day, $r_day,
    )
    SUITE["attenuation"]["photo_attenuation_chapman_terminator"] = @benchmarkable photo_attenuation(
        $exo_chapman, $x_term, $r_term,
    )

    # 3. Ion-neutral source rates
    pts = [
        SVector{3, Float64}(
            1.4 * cos(0.00628 * i),
            1.4 * sin(0.00628 * i),
            0.2 * sin(0.01256 * i),
        ) for i in 1:1000
    ]
    SUITE["sources"] = BenchmarkGroup(["sources"])
    SUITE["sources"]["photoionization_rate"] = @benchmarkable photoionization_rate(1.0e-6, 100.0, 0.85)
    SUITE["sources"]["charge_exchange_rate"] = @benchmarkable charge_exchange_rate(2.0e-19, 5.0, 100.0, 4.0e5)
    SUITE["sources"]["batch_photoionization_1000"] = @benchmarkable begin
        s = 0.0
        @inbounds for x in $pts
            r = hypot(x[1], x[2], x[3])
            n = exo_density($exo_chapman, r)
            att = photo_attenuation($exo_chapman, x, r)
            s += photoionization_rate(1.0e-6, n, att)
        end
        s
    end

    # 4. Adaptive octree dual-cell box quadrature
    sumN = zeros(3)
    sumNA = zeros(3)
    SUITE["quadrature"] = BenchmarkGroup(["quadrature"])
    SUITE["quadrature"]["box_exterior_3d"] = @benchmarkable integrate_exosphere_box!(
        $sumN, $sumNA, $exo_chapman, $(SVector{3, Float64}(1.5, 0.0, 0.0)),
        $(SVector{3, Float64}(0.2, 0.2, 0.2)), 0, 1.0,
    ) setup = (fill!($sumN, 0.0); fill!($sumNA, 0.0))
    SUITE["quadrature"]["box_straddling_3d"] = @benchmarkable integrate_exosphere_box!(
        $sumN, $sumNA, $exo_chapman, $(SVector{3, Float64}(1.05, 0.1, 0.0)),
        $(SVector{3, Float64}(0.4, 0.4, 0.4)), 0, 1.0,
    ) setup = (fill!($sumN, 0.0); fill!($sumNA, 0.0))
    SUITE["quadrature"]["box_straddling_2d"] = @benchmarkable integrate_exosphere_box!(
        $sumN, $sumNA, $exo_chapman, $(SVector{3, Float64}(1.05, 0.1, 0.0)),
        $(SVector{3, Float64}(0.4, 0.4, 0.0)), 0, 1.0,
    ) setup = (fill!($sumN, 0.0); fill!($sumNA, 0.0))
end
