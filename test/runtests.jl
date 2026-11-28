using Adapt: Adapt
using Aqua: Aqua
using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports
using Exosphere
using ForwardDiff: ForwardDiff
using LinearAlgebra: dot, norm
import PlasmaPopulations as PP
using Random: MersenneTwister
using StaticArrays: SMatrix, SVector
using Statistics: cov, mean
using Test
using Unitful: @u_str, uconvert, ustrip

function _alloc_neutral_density(p, r)
    neutral_density(p, r)
    return @allocated neutral_density(p, r)
end

function _alloc_exosphere_profile(ep, x, y, z)
    ep(x, y, z)
    return @allocated ep(x, y, z)
end

function _alloc_shadow_factor(sc, x)
    shadow_factor(sc, x)
    return @allocated shadow_factor(sc, x)
end

function _alloc_exo_component_density(exo, r, c)
    exo_component_density(exo, r, c)
    return @allocated exo_component_density(exo, r, c)
end

function _alloc_exo_density(exo, r)
    exo_density(exo, r)
    return @allocated exo_density(exo, r)
end

function _alloc_exo_scale_height(exo, r, c)
    exo_scale_height(exo, r, c)
    return @allocated exo_scale_height(exo, r, c)
end

function _alloc_chapman(xp, cos_sza)
    chapman_function(xp, cos_sza)
    return @allocated chapman_function(xp, cos_sza)
end

function _alloc_photo_attenuation(exo, x, r)
    photo_attenuation(exo, x, r)
    return @allocated photo_attenuation(exo, x, r)
end

function _alloc_photo_attenuation_2arg(model, x)
    photo_attenuation(model, x)
    return @allocated photo_attenuation(model, x)
end

function _alloc_integrate_box!(sumN, sumNA, exo, c, h, depth, rcut)
    integrate_exosphere_box!(sumN, sumNA, exo, c, h, depth, rcut)
    return @allocated integrate_exosphere_box!(sumN, sumNA, exo, c, h, depth, rcut)
end

function _alloc_photoionization_rate(nu, n, att)
    photoionization_rate(nu, n, att)
    return @allocated photoionization_rate(nu, n, att)
end

function _alloc_charge_exchange_rate(sig, ni, nn, v)
    charge_exchange_rate(sig, ni, nn, v)
    return @allocated charge_exchange_rate(sig, ni, nn, v)
end

function _alloc_neutral_state(model, x, s)
    neutral_state(model, x; species = s)
    return @allocated neutral_state(model, x; species = s)
end

function _alloc_compute_exchange(ch, plasma, neutral, m_n, att)
    compute_exchange(ch, plasma, neutral, m_n; atten = att)
    return @allocated compute_exchange(ch, plasma, neutral, m_n; atten = att)
end

function _alloc_couple_point!(model, loc, ch, plasma, s)
    couple_point!(model, loc, ch, plasma; species = s)
    return @allocated couple_point!(model, loc, ch, plasma; species = s)
end

function _alloc_vdf_rand(rng, vdf)
    rand(rng, vdf)
    return @allocated rand(rng, vdf)
end

function _alloc_vdf_pdf(vdf, v)
    pdf(vdf, v)
    return @allocated pdf(vdf, v)
end

function _alloc_deposit_source!(model, loc, src, s)
    deposit_neutral_source!(model, loc, src, s)
    return @allocated deposit_neutral_source!(model, loc, src, s)
end

struct _TestFloat32Adaptor end
Adapt.adapt_storage(::_TestFloat32Adaptor, x::Array{Float64}) = Float32.(x)
Adapt.adapt_storage(::_TestFloat32Adaptor, x::Array{SVector{3, Float64}}) =
    SVector{3, Float32}.(x)

@testset "Exosphere.jl" begin
    @testset "Package Hygiene, Aqua.jl & API Docstrings" begin
        Aqua.test_all(
            Exosphere;
            ambiguities = true,
            piracies = true,
            stale_deps = true,
            deps_compat = (check_extras = (ignore = [:Test],),),
        )
        @test check_no_implicit_imports(Exosphere; skip = (Base, Core)) === nothing
        @test check_no_stale_explicit_imports(Exosphere) === nothing

        docs_meta = Base.Docs.meta(Exosphere)
        public_and_exported = [
            s for s in names(Exosphere; all = true) if
                s !== :Exosphere && Base.ispublic(Exosphere, s)
        ]
        missing_docstrings = [
            s for s in public_and_exported if
                !haskey(docs_meta, Base.Docs.Binding(Exosphere, s))
        ]
        @test isempty(missing_docstrings)
    end

    @testset "Validation & Edge Cases (v0.1.0 Backwards Compatibility)" begin
        # Profile constructor validation
        @test_throws ArgumentError ExponentialProfile(-1.0, 1.0, 0.5)
        @test_throws ArgumentError ExponentialProfile(Inf, 1.0, 0.5)
        @test_throws ArgumentError ExponentialProfile(1.0, 0.0, 0.5)
        @test_throws ArgumentError ExponentialProfile(1.0, 1.0, 0.0)

        @test_throws ArgumentError PowerLawProfile(-1.0, 1.0, 2.0)
        @test_throws ArgumentError PowerLawProfile(1.0, 0.0, 2.0)
        @test_throws ArgumentError PowerLawProfile(1.0, 1.0, -0.5)

        @test_throws ArgumentError ChamberlainProfile(-1.0, 1.0, 2.0)
        @test_throws ArgumentError ChamberlainProfile(1.0, 0.0, 2.0)
        @test_throws ArgumentError ChamberlainProfile(1.0, 1.0, Inf)

        @test_throws ArgumentError exosphere_profile(; n0 = -1.0)
        @test_throws ArgumentError exosphere_profile(; r0 = 0.0)
        @test_throws ArgumentError exosphere_profile(; H = 0.0)
        @test_throws ArgumentError exosphere_profile(; power = -1.0)
        @test_throws ArgumentError exosphere_profile(; center = (NaN, 0.0, 0.0))

        # Clamping at r <= r0 and H = Inf exponential limit
        exp_inf = ExponentialProfile(5.0, 1.0, Inf)
        @test neutral_density(exp_inf, 0.5) == 5.0
        @test neutral_density(exp_inf, 3.0) == 5.0
        @test local_scale_height(exp_inf, 2.0) == 0.0

        pl = PowerLawProfile(7.0, 1.5, 2.0)
        @test neutral_density(pl, 0.8) == 7.0
        ch = ChamberlainProfile(7.0, 1.5, 2.0)
        @test neutral_density(ch, 0.8) == 7.0
        ep = exosphere_profile(; n0 = 4.0, r0 = 1.5, H = 0.5, power = 2.0)
        @test ep(1.0) == 4.0
        @test local_scale_height(ep, 2.0) ≈ inv(2.0 / 2.0 + inv(0.5))

        # NeutralComponentCode, ShadowCylinder, OpticalDepth & build_exosphere_model validation
        @test_throws ArgumentError NeutralComponentCode(:unknown, 1.0, 0.5, 0.0, 1.0)
        @test_throws ArgumentError NeutralComponentCode(:exponential, 1.0, 0.0, 0.0, 1.0, "O")
        @test_throws ArgumentError NeutralComponentCode(:chamberlain, 1.0, 0.0, 0.0, 1.0, "He")
        @test_throws ArgumentError NeutralComponentCode(:powerlaw, 1.0, 0.0, 2.0, 0.0, "H")

        @test_throws ArgumentError ShadowCylinder(; solar_dir = (0.0, 0.0, 0.0))
        @test_throws ArgumentError ShadowCylinder(; solar_dir = [1.0, 0.0])

        @test_throws ArgumentError OpticalDepth(; cross_sections = [-0.1])
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.0, 0.0])
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.1], scale_heights = [1.0, 2.0])
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.1], solar_dir = (0.0, 0.0, 0.0))
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.1], cos_sza_floor = 0.0)
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.1], tau_floor = -1.0)
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.1], tau_cutoff = 0.0)
        @test_throws ArgumentError OpticalDepth(; cross_sections = [0.1], min_production = 1.5)

        c1 = NeutralComponentCode(:exponential, 100.0, 0.2, 0.0, 0.01, "O")
        c_zero_k = NeutralComponentCode(:powerlaw, 10.0, 0.0, 0.0, 0.01, "Flat")
        sc = ShadowCylinder(; solar_dir = (1.0, 0.0, 0.0), radius = 1.0)
        od = OpticalDepth(; cross_sections = [0.05], chapman = true)
        @test_throws ArgumentError build_exosphere_model(;
            planet_radius = 1.0, components = [c1], shadow = sc, optical_depth = od,
        )
        @test_throws ArgumentError build_exosphere_model(;
            planet_radius = 1.0, components = NeutralComponentCode[],
        )
        @test_throws ArgumentError build_exosphere_model(;
            planet_radius = 1.0, components = [c1, c_zero_k], optical_depth = od,
        )
        od_zero_h = OpticalDepth(; cross_sections = [0.05])
        @test_throws ArgumentError build_exosphere_model(;
            planet_radius = 1.0, components = [c_zero_k], optical_depth = od_zero_h,
        )

        # Unattenuated ExosphereModel & BoundsError
        exo_plain = build_exosphere_model(; planet_radius = 1.0, components = [c1])
        @test photo_attenuation(exo_plain, SVector(-2.0, 0.0, 0.0), 2.0) == 1.0
        @test exo_plain.shadow === nothing
        @test exo_plain.optical === nothing
        @test_throws BoundsError exo_scale_height(exo_plain, 1.5, 0)
        @test_throws BoundsError exo_scale_height(exo_plain, 1.5, 2)

        # Explicit scale_heights & chapman_component override
        od_custom = OpticalDepth(;
            cross_sections = [0.05, 0.01],
            scale_heights = [0.4, nothing],
            chapman = true,
            chapman_component = 2,
        )
        c2 = NeutralComponentCode(:powerlaw, 10.0, 0.0, 2.0, 0.01, "H")
        exo_custom = build_exosphere_model(;
            planet_radius = 1.0, components = [c1, c2], optical_depth = od_custom,
        )
        @test exo_scale_height(exo_custom, 2.0, 1) ≈ 0.4
        @test exo_scale_height(exo_custom, 2.0, 2) ≈ 1.0
        @test exo_custom.optical.chapman_component == 2

        # Non-Chapman grazing shadow, tau_cutoff, and chapman_tau_max branches
        od_cut = OpticalDepth(;
            cross_sections = [5.0],
            chapman = false,
            min_production = 1.0e-5,
            tau_cutoff = 2.0,
        )
        exo_cut = build_exosphere_model(;
            planet_radius = 1.0, components = [c1], optical_depth = od_cut,
        )
        @test photo_attenuation(exo_cut, SVector(-1.5, 0.5, 0.0), hypot(-1.5, 0.5)) == 1.0e-5
        @test photo_attenuation(exo_cut, SVector(1.0, 0.0, 0.0), 1.0) == 1.0e-5
        @test 1.0e-5 < photo_attenuation(exo_cut, SVector(-1.5, 2.3, 0.0), hypot(-1.5, 2.3)) < 1.0

        od_tmax = OpticalDepth(;
            cross_sections = [1.0],
            chapman = true,
            min_production = 1.0e-4,
            chapman_tau_max = 0.5,
        )
        exo_tmax = build_exosphere_model(;
            planet_radius = 1.0, components = [c1], optical_depth = od_tmax,
        )
        @test photo_attenuation(exo_tmax, SVector(1.0, 0.0, 0.0), 1.0) ≈ 1.0e-4

        # Chapman function branches (y < 8, 8 <= y < 100, y >= 100 for dayside & nightside)
        @test chapman_function(20.0, 0.7) > 1.0
        @test 0.0 < chapman_function(200.0, 1.0) < 2.0
        @test chapman_function(25000.0, 1.0) == 0.0
        @test chapman_function(20.0, -0.1) > 1.0
        @test chapman_function(200.0, -0.9) > 0.0
        @test chapman_function(25000.0, -0.95) == 0.0

        # Box quadrature: fully interior, 2D collapsed axis, and analytic sinh(b)/b accuracy
        sumN = zeros(1)
        sumNA = zeros(1)
        @test integrate_exosphere_box!(
            sumN, sumNA, exo_plain, SVector(0.2, 0.0, 0.0), SVector(0.2, 0.2, 0.2), 0, 1.0,
        ) == 0.0
        @test iszero(sumN) && iszero(sumNA)

        fill!(sumN, 0.0)
        fill!(sumNA, 0.0)
        area_straddle = integrate_exosphere_box!(
            sumN, sumNA, exo_plain, SVector(1.0, 0.0, 0.0), SVector(0.4, 0.4, 0.0), 0, 1.0,
        )
        @test 0.0 < area_straddle < 0.16
        @test sumN[1] > 0.0

        c_wide = NeutralComponentCode(:exponential, 10.0, 1.0, 0.0, 0.01, "O")
        exo_wide = build_exosphere_model(; planet_radius = 1.0, components = [c_wide])
        fill!(sumN, 0.0)
        fill!(sumNA, 0.0)
        dx, dy, dz = 0.2, 1.0e-5, 1.0e-5
        x0 = 2.0
        vol_exact = integrate_exosphere_box!(
            sumN, sumNA, exo_wide, SVector(x0, 0.0, 0.0), SVector(dx, dy, dz), 0, 1.0,
        )
        @test vol_exact ≈ dx * dy * dz
        b = 0.5 * dx / 1.0
        n_exact = 10.0 * exp(-(x0 - 1.0) / 1.0) * (sinh(b) / b) * dx * dy * dz
        @test isapprox(sumN[1], n_exact; rtol = 1.0e-9)
    end

    @testset "Neutral Profiles (1D/2D/3D & Allocation-Free)" begin
        exp_p = ExponentialProfile(10.0, 1.0, 0.5)
        @test neutral_density(exp_p, 1.0) ≈ 10.0
        @test neutral_density(exp_p, 1.5) ≈ 10.0 * exp(-1.0)
        @test local_scale_height(exp_p, 2.0) ≈ 0.5
        @test _alloc_neutral_density(exp_p, 1.5) == 0

        pl_p = PowerLawProfile(10.0, 1.0, 2.0)
        @test neutral_density(pl_p, 2.0) ≈ 2.5
        @test local_scale_height(pl_p, 2.0) ≈ 1.0
        @test _alloc_neutral_density(pl_p, 2.0) == 0

        ch_p = ChamberlainProfile(10.0, 1.0, 2.0)
        @test neutral_density(ch_p, 2.0) ≈ 10.0 * exp(-1.0)
        @test local_scale_height(ch_p, 2.0) ≈ 2.0
        @test _alloc_neutral_density(ch_p, 2.0) == 0

        ep = exosphere_profile(; n0 = 4.0, r0 = 1.0, H = 0.5, center = (1.0, 0.0, 0.0))
        @test ep(1.0) ≈ 4.0
        @test ep(2.5, 0.0) ≈ 4.0 * exp(-1.0)
        @test ep(1.0, 0.3, 0.4) ≈ 4.0
        @test _alloc_exosphere_profile(ep, 1.5, 0.2, -0.1) == 0

        haser = exosphere_profile(; n0 = 9.0, r0 = 1.0, H = Inf, power = 2.0)
        @test haser(3.0, 0.0, 0.0) ≈ 1.0
    end

    @testset "ShadowCylinder, OpticalDepth, Chapman & Box Quadrature" begin
        sc = ShadowCylinder(; solar_dir = (1.0, 0.0, 0.0), radius = 1.0, half_height = 5.0)
        @test shadow_factor(sc, 2.0, 0.0, 0.0) == 1.0
        @test shadow_factor(sc, -2.0, 0.5, 0.0) == 0.0
        @test shadow_factor(sc, -2.0, 1.5, 0.0) == 1.0
        @test shadow_factor(sc, -6.0, 0.5, 0.0) == 1.0

        c1 = NeutralComponentCode(:exponential, 100.0, 0.2, 0.0, 0.01, "O")
        c2 = NeutralComponentCode(:powerlaw, 10.0, 0.0, 2.0, 0.01, "H")
        od = OpticalDepth(; cross_sections = [0.05, 0.0], solar_dir = (1.0, 0.0, 0.0), chapman = true)
        exo = build_exosphere_model(;
            planet_radius = 1.0, components = [c1, c2], optical_depth = od, inner_radius = 1.0,
        )
        @test n_components(exo) == 2
        @test exo_component_density(exo, 0.9, 1) == 0.0
        @test exo_density(exo, 1.0) ≈ 110.0

        @test chapman_function(20.0, 1.0) ≈ 1.0 atol = 0.05
        att_day = photo_attenuation(exo, SVector(1.5, 0.0, 0.0), 1.5)
        att_night = photo_attenuation(exo, SVector(-1.5, 0.0, 0.0), 1.5)
        @test 0.0 < att_night < att_day <= 1.0

        sumN = zeros(2)
        sumNA = zeros(2)
        vol = integrate_exosphere_box!(sumN, sumNA, exo, SVector(1.5, 0.0, 0.0), SVector(0.2, 0.2, 0.2), 0, 1.0)
        @test vol ≈ 0.008
        @test sumN[1] > 0.0 && sumNA[1] > 0.0 && sumNA[1] <= sumN[1]

        @test photoionization_rate(2.0, 3.0, 0.5) ≈ 3.0
        @test charge_exchange_rate(0.1, 2.0, 3.0, 4.0) ≈ 2.4
    end

    @testset "NeutralSpecies & Neutral VDFs (PlasmaPopulations.jl Style)" begin
        sp_o = NeutralSpecies("O")
        sp_co2 = NeutralSpecies("CO2")
        sp_o18 = NeutralSpecies("{18}O")
        sp_hdo = NeutralSpecies("HDO")
        @test isbits(sp_o)
        @test atomic_mass(sp_o) ≈ 15.999
        @test atomic_mass(sp_co2) ≈ 12.011 + 2 * 15.999
        @test atomic_mass(sp_o18) ≈ 17.99915961286
        @test atomic_mass(sp_hdo) ≈ 1.008 + 2.014101777844 + 15.999
        @test NeutralSpecies{Float32}(sp_o) isa NeutralSpecies{Float32}
        @test_throws ArgumentError NeutralSpecies(; m = -1.0)
        @test_throws ArgumentError NeutralSpecies("Zz2")
        @test_throws ArgumentError NeutralSpecies("{99}O")

        # Code-unit scaling of species
        scale = UnitScale(; length = 3390.0e3, density = 1.0e6, velocity = 1.0e5, mass = Exosphere.MP_SI)
        sp_o_code = NeutralSpecies(scale, "O")
        @test sp_o_code.m ≈ (15.999 * Exosphere.U_SI) / Exosphere.MP_SI
        @test to_code_units(scale, sp_o).m ≈ sp_o_code.m

        # ColdNeutral
        sp_c = NeutralSpecies(; m = 16.0)
        cold = ColdNeutral(sp_c; u = (1.0, -2.0, 0.5))
        @test isbits(cold)
        @test mean(cold) == SVector(1.0, -2.0, 0.5)
        @test cov(cold) == zero(SMatrix{3, 3, Float64})
        @test thermal_speed(cold) == 0.0
        @test vdf_temperature(sp_c, cold) == 0.0
        rng = MersenneTwister(42)
        @test rand(rng, cold) == SVector(1.0, -2.0, 0.5)
        @test _alloc_vdf_rand(rng, cold) == 0

        # Maxwellian
        maxw = Maxwellian(sp_c; T = 0.08, u = (1.0, 0.5, -0.25))
        @test isbits(maxw)
        @test thermal_speed(maxw) ≈ sqrt(2 * 0.08 / 16.0)
        @test vdf_temperature(sp_c, maxw) ≈ 0.08
        @test temperature_perp_para(sp_c, maxw) == (vdf_temperature(sp_c, maxw), vdf_temperature(sp_c, maxw))
        @test mean(maxw) == SVector(1.0, 0.5, -0.25)
        @test cov(maxw)[1, 1] ≈ 0.5 * maxw.vth^2
        @test pdf(maxw, maxw.u) ≈ inv((π * maxw.vth^2)^1.5)
        @test pdf(maxw)(maxw.u) == pdf(maxw, maxw.u)
        @test _alloc_vdf_rand(rng, maxw) == 0
        @test _alloc_vdf_pdf(maxw, SVector(1.1, 0.5, -0.2)) == 0
        # Monte Carlo sample moments match mean & cov
        samples = rand(rng, maxw, 40_000)
        m_est = mean(samples)
        @test isapprox(m_est, maxw.u; atol = 5.0e-3)
        v_est = mean(s -> sum(abs2, s - maxw.u), samples)
        @test isapprox(v_est, 1.5 * maxw.vth^2; rtol = 0.03)

        # BiMaxwellianNeutral
        bimax = BiMaxwellianNeutral(
            sp_c; T_perp = 0.04, T_para = 0.16, u = (0.2, 0.0, 0.0), axis = (1.0, 0.0, 0.0),
        )
        @test isbits(bimax)
        tp, tl = temperature_perp_para(sp_c, bimax)
        @test tp ≈ 0.04 && tl ≈ 0.16
        @test vdf_temperature(sp_c, bimax) ≈ (2 * 0.04 + 0.16) / 3
        @test cov(bimax)[1, 1] ≈ 0.5 * bimax.vth_para^2
        @test cov(bimax)[2, 2] ≈ 0.5 * bimax.vth_perp^2
        @test _alloc_vdf_rand(rng, bimax) == 0
        @test _alloc_vdf_pdf(bimax, SVector(0.3, 0.1, 0.0)) == 0
        bm_samples = rand(rng, bimax, 40_000)
        var_x = mean(s -> (s[1] - 0.2)^2, bm_samples)
        var_y = mean(s -> s[2]^2, bm_samples)
        @test isapprox(var_x, 0.5 * bimax.vth_para^2; rtol = 0.05)
        @test isapprox(var_y, 0.5 * bimax.vth_perp^2; rtol = 0.05)
    end

    @testset "3D Asymmetric, Composite, Tabulated, Temporal Profiles & Kinematics" begin
        # DayNightProfile
        base_exp = ExponentialProfile(100.0, 1.0, 0.2)
        dn = DayNightProfile(base_exp; solar_dir = (1.0, 0.0, 0.0), asymmetry = 0.5, min_factor = 0.2)
        @test isbits(dn)
        @test neutral_density(dn, SVector(1.5, 0.0, 0.0)) ≈ 1.5 * neutral_density(base_exp, 1.5)
        @test neutral_density(dn, SVector(-1.5, 0.0, 0.0)) ≈ 0.5 * neutral_density(base_exp, 1.5)
        @test neutral_density(dn, SVector(0.0, 1.5, 0.0)) ≈ neutral_density(base_exp, 1.5)
        @test local_scale_height(dn, SVector(1.5, 0.0, 0.0)) ≈ 0.2

        # PlumeProfile
        plume = PlumeProfile(;
            n0 = 50.0, origin = (0.0, 0.0, -1.0), axis = (0.0, 0.0, -1.0),
            r0 = 0.1, H = 0.5, k = 2.0, opening_angle = 0.3,
        )
        @test isbits(plume)
        @test neutral_density(plume, SVector(0.0, 0.0, -1.2)) > neutral_density(plume, SVector(0.1, 0.0, -1.2))
        @test neutral_density(plume, SVector(0.0, 0.0, 1.2)) == 0.0

        # CompositeProfile (cold + hot corona)
        comp = CompositeProfile(ExponentialProfile(100.0, 1.0, 0.1), ExponentialProfile(5.0, 1.0, 1.0))
        @test isbits(comp)
        @test neutral_density(comp, 1.2) ≈ 100.0 * exp(-2.0) + 5.0 * exp(-0.2)
        n1 = 100.0 * exp(-2.0)
        n2 = 5.0 * exp(-0.2)
        @test local_scale_height(comp, 1.2) ≈ (n1 + n2) / (n1 / 0.1 + n2 / 1.0)

        # TabulatedRadialProfile (SVector isbits GPU-ready)
        rs_s = SVector(1.0, 1.5, 2.0, 3.0)
        ns_s = SVector(100.0, 100.0 * exp(-1.0), 100.0 * exp(-2.0), 100.0 * exp(-4.0))
        tab = TabulatedRadialProfile(rs_s, ns_s; log_interp = true)
        @test isbits(tab)
        @test neutral_density(tab, 1.25) ≈ 100.0 * exp(-0.5)
        @test local_scale_height(tab, 1.25) ≈ 0.5
        @test neutral_density(tab, 4.0) == 0.0

        # TimeDependentProfile
        td = TimeDependentProfile((x, t) -> (t === nothing ? 1.0 : 1.0 + t) * exp(-norm(x)))
        @test neutral_density(td, SVector(1.0, 0.0, 0.0); t = 2.0) ≈ 3.0 * exp(-1.0)

        # Velocity & Temperature models
        @test neutral_velocity(ZeroVelocity(), SVector(1.0, 2.0, 3.0)) == SVector(0.0, 0.0, 0.0)
        corot = CorotationVelocity((0.0, 0.0, 2.0))
        @test neutral_velocity(corot, SVector(1.0, 0.0, 0.0)) == SVector(0.0, 2.0, 0.0)
        outf = RadialOutflowVelocity(3.0)
        @test neutral_velocity(outf, SVector(0.0, 2.0, 0.0)) ≈ SVector(0.0, 3.0, 0.0)
        @test neutral_velocity(outf, SVector(0.0, 0.0, 0.0)) == SVector(0.0, 0.0, 0.0)
        dnt = DayNightTemperature(; T_day = 0.05, T_night = 0.01)
        @test neutral_temperature(dnt, SVector(2.0, 0.0, 0.0)) ≈ 0.05
        @test neutral_temperature(dnt, SVector(-2.0, 0.0, 0.0)) ≈ 0.01
        @test neutral_temperature(dnt, SVector(0.0, 2.0, 0.0)) ≈ 0.03
    end

    @testset "GPU/isbits Tuple ExosphereModel & RayOpticalDepth" begin
        sp_o = NeutralSpecies(; m = 16.0)
        sp_h = NeutralSpecies(; m = 1.0)
        pop_o = NeutralPopulation(
            sp_o,
            ExponentialProfile(100.0, 1.0, 0.2);
            velocity = CorotationVelocity((0.0, 0.0, 0.01)),
            temperature = ConstantTemperature(0.01),
        )
        pop_h = NeutralPopulation(
            sp_h,
            PowerLawProfile(10.0, 1.0, 2.0),
            Maxwellian(sp_h; T = 0.02),
        )
        od_s = OpticalDepth(; cross_sections = (0.05, 0.01), chapman = true)
        exo_s = ExosphereModel(pop_o, pop_h; planet_radius = 1.0, optical_depth = od_s)
        @test isbits(exo_s)
        @test coupling_mode(exo_s) === OneWayCoupling()
        @test n_components(exo_s) == 2
        @test species_mass(exo_s, 1) == 16.0
        @test species_mass(exo_s, 2) == 1.0

        # Exact parity with legacy NeutralComponentCode ExosphereModel
        c_o = NeutralComponentCode(:exponential, 100.0, 0.2, 0.0, 0.01, "O")
        c_h = NeutralComponentCode(:powerlaw, 10.0, 0.0, 2.0, 0.02, "H")
        od_v = OpticalDepth(; cross_sections = [0.05, 0.01], chapman = true)
        exo_v = build_exosphere_model(; planet_radius = 1.0, components = [c_o, c_h], optical_depth = od_v)
        x_q = SVector(1.4, 0.3, 0.0)
        r_q = norm(x_q)
        @test exo_density(exo_s, r_q) ≈ exo_density(exo_v, r_q)
        @test exo_component_density(exo_s, r_q, 1) ≈ exo_component_density(exo_v, r_q, 1)
        @test exo_component_density(exo_s, r_q, 2) ≈ exo_component_density(exo_v, r_q, 2)
        @test photo_attenuation(exo_s, x_q, r_q) ≈ photo_attenuation(exo_v, x_q, r_q)

        # Float32 tuple ExosphereModel isbits & inferred
        pop_o32 = NeutralPopulation(
            NeutralSpecies(; m = 16.0f0),
            ExponentialProfile(100.0f0, 1.0f0, 0.2f0; FT = Float32);
            temperature = ConstantTemperature(0.01f0),
        )
        od32 = OpticalDepth(; cross_sections = (0.05f0,), chapman = true, FT = Float32)
        exo32 = ExosphereModel(pop_o32; planet_radius = 1.0f0, optical_depth = od32, FT = Float32)
        @test isbits(exo32)
        @test (@inferred exo_density(exo32, 1.5f0)) isa Float32
        @test (@inferred photo_attenuation(exo32, SVector(1.5f0, 0.0f0, 0.0f0), 1.5f0)) isa Float32
        sumN32 = zeros(Float32, 1)
        sumNA32 = zeros(Float32, 1)
        @test (
            @inferred integrate_exosphere_box!(
                sumN32, sumNA32, exo32, SVector(1.5f0, 0.0f0, 0.0f0),
                SVector(0.2f0, 0.2f0, 0.2f0), 0, 1.0f0,
            )
        ) isa Float32

        # RayOpticalDepth vs analytic vertical optical depth at subsolar point
        rod = RayOpticalDepth(;
            cross_sections = (0.05, 0.0), solar_dir = (1.0, 0.0, 0.0), s_max = 6.0,
            planet_radius = 1.0, n_quad = 16,
        )
        exo_ray = ExosphereModel(pop_o, pop_h; planet_radius = 1.0, attenuation = rod)
        @test isbits(exo_ray)
        # Analytic integral along +x from x = 1.4 to ∞ of 100 * exp(-(x - 1)/0.2) * 0.05 dx = 100 * exp(-2) * 0.05 * 0.2
        tau_exact = 100.0 * exp(-(1.4 - 1.0) / 0.2) * 0.05 * 0.2
        att_ray_day = photo_attenuation(exo_ray, SVector(1.4, 0.0, 0.0))
        @test isapprox(att_ray_day, exp(-tau_exact); rtol = 1.0e-4)
        # Solid-body nightside shadow
        @test photo_attenuation(exo_ray, SVector(-1.5, 0.5, 0.0)) == rod.min_production
        @test _alloc_photo_attenuation_2arg(exo_ray, SVector(1.4, 0.0, 0.0)) == 0
    end

    @testset "Conservative Charge-Neutral Interactions & Kinetic Kernels" begin
        plasma = PlasmaState(5.0, SVector(-2.0, 0.5, -0.3), 0.15, 16.0)
        neutral = NeutralState(100.0, SVector(0.1, 0.4, 0.0), 0.01)
        m_n = 16.0

        # 1. Photoionization conservation
        ch_ph = Photoionization(1.0e-5; E_excess = 0.05)
        ex_ph = compute_exchange(ch_ph, plasma, neutral, m_n; atten = 0.8)
        R_ph = photoionization_rate(1.0e-5, neutral.n, 0.8)
        @test ex_ph.plasma.dn ≈ R_ph
        @test ex_ph.plasma.dn + ex_ph.neutral.dn == 0.0
        @test norm(ex_ph.plasma.dmom + ex_ph.neutral.dmom) <= 1.0e-15
        @test isapprox(ex_ph.plasma.dE + ex_ph.neutral.dE, R_ph * 0.05; rtol = 1.0e-14)

        # 2. Electron-impact ionization conservation (callable rate)
        ch_imp = ElectronImpactIonization(Te -> 2.0e-4 * sqrt(Te); E_ion = 0.1)
        ex_imp = compute_exchange(ch_imp, plasma, neutral, m_n; n_e = 6.0, T_e = 0.25)
        R_imp = (2.0e-4 * 0.5) * 6.0 * neutral.n
        @test ex_imp.plasma.dn ≈ R_imp
        @test ex_imp.plasma.dn + ex_imp.neutral.dn == 0.0
        @test norm(ex_imp.plasma.dmom + ex_imp.neutral.dmom) <= 1.0e-15
        @test isapprox(ex_imp.plasma.dE + ex_imp.neutral.dE, -R_imp * 0.1; rtol = 1.0e-14)

        # 3. Resonant charge exchange exact conservation
        ch_cx = ResonantChargeExchange(v -> 1.0e-3 * (1.0 + 0.01 * v))
        ex_cx = compute_exchange(ch_cx, plasma, neutral, m_n)
        @test ex_cx.plasma.dn == 0.0 && ex_cx.neutral.dn == 0.0
        @test ex_cx.plasma.dmom == -ex_cx.neutral.dmom
        @test ex_cx.plasma.dE == -ex_cx.neutral.dE

        # 4. Non-resonant charge exchange 4-species conservation (H+ + O -> H + O+)
        ion_h = PlasmaState(4.0, SVector(-3.0, 0.2, 0.0), 0.2, 1.0)
        ch_nrcx = NonResonantChargeExchange(2.0e-3; E_defect = 0.02)
        nr = compute_nonresonant_exchange(ch_nrcx, ion_h, neutral, 16.0)
        @test nr.ion_loss.dn + nr.ion_gain.dn == 0.0
        @test nr.neutral_loss.dn + nr.neutral_gain.dn == 0.0
        @test nr.ion_loss.dn + nr.neutral_gain.dn == 0.0
        @test norm(nr.ion_loss.dmom + nr.ion_gain.dmom + nr.neutral_loss.dmom + nr.neutral_gain.dmom) <= 1.0e-12
        R_nr = -nr.ion_loss.dn
        @test isapprox(
            nr.ion_loss.dE + nr.ion_gain.dE + nr.neutral_loss.dE + nr.neutral_gain.dE,
            R_nr * 0.02;
            rtol = 1.0e-12,
        )

        # 5. Elastic collision exact conservation & equal-mass resonant CX limit
        ch_el_neq = ElasticCollision(2.0e-3; is_cross_section = true)
        ex_el_neq = compute_exchange(ch_el_neq, ion_h, neutral, 16.0)
        @test ex_el_neq.plasma.dn == 0.0 && ex_el_neq.neutral.dn == 0.0
        @test ex_el_neq.plasma.dmom == -ex_el_neq.neutral.dmom
        @test ex_el_neq.plasma.dE == -ex_el_neq.neutral.dE
        # Equal-mass σ_mt = 2 * σ_cx reduces identically to resonant CX!
        ex_cx_const = compute_exchange(ResonantChargeExchange(1.0e-3), plasma, neutral, 16.0)
        ex_el_eq = compute_exchange(ElasticCollision(2.0e-3; is_cross_section = true), plasma, neutral, 16.0)
        @test isapprox(ex_el_eq.plasma.dmom, ex_cx_const.plasma.dmom; rtol = 1.0e-14)
        @test isapprox(ex_el_eq.plasma.dE, ex_cx_const.plasma.dE; rtol = 1.0e-14)

        # Collision frequency mode (Maxwell molecule)
        ch_el_nu = ElasticCollision(1.0e-4; is_cross_section = false)
        ex_el_nu = compute_exchange(ch_el_nu, ion_h, neutral, 16.0)
        @test ex_el_nu.plasma.dmom == -ex_el_nu.neutral.dmom
        @test ex_el_nu.plasma.dE == -ex_el_nu.neutral.dE

        # 6. Recombination conservation
        ch_rec = Recombination(1.0e-5; E_rad = 0.03)
        ex_rec = compute_exchange(ch_rec, plasma, neutral, m_n; n_e = 5.0, T_e = 0.1)
        R_rec = 1.0e-5 * 5.0 * 5.0
        @test ex_rec.plasma.dn ≈ -R_rec
        @test ex_rec.plasma.dn + ex_rec.neutral.dn == 0.0
        @test norm(ex_rec.plasma.dmom + ex_rec.neutral.dmom) <= 1.0e-15
        @test isapprox(ex_rec.plasma.dE + ex_rec.neutral.dE, -R_rec * 0.03; rtol = 1.0e-12)

        # Combined tuple of channels
        channels = (ch_ph, ch_cx, ch_el_nu, ch_rec)
        ex_all = compute_exchange(channels, plasma, neutral, m_n; atten = 0.8, n_e = 5.0, T_e = 0.1)
        @test ex_all.plasma.dn ≈ ex_ph.plasma.dn + ex_rec.plasma.dn
        @test _alloc_compute_exchange(channels, plasma, neutral, m_n, 0.8) == 0

        # Kinetic / Hybrid-PIC kernels
        @test collision_probability(0.0, 0.1) == 0.0
        @test collision_probability(1.0e-12, 1.0) ≈ 1.0e-12
        @test collision_probability(2.0, 0.5) ≈ 1.0 - exp(-1.0)
        @test particle_charge_exchange_rate(1.0e-3, 100.0, 5.0) ≈ 0.5
        @test photoionization_creation_weight(1.0e-5, 100.0, 0.8, 0.01, 2.0) ≈ 1.6e-5
        @test sputtering_flux(1.0e10, 0.05) ≈ 5.0e8

        v_i = SVector(-4.0, 1.0, 0.0)
        v_n = SVector(0.1, -0.2, 0.3)
        ev = charge_exchange_collision(SVector(1.5, 0.0, 0.0), v_i, 16.0, 2.5, v_n, 1)
        @test ev.ion_vel == v_n
        @test ev.ena.vel == v_i
        @test ev.ena.mass == 16.0 && ev.ena.weight == 2.5 && ev.ena.species == Int32(1)
        @test ev.neutral_delta.dmom ≈ 16.0 * 2.5 * (v_i - v_n)
        @test ev.neutral_delta.dE ≈ 0.5 * 16.0 * 2.5 * (dot(v_i, v_i) - dot(v_n, v_n))
    end

    @testset "GriddedNeutralField, Two-Way Coupling, CIC Conservation & Adapt" begin
        sp_o = NeutralSpecies(; m = 16.0)
        pop_o = NeutralPopulation(
            sp_o,
            ExponentialProfile(100.0, 1.0, 0.5);
            velocity = CorotationVelocity((0.0, 0.0, 0.05)),
            temperature = ConstantTemperature(0.02),
        )
        exo = ExosphereModel(
            pop_o;
            planet_radius = 1.0,
            shadow = ShadowCylinder(; solar_dir = (1.0, 0.0, 0.0), radius = 1.0),
        )

        # 3D Cartesian GriddedNeutralField (one-way tabulated vs two-way)
        grid3 = CartesianNeutralGrid((-2.0, -2.0, -2.0), (0.5, 0.5, 0.5), (9, 9, 9))
        @test isbits(grid3)
        @test cell_volume(grid3) ≈ 0.125
        @test cell_center(grid3, CartesianIndex(1, 1, 1)) == SVector(-2.0, -2.0, -2.0)

        field_1way = GriddedNeutralField(grid3, 1; masses = (16.0,), two_way = false, background = exo)
        @test coupling_mode(field_1way) === OneWayCoupling()
        @test neutral_sources(field_1way) === nothing
        @test neutral_particles(field_1way) === nothing
        sample_to_grid!(field_1way, exo)
        # Exact match at grid nodes and fallback outside grid
        x_node = cell_center(grid3, CartesianIndex(8, 5, 5))
        @test neutral_state(field_1way, x_node; species = 1).n ≈ neutral_state(exo, x_node; species = 1).n
        @test neutral_state(field_1way, SVector(5.0, 0.0, 0.0); species = 1).n ≈
            neutral_state(exo, SVector(5.0, 0.0, 0.0); species = 1).n

        # Two-way field with Eulerian source buffer + kinetic particle buffer + RayOpticalDepth
        rod = RayOpticalDepth(; cross_sections = (0.01,), s_max = 4.0, planet_radius = 1.0, n_quad = 8)
        field_2way = GriddedNeutralField(
            grid3, 1;
            masses = (16.0,),
            two_way = true,
            particle_capacity = 16,
            attenuation = rod,
        )
        @test coupling_mode(field_2way) === TwoWayCoupling()
        sample_to_grid!(field_2way, exo)

        # Couple at arbitrary off-grid point & verify exact CIC conservation
        x_pt = SVector(1.37, -0.22, 0.19)
        ion = PlasmaState(5.0, SVector(-2.0, 0.3, 0.0), 0.1, 16.0)
        channels = (Photoionization(1.0e-5), ResonantChargeExchange(2.0e-3))
        reset_sources!(field_2way)
        psrc = couple_point!(field_2way, x_pt, channels, ion; species = 1)
        sbuf = neutral_sources(field_2way)
        @test sum(sbuf.dn) ≈ -psrc.dn
        @test isapprox(sum(sbuf.dmom), -psrc.dmom; rtol = 1.0e-13)
        @test isapprox(sum(sbuf.dE), -psrc.dE; rtol = 1.0e-13)
        @test _alloc_couple_point!(field_2way, x_pt, channels, ion, 1) == 0

        # Matched-grid direct cell coupling via CartesianIndex
        reset_sources!(field_2way)
        idx = CartesianIndex(8, 5, 5)
        psrc_idx = couple_point!(field_2way, idx, channels, ion; species = 1)
        @test sbuf.dn[idx, 1] ≈ -psrc_idx.dn
        @test sbuf.dmom[idx, 1] ≈ -psrc_idx.dmom
        @test sbuf.dE[idx, 1] ≈ -psrc_idx.dE

        # Multi-threaded buffer reduction helper
        buf_b = NeutralSourceBuffer(Float64, (9, 9, 9), 1)
        merge_sources!(buf_b, sbuf)
        @test buf_b.dn[idx, 1] == sbuf.dn[idx, 1]

        # Kinetic ENA particle buffer (zero-alloc push below capacity)
        ena = NeutralParticle(x_pt, SVector(-3.0, 0.0, 0.0), 16.0, 1.0, Int32(1))
        deposit_neutral_particle!(field_2way, ena)
        @test length(active_particles(neutral_particles(field_2way))) == 1
        @test active_particles(neutral_particles(field_2way))[1] == ena
        # Static one-way model ignores particle & source deposition at zero cost
        @test deposit_neutral_particle!(exo, ena) === nothing
        @test _alloc_deposit_source!(exo, x_pt, psrc, 1) == 0
        reset_sources!(field_2way)
        @test isempty(active_particles(neutral_particles(field_2way)))

        # SphericalNeutralGrid (1D radial & 2D axisymmetric & 3D)
        r_nodes = collect(range(1.0, 3.0; length = 11))
        sgrid_2d = SphericalNeutralGrid(r_nodes; ntheta = 8, nphi = 1)
        sfield_2d = GriddedNeutralField(sgrid_2d, 1; masses = (16.0,), two_way = true)
        sample_to_grid!(sfield_2d, exo)
        x_sp = SVector(1.4, 0.6, 0.0)
        @test neutral_density(sfield_2d, x_sp; species = 1) > 0.0
        psrc_sp = couple_point!(sfield_2d, x_sp, (ResonantChargeExchange(1.0e-3),), ion; species = 1)
        @test isapprox(sum(sfield_2d.sources.dmom), -psrc_sp.dmom; rtol = 1.0e-13)

        # Adaptive box quadrature on general AbstractNeutralModel (GriddedNeutralField)
        sumN_g = zeros(1)
        sumNA_g = zeros(1)
        vol_g = integrate_exosphere_box!(
            sumN_g, sumNA_g, field_2way, SVector(1.5, 0.0, 0.0), SVector(0.2, 0.2, 0.2), 0, 1.0,
        )
        @test vol_g ≈ 0.008 && sumN_g[1] > 0.0 && sumNA_g[1] > 0.0

        # Adapt.jl extension (GPU-style array backend & Float32 element adaptation)
        f32_adapted = Adapt.adapt(_TestFloat32Adaptor(), field_2way)
        @test eltype(f32_adapted.n) === Float32
        @test eltype(f32_adapted.u) === SVector{3, Float32}
        @test eltype(f32_adapted.sources.dn) === Float32
        @test Adapt.adapt(Array, field_1way).n == field_1way.n
    end

    @testset "Automatic Differentiation (ForwardDiff)" begin
        exp_p = ExponentialProfile(100.0, 1.0, 0.25)
        pl_p = PowerLawProfile(10.0, 1.0, 2.0)
        ch_p = ChamberlainProfile(50.0, 1.0, 2.0)
        ep = exosphere_profile(; n0 = 10.0, r0 = 1.0, H = 0.5, power = 1.0)
        plume = PlumeProfile(;
            n0 = 20.0, origin = (0.0, 0.0, -1.0), axis = (0.0, 0.0, -1.0),
            r0 = 0.1, H = 0.5, k = 2.0, opening_angle = 0.4,
        )

        # Analytic radial derivative checks
        r0 = 1.6
        @test ForwardDiff.derivative(r -> neutral_density(exp_p, r), r0) ≈
            -neutral_density(exp_p, r0) / 0.25
        @test ForwardDiff.derivative(r -> neutral_density(pl_p, r), r0) ≈
            -2.0 * neutral_density(pl_p, r0) / r0
        @test ForwardDiff.derivative(r -> neutral_density(ch_p, r), r0) ≈
            -2.0 * neutral_density(ch_p, r0) / (r0 * r0)
        @test ForwardDiff.derivative(r -> neutral_density(ep, r), r0) ≈
            -neutral_density(ep, r0) / local_scale_height(ep, r0)

        # Smooth PlumeProfile gradient right ON the plume axis (no acos NaN!)
        g_axis = ForwardDiff.gradient(x -> neutral_density(plume, x), SVector(0.0, 0.0, -1.5))
        @test all(isfinite, g_axis)
        @test g_axis[1] == 0.0 && g_axis[2] == 0.0 && g_axis[3] > 0.0

        # Chapman & photo_attenuation derivatives vs finite differences
        fd_chap = (chapman_function(20.0, 0.5 + 1.0e-6) - chapman_function(20.0, 0.5 - 1.0e-6)) / 2.0e-6
        ad_chap = ForwardDiff.derivative(c -> chapman_function(20.0, c), 0.5)
        @test isapprox(ad_chap, fd_chap; rtol = 1.0e-6)

        sp_o = NeutralSpecies(; m = 16.0)
        pop_o = NeutralPopulation(sp_o, exp_p; temperature = ConstantTemperature(0.01))
        exo_ad = ExosphereModel(
            pop_o; planet_radius = 1.0,
            optical_depth = OpticalDepth(; cross_sections = (0.05,), chapman = true),
        )
        x_ad = SVector(1.5, 0.3, 0.1)
        g_att = ForwardDiff.gradient(x -> photo_attenuation(exo_ad, x), x_ad)
        @test all(isfinite, g_att)
        fd_att_x = (
            photo_attenuation(exo_ad, x_ad + SVector(1.0e-6, 0.0, 0.0)) -
                photo_attenuation(exo_ad, x_ad - SVector(1.0e-6, 0.0, 0.0))
        ) / 2.0e-6
        @test isapprox(g_att[1], fd_att_x; rtol = 1.0e-5)

        # ForwardDiff through compute_exchange
        d_cx = ForwardDiff.derivative(
            vx -> compute_exchange(
                ResonantChargeExchange(1.0e-3),
                PlasmaState(5.0, SVector(vx, 0.0, 0.0), 0.1, 16.0),
                NeutralState(100.0, SVector(0.0, 0.0, 0.0), 0.01),
                16.0,
            ).plasma.dmom[1],
            -2.0,
        )
        @test isfinite(d_cx) && d_cx < 0.0

        # ForwardDiff through GriddedNeutralField trilinear interpolation
        g_lin = CartesianNeutralGrid((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), (4, 4, 4))
        f_lin = GriddedNeutralField(g_lin, 1)
        for idx in CartesianIndices((4, 4, 4))
            xc = cell_center(g_lin, idx)
            f_lin.n[idx, 1] = 2.0 * xc[1] - 3.0 * xc[2] + 5.0 * xc[3] + 20.0
        end
        g_interp = ForwardDiff.gradient(x -> neutral_density(f_lin, x; species = 1), SVector(0.3, 0.4, -0.2))
        @test isapprox(g_interp, SVector(2.0, -3.0, 5.0); atol = 1.0e-12)
    end

    @testset "Unitful.jl Extension" begin
        # Direct Unitful arithmetic in source rates & conservative exchange
        q_u = photoionization_rate(2.0e-7u"s^-1", 1.0e6u"cm^-3", 0.5)
        @test isapprox(uconvert(u"cm^-3 * s^-1", q_u), 0.1u"cm^-3 * s^-1")
        r_cx_u = charge_exchange_rate(2.0e-15u"cm^2", 5.0u"cm^-3", 1.0e6u"cm^-3", 400.0u"km/s")
        @test isapprox(uconvert(u"cm^-3 * s^-1", r_cx_u), 0.4u"cm^-3 * s^-1")

        # UnitfulProfile & strip_units to code units
        up_exp = ExponentialProfile(1.0e6u"cm^-3", 3390.0u"km", 200.0u"km")
        @test isapprox(neutral_density(up_exp, 3590.0u"km"), 1.0e6 * exp(-1.0) * u"cm^-3")
        @test isapprox(local_scale_height(up_exp, 3590.0u"km"), 200.0u"km")

        scale = UnitScale(;
            length = 3390.0u"km", density = 1.0e6u"cm^-3", velocity = 100.0u"km/s",
            mass = 16.0u"u",
        )
        p_code = strip_units(up_exp, scale)
        @test isbits(p_code) && p_code isa ExponentialProfile{Float64}
        @test neutral_density(p_code, 3590.0 / 3390.0) ≈ exp(-1.0)
        @test to_code_units(scale, 3390.0u"km") ≈ 1.0
        @test isapprox(from_code_units(scale, 1.0, u"km"), 3390.0u"km")

        # NeutralSpecies with Unitful mass
        sp_u = NeutralSpecies(; m = 16.0u"u")
        @test atomic_mass(sp_u) ≈ 16.0

        # Physical SI conservative exchange with Unitful quantities and kB = KB_SI
        p_si = PlasmaState(5.0u"cm^-3", SVector(-300.0u"km/s", 0.0u"km/s", 0.0u"km/s"), 1.0e5u"K", 16.0u"u")
        n_si = NeutralState(1.0e5u"cm^-3", SVector(0.0u"km/s", 0.0u"km/s", 0.0u"km/s"), 200.0u"K")
        ex_si = compute_exchange(
            ResonantChargeExchange(2.0e-19u"m^2"), p_si, n_si, 16.0u"u";
            kB = Exosphere.KB_SI * u"J/K",
        )
        @test ex_si.plasma.dmom == -ex_si.neutral.dmom
        @test ex_si.plasma.dE == -ex_si.neutral.dE
    end

    @testset "PlasmaPopulations.jl Extension" begin
        pp_o0 = PP.Particle("O", 0)
        sp_from_pp = NeutralSpecies(pp_o0)
        @test atomic_mass(sp_from_pp) ≈ 15.999
        @test_throws ArgumentError NeutralSpecies(PP.Proton)

        pp_vdf = PP.BiMaxwellian(pp_o0; T = 0.02, vd_para = 100.0)
        m_exo = Maxwellian(pp_vdf)
        @test m_exo.vth ≈ pp_vdf.vth_para
        @test mean(m_exo) == SVector(0.0, 0.0, 100.0)

        pop_n = NeutralPopulation(pp_o0, ExponentialProfile(1.0e12, 3.39e6, 2.0e5), pp_vdf)
        @test isbits(pop_n)
        @test species_mass(pop_n) ≈ pp_o0.m

        pp_ion_pop = PP.Population(PP.Proton, PP.BiMaxwellian; T = 10.0, n = 5.0e6, vd_para = -4.0e5)
        ps = PlasmaState(pp_ion_pop)
        @test ps.n ≈ 5.0e6
        @test ps.m ≈ PP.Proton.m
        @test ps.u ≈ SVector(0.0, 0.0, -4.0e5)
        @test ps.T ≈ (10.0 * Exosphere.E_SI) / Exosphere.KB_SI
    end

    @testset "Type Stability (@inferred) & Zero-Allocation Hot Paths" begin
        exp32 = ExponentialProfile(10.0f0, 1.0f0, 0.5f0; FT = Float32)
        pl32 = PowerLawProfile(10.0f0, 1.0f0, 2.0f0; FT = Float32)
        ch32 = ChamberlainProfile(10.0f0, 1.0f0, 2.0f0; FT = Float32)
        @test (@inferred neutral_density(exp32, 1.5f0)) isa Float32
        @test (@inferred local_scale_height(exp32, 1.5f0)) ≈ 0.5f0
        @test (@inferred neutral_density(pl32, 2.0f0)) ≈ 2.5f0
        @test (@inferred neutral_density(ch32, 2.0f0)) ≈ 10.0f0 * exp(-1.0f0)
        @test (@inferred photoionization_rate(1.0f-6, 100.0f0)) isa Float32
        @test (@inferred charge_exchange_rate(2.0f-19, 5.0f0, 100.0f0, 4.0f5)) isa Float32

        for k in (0.0, 1.0, 2.0, 2.5, 3.0, 4.0)
            pk = PowerLawProfile(8.0, 1.0, k)
            @test neutral_density(pk, 2.0) ≈ 8.0 * 0.5^k
        end

        ep = exosphere_profile(; n0 = 4.0, r0 = 1.0, H = 0.5, center = (1.0, 0.0, 0.0))
        @test (@inferred ep(1.5)) isa Float64
        @test (@inferred ep(1.5, 0.2)) isa Float64
        @test (@inferred ep(1.5, 0.2, -0.1)) isa Float64

        c1 = NeutralComponentCode(:exponential, 100.0, 0.2, 0.0, 0.01, "O")
        c2 = NeutralComponentCode(:powerlaw, 10.0, 0.0, 2.0, 0.01, "H")
        c3 = NeutralComponentCode(:chamberlain, 50.0, 1.5, 0.0, 0.01, "He")
        sc = ShadowCylinder(; solar_dir = (1.0, 0.0, 0.0), radius = 1.0, half_height = 5.0)
        od_pl = OpticalDepth(; cross_sections = [0.05, 0.01, 0.02], chapman = false)
        od_ch = OpticalDepth(; cross_sections = [0.05, 0.01, 0.02], chapman = true)
        exo_sc = build_exosphere_model(; planet_radius = 1.0, components = [c1, c2, c3], shadow = sc, inner_radius = 1.0)
        exo_od = build_exosphere_model(; planet_radius = 1.0, components = [c1, c2, c3], optical_depth = od_pl, inner_radius = 1.0)
        exo_ch = build_exosphere_model(; planet_radius = 1.0, components = [c1, c2, c3], optical_depth = od_ch, inner_radius = 1.0)

        x_day = SVector(1.5, 0.0, 0.0)
        x_night = SVector(-1.5, 0.5, 0.0)
        x_graz = SVector(-1.5, 1.2, 0.0)
        x_term = SVector(-0.15, 1.5, 0.0)

        @test (@inferred shadow_factor(sc, x_night)) == 0.0
        @test (@inferred exo_component_density(exo_ch, 1.5, 1)) isa Float64
        @test (@inferred exo_density(exo_ch, 1.5)) isa Float64
        @test (@inferred exo_scale_height(exo_ch, 1.5, 1)) isa Float64
        @test (@inferred chapman_function(20.0, 0.7)) isa Float64
        @test (@inferred chapman_function(20.0, -0.1)) isa Float64
        @test (@inferred photo_attenuation(exo_sc, x_night, hypot(x_night...))) == 0.0
        @test (@inferred photo_attenuation(exo_od, x_day, 1.5)) isa Float64
        @test (@inferred photo_attenuation(exo_od, x_graz, hypot(x_graz...))) isa Float64
        @test (@inferred photo_attenuation(exo_ch, x_day, 1.5)) isa Float64
        @test (@inferred photo_attenuation(exo_ch, x_term, hypot(x_term...))) isa Float64

        sumN = zeros(3)
        sumNA = zeros(3)
        @test (@inferred integrate_exosphere_box!(sumN, sumNA, exo_ch, x_day, SVector(0.2, 0.2, 0.2), 0, 1.0)) isa Float64

        @test _alloc_shadow_factor(sc, x_night) == 0
        @test _alloc_exo_component_density(exo_ch, 1.5, 2) == 0
        @test _alloc_exo_density(exo_ch, 1.5) == 0
        @test _alloc_exo_scale_height(exo_ch, 1.5, 3) == 0
        @test _alloc_chapman(20.0, -0.1) == 0
        @test _alloc_photo_attenuation(exo_sc, x_night, hypot(x_night...)) == 0
        @test _alloc_photo_attenuation(exo_od, x_day, 1.5) == 0
        @test _alloc_photo_attenuation(exo_od, x_graz, hypot(x_graz...)) == 0
        @test _alloc_photo_attenuation(exo_ch, x_day, 1.5) == 0
        @test _alloc_photo_attenuation(exo_ch, x_term, hypot(x_term...)) == 0
        @test _alloc_integrate_box!(sumN, sumNA, exo_ch, x_day, SVector(0.2, 0.2, 0.2), 0, 1.0) == 0
        @test _alloc_integrate_box!(sumN, sumNA, exo_ch, SVector(1.05, 0.1, 0.0), SVector(0.4, 0.4, 0.4), 0, 1.0) == 0
        @test _alloc_integrate_box!(sumN, sumNA, exo_ch, SVector(1.05, 0.1, 0.0), SVector(0.4, 0.4, 0.0), 0, 1.0) == 0
        @test _alloc_photoionization_rate(1.0e-6, 100.0, 0.85) == 0
        @test _alloc_charge_exchange_rate(2.0e-19, 5.0, 100.0, 4.0e5) == 0

        # Tuple-backed isbits ExosphereModel zero-alloc & @inferred checks
        pop1 = NeutralPopulation(NeutralSpecies(; m = 16.0), ExponentialProfile(100.0, 1.0, 0.2); temperature = ConstantTemperature(0.01))
        pop2 = NeutralPopulation(NeutralSpecies(; m = 1.0), PowerLawProfile(10.0, 1.0, 2.0); temperature = ConstantTemperature(0.01))
        pop3 = NeutralPopulation(NeutralSpecies(; m = 4.0), ChamberlainProfile(50.0, 1.0, 1.5); temperature = ConstantTemperature(0.01))
        exo_tup = ExosphereModel(
            pop1, pop2, pop3;
            planet_radius = 1.0,
            optical_depth = OpticalDepth(; cross_sections = (0.05, 0.01, 0.02), chapman = true),
            inner_radius = 1.0,
        )
        @test isbits(exo_tup)
        @test (@inferred exo_density(exo_tup, 1.5)) isa Float64
        @test (@inferred neutral_state(exo_tup, x_day; species = 1)) isa NeutralState{Float64, Float64, Float64}
        ion = PlasmaState(5.0, SVector(-2.0, 0.0, 0.0), 0.1, 16.0)
        ch_pair = (Photoionization(1.0e-6), ResonantChargeExchange(2.0e-3))
        @test (@inferred couple_point!(exo_tup, x_day, ch_pair, ion; species = 1)) isa MomentSource{Float64, Float64, Float64}
        @test _alloc_exo_density(exo_tup, 1.5) == 0
        @test _alloc_photo_attenuation(exo_tup, x_term, hypot(x_term...)) == 0
        @test _alloc_neutral_state(exo_tup, x_day, 1) == 0
        @test _alloc_couple_point!(exo_tup, x_day, ch_pair, ion, 1) == 0
        @test _alloc_integrate_box!(sumN, sumNA, exo_tup, x_day, SVector(0.2, 0.2, 0.2), 0, 1.0) == 0
    end
end
