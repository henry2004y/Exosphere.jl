# ==============================================================================
# Standalone Planetary Exosphere & Chapman EUV Attenuation Example (3-Panel)
# ==============================================================================
#
# Usage:
#   julia --project=<env-with-CairoMakie-and-Exosphere> \
#       examples/planetary_exosphere_euv.jl [output_path.png]

using CairoMakie
using Exosphere
using StaticArrays

function run_planetary_exosphere_euv(;
        outpath::AbstractString = isempty(ARGS) ? joinpath(@__DIR__, "planetary_exosphere_euv.png") : ARGS[1],
    )
    # 3-component Mars-like exosphere in planetary radius units (R_p = 1.0), Sun along +x
    o_comp = NeutralComponentCode(:exponential, 120.0, 0.18, 0.0, 0.01, "O (Exp)")
    h_comp = NeutralComponentCode(:powerlaw, 12.0, 0.0, 2.0, 0.01, "H (PowerLaw)")
    he_comp = NeutralComponentCode(:chamberlain, 35.0, 1.8, 0.0, 0.01, "He (Chamberlain)")
    comps = [o_comp, h_comp, he_comp]

    od = OpticalDepth(;
        cross_sections = [0.06, 0.01, 0.02],
        solar_dir = (1.0, 0.0, 0.0),
        chapman = true,
    )
    exo = build_exosphere_model(;
        planet_radius = 1.0,
        components = comps,
        optical_depth = od,
        inner_radius = 1.0,
    )

    # Panel 1: Radial neutral density profiles
    rs = collect(range(1.0, 4.0; length = 300))
    n_O = [exo_component_density(exo, r, 1) for r in rs]
    n_H = [exo_component_density(exo, r, 2) for r in rs]
    n_He = [exo_component_density(exo, r, 3) for r in rs]
    n_tot = [exo_density(exo, r) for r in rs]

    # Panels 2 & 3: Equatorial plane (z = 0) EUV attenuation & photoionization rate
    xs = collect(range(-2.5, 2.5; length = 240))
    ys = collect(range(-2.5, 2.5; length = 240))
    atten = fill(NaN, length(xs), length(ys))
    log10_q = fill(NaN, length(xs), length(ys))
    nu_photo = 1.0e-6

    for (j, y) in pairs(ys), (i, x) in pairs(xs)
        r = hypot(x, y)
        r < exo.Rp && continue
        pos = SVector{3, Float64}(x, y, 0.0)
        a = photo_attenuation(exo, pos, r)
        n = exo_density(exo, r)
        q = photoionization_rate(nu_photo, n, a)
        atten[i, j] = a
        log10_q[i, j] = log10(max(q, 1.0e-14))
    end

    th_circ = range(0.0, 2.0 * pi; length = 200)
    planet_x = cos.(th_circ)
    planet_y = sin.(th_circ)

    fig = Figure(; size = (1560, 490), fontsize = 18, figure_padding = 18)

    # Panel 1: Radial density profiles
    ax1 = Axis(
        fig[1, 1];
        title = "Radial Neutral Profiles",
        xlabel = "r / Rₚ",
        ylabel = "Neutral Density (code units)",
        yscale = log10,
        titlesize = 20,
        xlabelsize = 18,
        ylabelsize = 18,
        xticklabelsize = 16,
        yticklabelsize = 16,
    )
    lines!(ax1, rs, n_tot; label = "Total", color = :black, linewidth = 2.8)
    lines!(ax1, rs, n_O; label = "O (Exponential)", color = :steelblue, linewidth = 2.2)
    lines!(ax1, rs, n_He; label = "He (Chamberlain)", color = :darkorange, linewidth = 2.2)
    lines!(ax1, rs, n_H; label = "H (PowerLaw)", color = :seagreen, linewidth = 2.2)
    xlims!(ax1, 1.0, 4.0)
    ylims!(ax1, 0.5, 250.0)
    axislegend(ax1; position = :rt, framevisible = true, labelsize = 15, patchsize = (26, 14))

    # Panel 2: Chapman EUV attenuation factor
    ax2 = Axis(
        fig[1, 2];
        title = "Chapman EUV Attenuation (Sun +x)",
        xlabel = "x / Rₚ",
        ylabel = "y / Rₚ",
        aspect = DataAspect(),
        titlesize = 20,
        xlabelsize = 18,
        ylabelsize = 18,
        xticklabelsize = 16,
        yticklabelsize = 16,
    )
    hm2 = heatmap!(ax2, xs, ys, atten; colormap = :inferno, colorrange = (0.0, 1.0), nan_color = :gray25)
    lines!(ax2, planet_x, planet_y; color = :white, linewidth = 1.8)
    lines!(ax2, [-2.5, 0.0], [1.0, 1.0]; color = :cyan, linestyle = :dash, linewidth = 1.8)
    lines!(ax2, [-2.5, 0.0], [-1.0, -1.0]; color = :cyan, linestyle = :dash, linewidth = 1.8)
    Colorbar(fig[1, 2][1, 2], hm2; label = "Attenuation A", labelsize = 17, ticklabelsize = 15, width = 16)

    # Panel 3: Photoionization rate
    ax3 = Axis(
        fig[1, 3];
        title = "Photoionization Rate log₁₀(q)",
        xlabel = "x / Rₚ",
        ylabel = "y / Rₚ",
        aspect = DataAspect(),
        titlesize = 20,
        xlabelsize = 18,
        ylabelsize = 18,
        xticklabelsize = 16,
        yticklabelsize = 16,
    )
    hm3 = heatmap!(ax3, xs, ys, log10_q; colormap = :viridis, colorrange = (-10.0, -4.0), nan_color = :gray25)
    lines!(ax3, planet_x, planet_y; color = :white, linewidth = 1.8)
    Colorbar(fig[1, 3][1, 2], hm3; label = "log₁₀(q)", labelsize = 17, ticklabelsize = 15, width = 16)

    colgap!(fig.layout, 24)
    mkpath(dirname(abspath(outpath)))
    save(outpath, fig; px_per_unit = 2)
    return outpath
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_planetary_exosphere_euv()
end
