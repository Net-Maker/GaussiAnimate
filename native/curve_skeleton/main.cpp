// SPDX-License-Identifier: GPL-3.0-or-later

#include <CGAL/Mean_curvature_flow_skeletonization.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/Surface_mesh/IO.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
namespace PMP = CGAL::Polygon_mesh_processing;

using Kernel = CGAL::Simple_cartesian<double>;
using Point = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point>;
using Skeletonization = CGAL::Mean_curvature_flow_skeletonization<Mesh>;
using Skeleton = Skeletonization::Skeleton;
using SkeletonVertex = boost::graph_traits<Skeleton>::vertex_descriptor;

struct Options {
    fs::path input;
    fs::path output_prefix = "curve_skeleton";
    double quality = 0.1;
    double medial = 0.25;
    double convergence = 1e-4;
    int iterations = 500;
    bool medially_centered = true;
};

void usage(const char* program) {
    std::cout
        << "Usage: " << program << " INPUT.off OUTPUT_PREFIX [options]\n\n"
        << "Extract a mean-curvature-flow curve skeleton from one closed,\n"
        << "connected triangle mesh. OUTPUT_PREFIX produces:\n"
        << "  *_points.txt  x y z per skeleton vertex\n"
        << "  *_edges.txt   zero-based endpoint indices\n"
        << "  *.obj         the same skeleton as OBJ lines\n\n"
        << "Options:\n"
        << "  --quality VALUE      quality/speed tradeoff (default: 0.1)\n"
        << "  --medial VALUE       medial-centering tradeoff (default: 0.25)\n"
        << "  --convergence VALUE  area variation threshold (default: 1e-4)\n"
        << "  --iterations COUNT   maximum iterations (default: 500)\n"
        << "  --no-medial          disable medial centering\n"
        << "  --help                show this message\n";
}

double parse_unit_value(const std::string& text, const char* name) {
    const double value = std::stod(text);
    if (value < 0.0 || value > 1.0) {
        throw std::invalid_argument(std::string(name) + " must be in [0, 1]");
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        usage(argv[0]);
        std::exit(0);
    }
    if (argc < 3) {
        usage(argv[0]);
        throw std::invalid_argument("INPUT.off and OUTPUT_PREFIX are required");
    }
    Options options;
    options.input = argv[1];
    options.output_prefix = argv[2];
    for (int index = 3; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string {
            if (++index >= argc) {
                throw std::invalid_argument(argument + " requires a value");
            }
            return argv[index];
        };
        if (argument == "--quality") {
            options.quality = parse_unit_value(next(), "quality");
        } else if (argument == "--medial") {
            options.medial = parse_unit_value(next(), "medial");
        } else if (argument == "--convergence") {
            options.convergence = std::stod(next());
            if (options.convergence <= 0.0) {
                throw std::invalid_argument("convergence must be positive");
            }
        } else if (argument == "--iterations") {
            options.iterations = std::stoi(next());
            if (options.iterations < 1) {
                throw std::invalid_argument("iterations must be positive");
            }
        } else if (argument == "--no-medial") {
            options.medially_centered = false;
        } else if (argument == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }
    return options;
}

void validate_mesh(Mesh& mesh) {
    if (!CGAL::is_triangle_mesh(mesh)) {
        throw std::runtime_error("input must be a triangle mesh");
    }
    if (!CGAL::is_closed(mesh)) {
        throw std::runtime_error(
            "input mesh is not watertight; provide a closed mesh or use the "
            "GaussiAnimate Poisson proxy"
        );
    }
    auto components = mesh.add_property_map<Mesh::Face_index, std::size_t>(
        "f:component", 0
    ).first;
    if (PMP::connected_components(mesh, components) != 1) {
        throw std::runtime_error("input mesh must contain exactly one component");
    }
}

void save_skeleton(const Skeleton& skeleton, const fs::path& prefix) {
    if (!prefix.parent_path().empty()) {
        fs::create_directories(prefix.parent_path());
    }
    std::ofstream points(prefix.string() + "_points.txt");
    std::ofstream edges(prefix.string() + "_edges.txt");
    std::ofstream object(prefix.string() + ".obj");
    if (!points || !edges || !object) {
        throw std::runtime_error("could not create curve-skeleton outputs");
    }
    points << std::setprecision(17);
    object << std::setprecision(17) << "# GaussiAnimate CGAL curve skeleton\n";
    std::map<SkeletonVertex, std::size_t> indices;
    std::size_t index = 0;
    for (const SkeletonVertex vertex : CGAL::make_range(vertices(skeleton))) {
        const Point& point = skeleton[vertex].point;
        indices[vertex] = index++;
        points << point.x() << ' ' << point.y() << ' ' << point.z() << '\n';
        object << "v " << point.x() << ' ' << point.y() << ' ' << point.z() << '\n';
    }
    for (const auto edge : CGAL::make_range(boost::edges(skeleton))) {
        const auto source = boost::source(edge, skeleton);
        const auto target = boost::target(edge, skeleton);
        edges << indices[source] << ' ' << indices[target] << '\n';
        object << "l " << indices[source] + 1 << ' ' << indices[target] + 1 << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        Mesh mesh;
        std::ifstream input(options.input);
        if (!input || !(input >> mesh)) {
            throw std::runtime_error("could not read input mesh: " + options.input.string());
        }
        validate_mesh(mesh);

        Skeletonization extractor(mesh);
        extractor.set_quality_speed_tradeoff(options.quality);
        extractor.set_medially_centered_speed_tradeoff(options.medial);
        extractor.set_area_variation_factor(options.convergence);
        extractor.set_max_iterations(options.iterations);
        extractor.set_is_medially_centered(options.medially_centered);

        Skeleton skeleton;
        extractor(skeleton);
        if (boost::num_vertices(skeleton) == 0) {
            throw std::runtime_error("CGAL produced an empty skeleton");
        }
        save_skeleton(skeleton, options.output_prefix);
        std::cout << "curve skeleton: " << boost::num_vertices(skeleton)
                  << " points, " << boost::num_edges(skeleton) << " edges\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "curve_skeleton: " << error.what() << '\n';
        return 1;
    }
}
