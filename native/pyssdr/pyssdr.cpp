// Minimal NumPy binding for the MIT-licensed Dem Bones headers.

#include <DemBones/DemBonesExt.h>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <string>
#include <vector>

namespace py = pybind11;

class MyDemBones : public Dem::DemBonesExt<double, float> {
public:
    double tolerance = 1e-3;
    int patience = 3;
    double rmse_err = 0.0;
    int initializedBones = 0;

    MyDemBones() {
        nIters = 100;
        previous_error_ = -1.0;
        remaining_patience_ = patience;
    }

    void load_data(
        const Eigen::MatrixXd& trajectory,
        const std::vector<std::vector<int>>& faces
    ) {
        if (trajectory.rows() % 3 != 0 || trajectory.cols() == 0) {
            throw std::invalid_argument(
                "trajectory must have shape [3 * frames, vertices]"
            );
        }
        clear();
        nS = 1;
        nF = static_cast<int>(trajectory.rows() / 3);
        nV = static_cast<int>(trajectory.cols());
        fStart.resize(2);
        fStart << 0, nF;
        subjectID = Eigen::VectorXi::Zero(nF);
        v = trajectory.cast<float>();
        u = trajectory.topRows(3);
        fv = faces;
        fTime = Eigen::VectorXd::LinSpaced(nF, 0.0, std::max(0, nF - 1));
    }

    py::tuple run_ssdr(int initial_bones = 30, const std::string& = "") {
        if (nV == 0 || nF == 0) {
            throw std::runtime_error("load_data must be called before run_ssdr");
        }
        if (nB == 0) {
            nB = initial_bones;
        }
        previous_error_ = -1.0;
        remaining_patience_ = patience;
        init();
        initializedBones = nB;
        compute();
        rmse_err = rmse();
        return py::make_tuple(w, m, rmse_err);
    }

    bool cbIterEnd() override {
        rmse_err = rmse();
        if (
            previous_error_ >= 0.0 &&
            rmse_err < previous_error_ * (1.0 + weightEps) &&
            previous_error_ - rmse_err < tolerance * previous_error_
        ) {
            if (--remaining_patience_ == 0) {
                return true;
            }
        } else {
            remaining_patience_ = patience;
        }
        previous_error_ = rmse_err;
        return false;
    }

private:
    double previous_error_;
    int remaining_patience_;
};

PYBIND11_MODULE(pyssdr, module) {
    module.doc() = "Minimal Dem Bones SSDR binding for GaussiAnimate";
    py::class_<MyDemBones>(module, "MyDemBones")
        .def(py::init<>())
        .def("load_data", &MyDemBones::load_data)
        .def("run_ssdr", &MyDemBones::run_ssdr,
             py::arg("initial_bones") = 30, py::arg("output_path") = "")
        .def_readwrite("weightsSmoothStep", &MyDemBones::weightsSmoothStep)
        .def_readwrite("weightsSmooth", &MyDemBones::weightsSmooth)
        .def_readwrite("nnz", &MyDemBones::nnz)
        .def_readwrite("nWeightsIters", &MyDemBones::nWeightsIters)
        .def_readwrite("transAffineNorm", &MyDemBones::transAffineNorm)
        .def_readwrite("transAffine", &MyDemBones::transAffine)
        .def_readwrite("bindUpdate", &MyDemBones::bindUpdate)
        .def_readwrite("nTransIters", &MyDemBones::nTransIters)
        .def_readwrite("patience", &MyDemBones::patience)
        .def_readwrite("tolerance", &MyDemBones::tolerance)
        .def_readwrite("nIters", &MyDemBones::nIters)
        .def_readwrite("nInitIters", &MyDemBones::nInitIters)
        .def_readwrite("clusterThreshold", &MyDemBones::clusterThreshold)
        .def_readwrite("labelAssignmentMode", &MyDemBones::labelAssignmentMode)
        .def_readwrite("splitSeedNeighbors", &MyDemBones::splitSeedNeighbors)
        .def_readwrite("pruneDelay", &MyDemBones::pruneDelay)
        .def_readwrite("zeroGrowthPatience", &MyDemBones::zeroGrowthPatience)
        .def_readwrite("pointCloudSmoothMode", &MyDemBones::pointCloudSmoothMode)
        .def_readwrite("nB", &MyDemBones::nB)
        .def_readwrite("label", &MyDemBones::label)
        .def_readonly("initializedBones", &MyDemBones::initializedBones)
        .def_readonly("nV", &MyDemBones::nV)
        .def_readonly("nF", &MyDemBones::nF)
        .def("labelToWeights", &MyDemBones::labelToWeights)
        .def("computeTransFromLabel", &MyDemBones::computeTransFromLabel);
}
