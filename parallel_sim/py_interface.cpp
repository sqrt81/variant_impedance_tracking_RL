#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "parallel_sim.h"

namespace py = pybind11;
namespace lit = pybind11::literals;

// python interface
class ParSim : public mujoco_parallel_sim::ParallelSimEnv
{
private:
    using base = mujoco_parallel_sim::ParallelSimEnv;
    using NpArrayd = Eigen::Matrix<double, 1, -1, Eigen::RowMajor>;
    using NpArrayb = Eigen::Matrix<bool, 1, -1, Eigen::RowMajor>;
    using NpMatrixd = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;

public:
    ParSim(int num_envs, int num_threads)
        : mujoco_parallel_sim::ParallelSimEnv(num_envs, num_threads) {}

    void StepPy(const Eigen::Ref<const NpMatrixd> act,
                Eigen::Ref<NpMatrixd> obs, Eigen::Ref<NpArrayb> done,
                Eigen::Ref<NpArrayd> rew, Eigen::Ref<NpArrayb> timeout,
                Eigen::Ref<NpArrayd> v_mean)
    {
        mujoco_parallel_sim::ParallelSimEnv::Step(
                    act.data(), obs.data(), done.data(),
                    rew.data(), timeout.data(),
                    v_mean.size() ? v_mean.data() : nullptr);
    }

    void GetDifficultyPy(Eigen::Ref<NpArrayd> v_cmd, Eigen::Ref<NpArrayd> height)
    {
        mujoco_parallel_sim::ParallelSimEnv::GetEnvDifficulty(
                    v_cmd.data(), height.data());
    }

    void GetMaxDifficultyPy(Eigen::Ref<NpArrayd> v_cmd, Eigen::Ref<NpArrayd> height)
    {
        mujoco_parallel_sim::ParallelSimEnv::GetEnvMaxDifficulty(
                    v_cmd.data(), height.data());
    }

    std::map<std::string, double> GetStat() const
    {
        double vx_mean;
        double eposide_len_mean;
        double terrain_height_mean;
        base::GetStatistics(vx_mean, eposide_len_mean,
                            terrain_height_mean);

        return {
            std::make_pair("vx_mean", vx_mean),
            std::make_pair("eposide_len_mean", eposide_len_mean),
            std::make_pair("terrain_height_mean", terrain_height_mean),
        };
    }
};

PYBIND11_MODULE(_mj_parallel, pym)
{
    pym.doc() = "parallel simulation env by mujoco";
    py::class_<ParSim>(pym, "ParallelSim")
            .def(py::init<int, int>())
            .def("LoadParam", &ParSim::LoadParams)
            .def("LoadModelXml", &ParSim::LoadModelXml)
            .def("Step", &ParSim::StepPy, py::arg().noconvert(),
                 py::arg(), py::arg(), py::arg(), py::arg(), py::arg())
            .def("SetToEvalMode", &ParSim::SetToEvalMode)
            .def("GetDifficulty", &ParSim::GetDifficultyPy,
                 py::arg(), py::arg())
            .def("GetMaxDifficulty", &ParSim::GetMaxDifficultyPy,
                 py::arg(), py::arg())
            .def("GetStat", &ParSim::GetStat)
            .def("Render", &ParSim::Render)
            .def_property_readonly_static(
                        "kACT_SIZE", [](py::object){
                return mujoco_parallel_sim::A1GymEnv::kACT_SIZE;
            })
            .def_property_readonly_static(
                        "kOBS_SIZE", [](py::object){
                return mujoco_parallel_sim::A1GymEnv::kOBS_SIZE;
            });
}
