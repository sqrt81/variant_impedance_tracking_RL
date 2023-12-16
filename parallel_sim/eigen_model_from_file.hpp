#ifndef EIGEN_MODEL_FROM_FILE_HPP_
#define EIGEN_MODEL_FROM_FILE_HPP_

#include <fstream>
#include <Eigen/Eigen>
#include <iostream>

namespace {

void ReadFromFile(std::ifstream &file, Eigen::VectorXf &vec)
{
    int rows, cols;
    file.read((char *)&rows, sizeof(int));
    file.read((char *)&cols, sizeof(int));
    vec.resize(rows);
    file.read((char *)vec.data(), vec.size() * sizeof(float));
}

void ReadFromFile(std::ifstream &file, Eigen::MatrixXf &mat)
{
    int rows, cols;
    file.read((char *)&rows, sizeof(int));
    file.read((char *)&cols, sizeof(int));
    mat.resize(rows, cols);
    file.read((char *)mat.data(), mat.size() * sizeof(float));
}

} // anonymous


class EigenModelFromFile {
public:
    EigenModelFromFile(const std::string &model_path) {
        std::ifstream ifs(model_path, std::ios::in | std::ios::binary);

        ReadFromFile(ifs, input_mean_);
        ReadFromFile(ifs, input_inv_std_);
        ReadFromFile(ifs, mlp_w_0_);
        ReadFromFile(ifs, mlp_b_0_);
        ReadFromFile(ifs, mlp_w_1_);
        ReadFromFile(ifs, mlp_b_1_);
        ReadFromFile(ifs, mlp_w_2_);
        ReadFromFile(ifs, mlp_b_2_);
        ReadFromFile(ifs, mlp_w_3_);
        ReadFromFile(ifs, mlp_b_3_);
        ReadFromFile(ifs, mlp_w_4_);
        ReadFromFile(ifs, mlp_b_4_);
    }

    Eigen::VectorXf operator() (const Eigen::VectorXf &input) {
        Eigen::VectorXf out = input - input_mean_;
        out.array() *= input_inv_std_.array();
        Eigen::VectorXf tmp = out.cwiseMin(5).cwiseMax(-5);
        out.swap(tmp);
        tmp = mlp_w_0_ * out + mlp_b_0_;
        out.swap(tmp);
        elu(out);
        tmp = mlp_w_1_ * out + mlp_b_1_;
        out.swap(tmp);
        elu(out);
        tmp = mlp_w_2_ * out + mlp_b_2_;
        out.swap(tmp);
        elu(out);
        tmp = mlp_w_3_ * out + mlp_b_3_;
        out.swap(tmp);
        elu(out);
        tmp = mlp_w_4_ * out + mlp_b_4_;
        out.swap(tmp);
        return out;
    }
private:
    static void elu(Eigen::VectorXf &input) {
        const int len = input.size();
        float *ptr = input.data();
        for (int i = 0; i < len; i++) {
            if (*ptr < 0)
                *ptr = std::exp(*ptr) - 1;
            ptr++;
        }
    }

    Eigen::VectorXf input_mean_;
    Eigen::VectorXf input_inv_std_;
    Eigen::MatrixXf mlp_w_0_;
    Eigen::VectorXf mlp_b_0_;
    Eigen::MatrixXf mlp_w_1_;
    Eigen::VectorXf mlp_b_1_;
    Eigen::MatrixXf mlp_w_2_;
    Eigen::VectorXf mlp_b_2_;
    Eigen::MatrixXf mlp_w_3_;
    Eigen::VectorXf mlp_b_3_;
    Eigen::MatrixXf mlp_w_4_;
    Eigen::VectorXf mlp_b_4_;
};

#endif // EIGEN_MODEL_FROM_FILE_HPP_
