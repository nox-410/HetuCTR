#include "partition.h"

#include <bits/stdc++.h>

namespace hetuCTR {

using std::pow;


static float quickpow(float a,int n) {
    float ans = 1, temp = a;
    while(n) {
      if(n&1) ans *= temp;
      n >>= 1;
      temp *= temp;
    }
    return ans;
}

template<typename T>
int argmax(const std::vector<T> &val) {
  int arg_res = 0;
  for (int i = 1 ; i < (int)val.size(); i++)
    if (val[i] > val[arg_res])
      arg_res = i;
  return arg_res;
}

class PartitionFunction {
public:
  PartitionFunction(const py::array_t<int>& _input_data, int _n_part) : n_part_(_n_part) {
    n_data_ = _input_data.shape(0);
    n_slot_ = _input_data.shape(1);
    n_edge_ = n_data_ * n_slot_;
    const int *data = _input_data.data();
    n_embed_ = 0;
    for (int i = 0 ; i < n_edge_; i++) {
      n_embed_ = std::max(n_embed_, data[i]);
    }
    n_embed_++;
    std::vector<int> count(n_embed_);
    data_indptr_.resize(n_data_ + 1);
    embed_indptr_.resize(n_embed_ + 1);
    data_indices_.resize(n_edge_);
    embed_indices_.resize(n_edge_);

    for (int i = 0; i <= n_data_; i++) data_indptr_[i] = i * n_slot_;
    for (int i = 0 ; i < n_edge_; i++) {
      count[data[i]]++;
      data_indices_[i] = data[i];
    }
    for (int i = 1;i <= n_embed_; i++) {
      embed_indptr_[i] = embed_indptr_[i-1] + count[i - 1];
      count[i - 1] = 0;
    }
    assert(embed_indptr_[n_embed_] == n_edge_);
    for (int i = 0 ; i < n_edge_; i++) {
      int data_id = i / n_slot_;
      int embed_id = data[i];
      embed_indices_[embed_indptr_[embed_id] + count[embed_id]] = data_id;
      count[embed_id]++;
    }
  }

  void initResult() {
    res_data_.resize(n_data_);
    res_embed_.resize(n_embed_);
    cnt_data_.resize(n_part_, 0);
    cnt_embed_.resize(n_part_, 0);

    cnt_part_embed_.resize(n_part_, std::vector<int>(n_embed_, 0));

    for (int i = 0; i < n_data_; i++) {
      res_data_[i] = rand() % n_part_;
      cnt_data_[res_data_[i]]++;
      for (int j = data_indptr_[i]; j < data_indptr_[i+1]; j++) {
        cnt_part_embed_[res_data_[i]][data_indices_[j]]++;
      }
    }
    for (int i = 0; i < n_embed_; i++) {
      res_embed_[i] = rand() % n_part_;
      cnt_embed_[res_embed_[i]]++;
    }
  }

  void initSoftLabel() {
    int max = n_data_ / n_part_;
    soft_cnt_.resize(n_data_, 1);
    for (int i = 0; i < max; i++) {
      soft_cnt_[i] = 1 - quickpow(1.0 - (float)i / (float)max, batch_size_);
      soft_cnt_[i] *= (float)max / (float)batch_size_;
    }
  }

  int edgecut() {
    int result = 0;
    for (int i = 0; i < n_data_; i++) {
      for (int j = data_indptr_[i]; j < data_indptr_[i+1]; j++)
        if (res_data_[i] != res_embed_[data_indices_[j]]) result++;
    }
    return result;
  }

  float costModel() {
    std::vector<float> cost(n_part_, 0);
    for (int i = 0; i < n_part_; i++) {
      for (int j = 0; j < n_embed_; j++) {
        if (res_embed_[j] != i) cost[res_embed_[j]] += soft_cnt_[cnt_part_embed_[i][j]];
      }
    }
    float result = 0;
    for (int i = 0; i < n_part_; i++) {
      cost[i] /= (float)n_data_ / (batch_size_ * n_part_);
      result += cost[i];
    }
    std::cout << "Cost : ";
    for (int i = 0; i < n_part_; i++) {
      std::cout << cost[i] << " ";
    }
    std::cout << std::endl;
    return result;
  }

  void refineData() {
    std::vector<float> score(n_part_);
    for (int i = 0; i < n_data_; i++) {
      int old = res_data_[i];
      for (int j = 0; j < n_part_; j++) score[j] = -(float)cnt_data_[j] * 0.1;
      for (int j = data_indptr_[i]; j < data_indptr_[i+1]; j++) {
        int embed_id = data_indices_[j];
        int belong = res_embed_[embed_id];
        score[belong]++;
      }
      int s = argmax(score);
      if (s != old) {
        cnt_data_[old]--;
        cnt_data_[s]++;
        for (int j = data_indptr_[i]; j < data_indptr_[i+1]; j++) {
          cnt_part_embed_[s][data_indices_[j]]++;
          cnt_part_embed_[old][data_indices_[j]]--;
        }
        res_data_[i] = s;
      }
    }
  }

  void refineEmbed() {
    embed_weight_.clear();
    embed_weight_.resize(n_part_, 0);
    for (int i = 0; i < n_part_; i++) {
      for (int j = 0; j < n_embed_; j++) {
        if (res_embed_[j] != i) embed_weight_[res_embed_[j]] += soft_cnt_[cnt_part_embed_[i][j]];
      }
    }
    for (int i = 0; i < n_part_; i++)
      embed_weight_[i] /= (float)n_data_ / (batch_size_ * n_part_);

    std::vector<float> score(n_part_), cnt(n_part_, 0);
    for (int i = 0; i < n_embed_; i++) {
      int old = res_embed_[i];
      for (int j = embed_indptr_[i]; j < embed_indptr_[i+1]; j++) {
        cnt[res_data_[embed_indices_[j]]]++;
      }
      for (int j = 0; j < n_part_; j++) {
        score[j] = soft_cnt_[cnt[j]] - 0.01 * cnt_embed_[j] - 0.01 * embed_weight_[j] * soft_cnt_[embed_indptr_[i+1]-embed_indptr_[i]];
        cnt[j] = 0;
      }
      int s = argmax(score);
      if (s != old) {
        cnt_embed_[old]--;
        cnt_embed_[s]++;
        res_embed_[i] = s;
        for (int j = 0; j < n_part_; j++) {
          if (j != s)
            embed_weight_[s] += soft_cnt_[cnt_part_embed_[j][i]] * (batch_size_ * n_part_) / n_data_;
          if (j != old)
            embed_weight_[old] -= soft_cnt_[cnt_part_embed_[j][i]] * (batch_size_ * n_part_) / n_data_;
        }
      }
    }
  }

  void printBalance() {
    std::cout << "Data : ";
    for (int i = 0; i < n_part_; i++) {
      std::cout << cnt_data_[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Embed : ";
    for (int i = 0; i < n_part_; i++) {
      std::cout << cnt_embed_[i] << " ";
    }
    std::cout << std::endl;
  }

  int n_part_, n_data_, n_slot_, n_edge_, n_embed_;
  int batch_size_ = 128;
  std::vector<int> embed_indptr_, embed_indices_;
  std::vector<int> data_indptr_, data_indices_;
  std::vector<int> res_data_, res_embed_;
  std::vector<int> cnt_data_, cnt_embed_;
  std::vector<float> soft_cnt_, embed_weight_;
  std::vector<std::vector<int>> cnt_part_embed_;
};

py::tuple partition(const py::array_t<int>& _input_data, int n_part) {
  PYTHON_CHECK_ARRAY(_input_data);
  assert(_input_data.ndim() == 2);
  auto func = PartitionFunction(_input_data, n_part);
  func.initSoftLabel();
  func.initResult();
  std::cout << func.costModel() << std::endl;
  for (int i = 0; i < 10; i++) {
    func.refineData();
    func.refineEmbed();
    func.printBalance();
    std::cout << "Refine " << i << " : " << func.costModel() << std::endl;
  }
  return py::make_tuple(bind::vec(func.res_data_), bind::vec(func.res_embed_));
}

}