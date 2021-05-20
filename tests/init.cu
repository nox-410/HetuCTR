#include "utils/initializer.h"
#include "thrust/device_vector.h"
#include <vector>

using namespace hetu;

int main(int argc, char* argv[]) {
    int N = atoi(argv[1]);
    unsigned int seed = atoi(argv[2]);
    thrust::device_vector<embed_t> dev(N);
    std::vector<embed_t> h(N);
    Initializer init(InitType::kNormal, 0, 1);
    hetu::initialize(dev.begin().base().get(), dev.size(), init, false, seed);
    std::cout << thrust::reduce(dev.begin(), dev.end()) << std::endl;
    // hetu::initialize(&h[0], dev.size(), init, true);
}
