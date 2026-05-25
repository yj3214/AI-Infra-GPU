#include "model.hpp"

#include <iostream>

int main() {
    std::cout << myai_gpu::version() << '\n';
    std::cout << myai_gpu::hello() << '\n';
    return 0;
}
