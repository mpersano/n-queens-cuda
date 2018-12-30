#include <cassert>
#include <functional>
#include <future>
#include <numeric>
#include <vector>
#include <iostream>
#include <chrono>

extern int count_solutions_cuda(int);

// probably could be optimized if we leveraged symmetry

int count_solutions_sequential(int row, int left, int down, int right, int size)
{
    if (row == size)
    {
        return 1;
    }
    else
    {
        const auto used = ~(left | down | right);
        auto count = 0;
        for (auto bit = 1; bit != 1 << size; bit <<= 1)
        {
            if (bit & used)
                count += count_solutions_sequential(row + 1, (left | bit) << 1, down | bit, (right | bit) >> 1, size);
        }
        return count;
    }
}

int count_solutions_parallel(int row, int left, int down, int right, int size)
{
    constexpr const auto MaxParallelLevel = 1;
    assert(row <= MaxParallelLevel);
    std::vector<std::future<int>> results;
    const auto used = ~(left | down | right);
    const auto job = row < MaxParallelLevel ? count_solutions_parallel : count_solutions_sequential;
    for (auto bit = 1; bit != 1 << size; bit <<= 1)
    {
        if (bit & used)
            results.push_back(std::async(std::launch::async, job, row + 1, (left | bit) << 1, down | bit, (right | bit) >> 1, size));
    }
    return std::accumulate(results.begin(), results.end(), 0, [](auto s, auto &result) { return s + result.get(); });
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "usage: " << argv[0] << " size\n";
        return -1;
    }

    const auto size = std::stoi(argv[1]);

    const auto time = [](const auto fn) {
        const auto start = std::chrono::steady_clock::now();
        const auto result = fn();
        const auto end = std::chrono::steady_clock::now();
        std::cout << result << " solutions (" << std::chrono::duration<double, std::milli>(end - start).count() << " ms)" << "\n";
    };

    std::cout << "sequential: ";
    time([size] { return count_solutions_sequential(0, 0, 0, 0, size); });

    std::cout << "parallel: ";
    time([size] { return count_solutions_parallel(0, 0, 0, 0, size); });

    std::cout << "CUDA: ";
    time([size] { return count_solutions_cuda(size); });
}
