#include <cassert>
#include <functional>

struct state
{
    int row;
    int left;
    int down;
    int right;
};

__global__ void count_solutions(int size, int *count, state *initial_states, int num_initial_states)
{
    const auto thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index >= num_initial_states)
        return;

    constexpr const auto MaxStackSize = 200;
    state stack[MaxStackSize];

    stack[0] = initial_states[thread_index];
    auto stack_top = 1;

    while (stack_top > 0)
    {
        const auto &cur_state = stack[--stack_top];

        const auto row = cur_state.row;
        const auto left = cur_state.left;
        const auto down = cur_state.down;
        const auto right = cur_state.right;

        if (row == size)
        {
            atomicAdd(count, 1);
        }
        else
        {
            const auto used = ~(left | down | right);
            for (auto bit = 1; bit != 1 << size; bit <<= 1)
            {
                if (bit & used)
                {
                    assert(stack_top < MaxStackSize);
                    auto &state = stack[stack_top++];
                    state.row = row + 1;
                    state.left = (left | bit) << 1;
                    state.down = down | bit;
                    state.right = (right | bit) >> 1;
                }
            }
        }
    }
}

int count_solutions_cuda(int size)
{
    constexpr const auto MaxParallelLevel = 5;
    constexpr const auto MaxInitialStates = 20000000;

    state *initial_states;
    cudaMallocManaged(&initial_states, MaxInitialStates * sizeof *initial_states);

    int num_initial_states = 0;

    const std::function<void(int, int, int, int, int)> populate_initial_states =
        [initial_states, &num_initial_states, &populate_initial_states](int row, int left, int down, int right, int size)
        {
            assert(row <= MaxParallelLevel);
            const auto used = ~(left | down | right);
            for (auto bit = 1; bit != 1 << size; bit <<= 1)
            {
                if (bit & used)
                {
                    if (row < MaxParallelLevel)
                    {
                        populate_initial_states(row + 1, (left | bit) << 1, down | bit, (right | bit) >> 1, size);
                    }
                    else
                    {
                        assert(num_initial_states < MaxInitialStates);
                        initial_states[num_initial_states++] = {row + 1, (left | bit) << 1, down | bit, (right | bit) >> 1};
                    }
                }
            }
        };

    populate_initial_states(0, 0, 0, 0, size);

    int *count;
    cudaMallocManaged(&count, sizeof count);
    *count = 0;

    constexpr const auto ThreadsPerBlock = 256;
    const auto num_blocks = (num_initial_states + ThreadsPerBlock - 1) / ThreadsPerBlock;

    count_solutions<<<num_blocks, ThreadsPerBlock>>>(size, count, initial_states, num_initial_states);
    cudaDeviceSynchronize();

    const auto result = *count;

    cudaFree(count);
    cudaFree(initial_states);

    return result;
}
