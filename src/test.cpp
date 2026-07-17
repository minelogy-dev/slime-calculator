#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <functional>

using namespace std;

const uint64_t MULT = 0x5DEECE66DULL;
const uint64_t ADD = 0xBULL;
const uint64_t MASK = (1ULL << 48) - 1;
const uint32_t REJECT = 2147483640U;

// 生成所有坏种子（8个坏u值 × 131072个低17位）
vector<uint64_t> generateBadSeeds() {
    vector<uint64_t> seeds;
    seeds.reserve(1048576); // 8 * 131072
    for (uint32_t u = REJECT; u <= 2147483647U; ++u) {
        uint64_t high = (uint64_t)u << 17;
        for (uint32_t low = 0; low < (1U << 17); ++low) {
            seeds.push_back(high | low);
        }
    }
    return seeds;
}

int main() {
    auto badSeeds = generateBadSeeds();
    unordered_set<uint64_t> badSet(badSeeds.begin(), badSeeds.end());
    unordered_map<uint64_t, int> memo;
    memo.reserve(badSeeds.size() * 2);
    function<int(uint64_t)> dfs = [&](uint64_t s) -> int {
        if (badSet.find(s) == badSet.end()) return 0; // 不是坏种子，链断
        auto it = memo.find(s);
        if (it != memo.end()) return it->second;

        uint64_t next_s = (s * MULT + ADD) & MASK;
        int res = 1 + dfs(next_s);
        memo[s] = res;
        return res;
    };

    int maxDepth = 0;
    for (uint64_t s : badSeeds) {
        maxDepth = max(maxDepth, dfs(s));
    }

    cout << "Worst-case number of iterations required for rejection sampling: " << maxDepth << endl;
    return 0;
}
