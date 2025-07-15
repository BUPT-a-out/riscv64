#include "ABI.h"

#include <array>
#include <stdexcept>
#include <unordered_map>

namespace riscv64::ABI {

// 使用一个静态函数来封装 map 的创建和返回，确保只初始化一次。
static const std::unordered_map<std::string, unsigned>& getAbiNameToNumMap() {
    // 使用 static 变量，该 map 将在首次调用时被创建和填充，且仅此一次。
    static const std::unordered_map<std::string, unsigned> abiNameMap = {
        {"zero", 0}, {"x0", 0},   {"ra", 1},   {"x1", 1},   {"sp", 2},
        {"x2", 2},   {"gp", 3},   {"x3", 3},   {"tp", 4},   {"x4", 4},
        {"t0", 5},   {"x5", 5},   {"t1", 6},   {"x6", 6},   {"t2", 7},
        {"x7", 7},   {"s0", 8},   {"x8", 8},   {"fp", 8},  // fp 是 s0 的别名
        {"s1", 9},   {"x9", 9},   {"a0", 10},  {"x10", 10}, {"a1", 11},
        {"x11", 11}, {"a2", 12},  {"x12", 12}, {"a3", 13},  {"x13", 13},
        {"a4", 14},  {"x14", 14}, {"a5", 15},  {"x15", 15}, {"a6", 16},
        {"x16", 16}, {"a7", 17},  {"x17", 17}, {"s2", 18},  {"x18", 18},
        {"s3", 19},  {"x19", 19}, {"s4", 20},  {"x20", 20}, {"s5", 21},
        {"x21", 21}, {"s6", 22},  {"x22", 22}, {"s7", 23},  {"x23", 23},
        {"s8", 24},  {"x24", 24}, {"s9", 25},  {"x25", 25}, {"s10", 26},
        {"x26", 26}, {"s11", 27}, {"x27", 27}, {"t3", 28},  {"x28", 28},
        {"t4", 29},  {"x29", 29}, {"t5", 30},  {"x30", 30}, {"t6", 31},
        {"x31", 31}};
    return abiNameMap;
}

unsigned getRegNumFromABIName(const std::string& name) {
    const auto& map = getAbiNameToNumMap();
    auto it = map.find(name);
    if (it != map.end()) {
        return it->second;
    }
    throw std::invalid_argument("Invalid ABI register name: " + name);
}
std::string getABINameFromRegNum(unsigned num) {
    // 对于从数字到名称的映射，一个简单的静态数组是最高效的。
    const size_t maxRegNum = 32;
    static const std::array<std::string, maxRegNum> regNumToName = {
        "zero", "ra", "sp", "gp", "tp",  "t0",  "t1", "t2", "s0", "s1", "a0",
        "a1",   "a2", "a3", "a4", "a5",  "a6",  "a7", "s2", "s3", "s4", "s5",
        "s6",   "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"};

    if (num < maxRegNum) {
        return regNumToName.at(num);
    }
    throw std::out_of_range("Register number " + std::to_string(num) + " out of range");
}

}  // namespace riscv64::ABI