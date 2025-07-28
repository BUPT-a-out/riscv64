#include "ABI.h"

#include <array>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace riscv64::ABI {

static const std::unordered_map<std::string, unsigned>& getAbiNameToNumMap() {
    static const std::unordered_map<std::string, unsigned> abiNameMap = {
        // 整数寄存器 (0-31)
        {"zero", 0},
        {"x0", 0},
        {"ra", 1},
        {"x1", 1},
        {"sp", 2},
        {"x2", 2},
        {"gp", 3},
        {"x3", 3},
        {"tp", 4},
        {"x4", 4},
        {"t0", 5},
        {"x5", 5},
        {"t1", 6},
        {"x6", 6},
        {"t2", 7},
        {"x7", 7},
        {"s0", 8},
        {"x8", 8},
        {"fp", 8},  // fp 是 s0 的别名
        {"s1", 9},
        {"x9", 9},
        {"a0", 10},
        {"x10", 10},
        {"a1", 11},
        {"x11", 11},
        {"a2", 12},
        {"x12", 12},
        {"a3", 13},
        {"x13", 13},
        {"a4", 14},
        {"x14", 14},
        {"a5", 15},
        {"x15", 15},
        {"a6", 16},
        {"x16", 16},
        {"a7", 17},
        {"x17", 17},
        {"s2", 18},
        {"x18", 18},
        {"s3", 19},
        {"x19", 19},
        {"s4", 20},
        {"x20", 20},
        {"s5", 21},
        {"x21", 21},
        {"s6", 22},
        {"x22", 22},
        {"s7", 23},
        {"x23", 23},
        {"s8", 24},
        {"x24", 24},
        {"s9", 25},
        {"x25", 25},
        {"s10", 26},
        {"x26", 26},
        {"s11", 27},
        {"x27", 27},
        {"t3", 28},
        {"x28", 28},
        {"t4", 29},
        {"x29", 29},
        {"t5", 30},
        {"x30", 30},
        {"t6", 31},
        {"x31", 31},

        // 浮点寄存器 (32-63)
        {"ft0", 32},
        {"f0", 32},
        {"ft1", 33},
        {"f1", 33},
        {"ft2", 34},
        {"f2", 34},
        {"ft3", 35},
        {"f3", 35},
        {"ft4", 36},
        {"f4", 36},
        {"ft5", 37},
        {"f5", 37},
        {"ft6", 38},
        {"f6", 38},
        {"ft7", 39},
        {"f7", 39},
        {"fs0", 40},
        {"f8", 40},
        {"fs1", 41},
        {"f9", 41},
        {"fa0", 42},
        {"f10", 42},
        {"fa1", 43},
        {"f11", 43},
        {"fa2", 44},
        {"f12", 44},
        {"fa3", 45},
        {"f13", 45},
        {"fa4", 46},
        {"f14", 46},
        {"fa5", 47},
        {"f15", 47},
        {"fa6", 48},
        {"f16", 48},
        {"fa7", 49},
        {"f17", 49},
        {"fs2", 50},
        {"f18", 50},
        {"fs3", 51},
        {"f19", 51},
        {"fs4", 52},
        {"f20", 52},
        {"fs5", 53},
        {"f21", 53},
        {"fs6", 54},
        {"f22", 54},
        {"fs7", 55},
        {"f23", 55},
        {"fs8", 56},
        {"f24", 56},
        {"fs9", 57},
        {"f25", 57},
        {"fs10", 58},
        {"f26", 58},
        {"fs11", 59},
        {"f27", 59},
        {"ft8", 60},
        {"f28", 60},
        {"ft9", 61},
        {"f29", 61},
        {"ft10", 62},
        {"f30", 62},
        {"ft11", 63},
        {"f31", 63}};
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
    const size_t maxRegNum = 64;
    static const std::array<std::string, maxRegNum> regNumToName = {
        // 整数寄存器 (0-31)
        "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0",
        "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s2", "s3", "s4", "s5", "s6",
        "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
        // 浮点寄存器 (32-63)
        "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0", "fs1",
        "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7", "fs2", "fs3",
        "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11", "ft8", "ft9",
        "ft10", "ft11"};

    if (num < maxRegNum) {
        return regNumToName.at(num);
    }
    throw std::out_of_range("Register number " + std::to_string(num) +
                            " out of range");
}

bool isCallerSaved(unsigned physreg, bool isFloat) {
    // 浮点Caller-saved寄存器: ft0-ft7 (32-39), ft8-ft11 (60-63), fa0-fa7
    // (42-49)
    return (physreg >= 32 && physreg <= 39) ||  // ft0-ft7
           (physreg >= 42 && physreg <= 49) ||  // fa0-fa7
           (physreg >= 60 && physreg <= 63)     // ft8-ft11

           // 整数Caller-saved寄存器: t0-t2 (5-7), t3-t6 (28-31), a0-a7 (10-17),
           // ra (1)
           || (physreg >= 5 && physreg <= 7) ||  // t0-t2
           (physreg >= 10 && physreg <= 17) ||   // a0-a7
           (physreg >= 28 && physreg <= 31) ||   // t3-t6
           (physreg == 1);                       // ra
}

bool isCalleeSaved(unsigned physreg, bool isFloat) {
    if (isFloat) {

        
        // 浮点Callee-saved寄存器: fs0-fs1 (40-41), fs2-fs11 (50-59)
        return (physreg >= 40 && physreg <= 41) ||  // fs0-fs1
        (physreg >= 50 && physreg <= 59);     // fs2-fs11
    } else {
        
        // 整数Callee-saved寄存器: s0-s1 (8-9), s2-s11 (18-27), sp (2)
        return (physreg >= 8 && physreg <= 9) ||  // s0-s1
        (physreg >= 18 && physreg <= 27) ||   // s2-s11
        (physreg == 2);                       // sp
    }
}

bool isArgumentReg(unsigned physreg, bool isFloat) {
    if (isFloat) {
        // 浮点参数寄存器: fa0-fa7 (42-49)
        return (physreg >= 42 && physreg <= 49);

    } else {

        // 整数参数寄存器: a0-a7 (10-17)
        return (physreg >= 10 && physreg <= 17);
    }
}

bool isReturnReg(unsigned physreg, bool isFloat) {
    if (isFloat) {
        // 浮点返回值寄存器: fa0-fa1 (42-43)
        return (physreg >= 42 && physreg <= 43);
    }
    else {

        // 整数返回值寄存器: a0-a1 (10-11)
        return (physreg >= 10 && physreg <= 11);
    }
}

bool isReservedReg(unsigned physreg, bool isFloat) {
    if (isFloat) {
        return false;
    } else {
        // x0 (zero), x1 (ra), x2 (sp), x3 (gp), x4 (tp)
        return physreg <= 4;
    }
}

std::vector<unsigned> getCallerSavedRegs(bool isFloat) {
    if (isFloat) {
        return {
            32, 33, 34, 35, 36, 37, 38, 39,  // ft0-ft7
            42, 43, 44, 45, 46, 47, 48, 49,  // fa0-fa7
            60, 61, 62, 63                   // ft8-ft11
        };
    } else {
        return {
            5,  6,  7,                       // t0-t2
            10, 11, 12, 13, 14, 15, 16, 17,  // a0-a7
            28, 29, 30, 31,                  // t3-t6
            1                                // ra
        };
    }
}

}  // namespace riscv64::ABI
