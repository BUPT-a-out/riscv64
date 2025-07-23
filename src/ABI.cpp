#include "ABI.h"
#include <array>
#include <stdexcept>
#include <unordered_map>

namespace riscv64::ABI {

// 整数寄存器ABI名称映射
static const std::unordered_map<std::string, unsigned>& getIntAbiNameToNumMap() {
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
        {"x31", 31}
    };
    return abiNameMap;
}

// 浮点寄存器ABI名称映射
static const std::unordered_map<std::string, unsigned>& getFloatAbiNameToNumMap() {
    static const std::unordered_map<std::string, unsigned> floatNameMap = {
        {"ft0", 0},  {"f0", 0},   {"ft1", 1},  {"f1", 1},   {"ft2", 2},  {"f2", 2},
        {"ft3", 3},  {"f3", 3},   {"ft4", 4},  {"f4", 4},   {"ft5", 5},  {"f5", 5},
        {"ft6", 6},  {"f6", 6},   {"ft7", 7},  {"f7", 7},   {"fs0", 8},  {"f8", 8},
        {"fs1", 9},  {"f9", 9},   {"fa0", 10}, {"f10", 10}, {"fa1", 11}, {"f11", 11},
        {"fa2", 12}, {"f12", 12}, {"fa3", 13}, {"f13", 13}, {"fa4", 14}, {"f14", 14},
        {"fa5", 15}, {"f15", 15}, {"fa6", 16}, {"f16", 16}, {"fa7", 17}, {"f17", 17},
        {"fs2", 18}, {"f18", 18}, {"fs3", 19}, {"f19", 19}, {"fs4", 20}, {"f20", 20},
        {"fs5", 21}, {"f21", 21}, {"fs6", 22}, {"f22", 22}, {"fs7", 23}, {"f23", 23},
        {"fs8", 24}, {"f24", 24}, {"fs9", 25}, {"f25", 25}, {"fs10", 26}, {"f26", 26},
        {"fs11", 27}, {"f27", 27}, {"ft8", 28}, {"f28", 28}, {"ft9", 29}, {"f29", 29},
        {"ft10", 30}, {"f30", 30}, {"ft11", 31}, {"f31", 31}
    };
    return floatNameMap;
}

unsigned getRegNumFromABIName(const std::string& name, bool isFloat) {
    const auto& map = isFloat ? getFloatAbiNameToNumMap() : getIntAbiNameToNumMap();
    auto it = map.find(name);
    if (it != map.end()) {
        return it->second;
    }
    throw std::invalid_argument("Invalid ABI register name: " + name);
}

std::string getABINameFromRegNum(unsigned num, bool isFloat) {
    const size_t maxRegNum = 32;
    
    if (isFloat) {
        // 浮点寄存器名称映射
        static const std::array<std::string, maxRegNum> floatRegNumToName = {
            "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
            "fs0", "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5",
            "fa6", "fa7", "fs2", "fs3", "fs4", "fs5", "fs6", "fs7",
            "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11"
        };
        
        if (num < maxRegNum) {
            return floatRegNumToName.at(num);
        }
    } else {
        // 整数寄存器名称映射
        static const std::array<std::string, maxRegNum> intRegNumToName = {
            "zero", "ra", "sp", "gp", "tp",  "t0",  "t1", "t2", "s0", "s1", "a0",
            "a1",   "a2", "a3", "a4", "a5",  "a6",  "a7", "s2", "s3", "s4", "s5",
            "s6",   "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"
        };
        
        if (num < maxRegNum) {
            return intRegNumToName.at(num);
        }
    }
    
    throw std::out_of_range("Register number " + std::to_string(num) + " out of range");
}

bool isCallerSaved(unsigned physreg, bool isFloat) {
    if (isFloat) {
        // 浮点Caller-saved寄存器: ft0-ft7 (0-7), ft8-ft11 (28-31), fa0-fa7 (10-17)
        return (physreg >= 0 && physreg <= 7) ||   // ft0-ft7
               (physreg >= 10 && physreg <= 17) ||  // fa0-fa7
               (physreg >= 28 && physreg <= 31);    // ft8-ft11
    } else {
        // 整数Caller-saved寄存器: t0-t2 (5-7), t3-t6 (28-31), a0-a7 (10-17), ra (1)
        return (physreg >= 5 && physreg <= 7) ||   // t0-t2
               (physreg >= 10 && physreg <= 17) ||  // a0-a7
               (physreg >= 28 && physreg <= 31) ||  // t3-t6
               (physreg == 1);                      // ra
    }
}

bool isCalleeSaved(unsigned physreg, bool isFloat) {
    if (isFloat) {
        // 浮点Callee-saved寄存器: fs0-fs1 (8-9), fs2-fs11 (18-27)
        return (physreg >= 8 && physreg <= 9) ||   // fs0-fs1
               (physreg >= 18 && physreg <= 27);    // fs2-fs11
    } else {
        // 整数Callee-saved寄存器: s0-s1 (8-9), s2-s11 (18-27), sp (2)
        return (physreg >= 8 && physreg <= 9) ||   // s0-s1
               (physreg >= 18 && physreg <= 27) ||  // s2-s11
               (physreg == 2);                      // sp
    }
}

bool isArgumentReg(unsigned physreg, bool isFloat) {
    if (isFloat) {
        // 浮点参数寄存器: fa0-fa7 (10-17)
        return (physreg >= 10 && physreg <= 17);
    } else {
        // 整数参数寄存器: a0-a7 (10-17)
        return (physreg >= 10 && physreg <= 17);
    }
}

bool isReturnReg(unsigned physreg, bool isFloat) {
    if (isFloat) {
        // 浮点返回值寄存器: fa0-fa1 (10-11)
        return (physreg >= 10 && physreg <= 11);
    } else {
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

}  // namespace riscv64::ABI
