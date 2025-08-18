#include "SpillCodeOptimizer.h"

#include <algorithm>
#include <iostream>
#include <set>

// Debug output macro - only outputs when A_OUT_DEBUG is defined
#ifdef A_OUT_DEBUG
#define DEBUG_OUT() std::cerr
#else
#define DEBUG_OUT() \
    if constexpr (false) std::cerr
#endif

namespace riscv64 {

void SpillCodeOptimizer::optimizeSpillCode(Function* function) {
    DEBUG_OUT() << "Starting linear scan spill code optimization..."
                << std::endl;
    for (auto& bb : *function) {
        optimizeBasicBlock(bb.get());
    }
    // TODO: post opt: reduce dead code
    DEBUG_OUT() << "Spill code optimization completed." << std::endl;
}

void SpillCodeOptimizer::optimizeBasicBlock(BasicBlock* bb) {
    DEBUG_OUT() << "Optimizing basic block with " << bb->size()
                << " instructions" << std::endl;
    BasicBlockTracker tracker;

    // 线性扫描基本块中的每条指令
    for (auto it = bb->begin(); it != bb->end(); ++it) {
        Instruction* inst = it->get();
        tracker.currentPos++;

        // 检查是否可以优化当前指令
        if (canOptimizeInstruction(inst, tracker)) {
            applyOptimization(inst, tracker);
        }

        // 根据指令类型更新状态
        if (isFrameAddrInst(inst)) {
            handleFrameAddr(tracker, inst);
        } else if (isLoadFromFI(inst)) {
            handleLoad(tracker, inst);
        } else if (isStoreToFI(inst)) {
            handleStore(tracker, inst);
        } else {
            // 处理其他指令对寄存器的定义
            handleRegisterDef(tracker, inst);
        }
    }
}

// BasicBlockTracker 方法实现
void SpillCodeOptimizer::BasicBlockTracker::updateRegister(unsigned reg,
                                                           RegisterState state,
                                                           unsigned fiId,
                                                           int pos) {
    // 清除该寄存器之前的FI关联
    if (regMap.count(reg)) {
        auto& oldInfo = regMap[reg];
        if (oldInfo.fiId != 0) {
            removeFIRegisterAssociation(oldInfo.fiId, reg, oldInfo.state);
        }
    }

    // 更新寄存器信息
    regMap[reg] = RegisterInfo(state, fiId, pos);

    // 更新FI信息 - 支持多个寄存器
    if (fiId != 0) {
        addFIRegisterAssociation(fiId, reg, state, pos);
    }
}

void SpillCodeOptimizer::BasicBlockTracker::clearRegister(unsigned reg) {
    if (regMap.count(reg)) {
        auto& info = regMap[reg];
        if (info.fiId != 0) {
            removeFIRegisterAssociation(info.fiId, reg, info.state);
        }
        regMap.erase(reg);
    }
}

void SpillCodeOptimizer::BasicBlockTracker::addFIRegisterAssociation(
    unsigned fiId, unsigned reg, RegisterState state, int pos) {
    auto& fiInfo = fiMap[fiId];

    if (state == RegisterState::FI_ADDRESS) {
        fiInfo.addressRegs[reg] = pos;
    } else if (state == RegisterState::FI_VALUE) {
        fiInfo.valueRegs[reg] = pos;
    }
}

void SpillCodeOptimizer::BasicBlockTracker::removeFIRegisterAssociation(
    unsigned fiId, unsigned reg, RegisterState state) {
    if (fiMap.count(fiId)) {
        auto& fiInfo = fiMap[fiId];
        if (state == RegisterState::FI_ADDRESS) {
            fiInfo.addressRegs.erase(reg);
        } else if (state == RegisterState::FI_VALUE) {
            fiInfo.valueRegs.erase(reg);
        }
    }
}

void SpillCodeOptimizer::BasicBlockTracker::clearFIRegister(unsigned fiId,
                                                            unsigned reg) {
    if (fiMap.count(fiId)) {
        auto& info = fiMap[fiId];
        info.addressRegs.erase(reg);
        info.valueRegs.erase(reg);
    }
}

bool SpillCodeOptimizer::BasicBlockTracker::canReuseFIAddress(
    unsigned fiId, unsigned& outReg) {
    if (fiMap.count(fiId) && !fiMap[fiId].addressRegs.empty()) {
        // 选择最近定义的有效地址寄存器
        int latestPos = -1;
        unsigned bestReg = 0;

        for (const auto& pair : fiMap[fiId].addressRegs) {
            unsigned reg = pair.first;
            int pos = pair.second;

            // 检查寄存器是否仍然有效（没有被重新定义）
            if (regMap.count(reg) &&
                regMap[reg].state == RegisterState::FI_ADDRESS &&
                regMap[reg].fiId == fiId && pos > latestPos) {
                bestReg = reg;
                latestPos = pos;
            }
        }

        if (bestReg != 0) {
            outReg = bestReg;
            return true;
        }
    }
    return false;
}

bool SpillCodeOptimizer::BasicBlockTracker::canReuseFIValue(unsigned fiId,
                                                            unsigned& outReg) {
    if (fiMap.count(fiId) && !fiMap[fiId].valueRegs.empty()) {
        // 选择最近定义的有效值寄存器
        int latestPos = -1;
        unsigned bestReg = 0;

        for (const auto& pair : fiMap[fiId].valueRegs) {
            unsigned reg = pair.first;
            int pos = pair.second;

            // 检查寄存器是否仍然有效
            if (regMap.count(reg) &&
                regMap[reg].state == RegisterState::FI_VALUE &&
                regMap[reg].fiId == fiId && pos > latestPos) {
                bestReg = reg;
                latestPos = pos;
            }
        }

        if (bestReg != 0) {
            outReg = bestReg;
            return true;
        }
    }
    return false;
}

// 获取FI的所有有效地址寄存器
std::vector<unsigned> SpillCodeOptimizer::BasicBlockTracker::getFIAddressRegs(
    unsigned fiId) {
    std::vector<unsigned> validRegs;
    if (fiMap.count(fiId)) {
        for (const auto& pair : fiMap[fiId].addressRegs) {
            unsigned reg = pair.first;
            if (regMap.count(reg) &&
                regMap[reg].state == RegisterState::FI_ADDRESS &&
                regMap[reg].fiId == fiId) {
                validRegs.push_back(reg);
            }
        }
    }
    return validRegs;
}

// 获取FI的所有有效值寄存器
std::vector<unsigned> SpillCodeOptimizer::BasicBlockTracker::getFIValueRegs(
    unsigned fiId) {
    std::vector<unsigned> validRegs;
    if (fiMap.count(fiId)) {
        for (const auto& pair : fiMap[fiId].valueRegs) {
            unsigned reg = pair.first;
            if (regMap.count(reg) &&
                regMap[reg].state == RegisterState::FI_VALUE &&
                regMap[reg].fiId == fiId) {
                validRegs.push_back(reg);
            }
        }
    }
    return validRegs;
}

// 指令处理函数
void SpillCodeOptimizer::handleFrameAddr(BasicBlockTracker& tracker,
                                         Instruction* inst) {
    unsigned targetReg = getTargetRegister(inst);
    unsigned fiId = getFrameIndex(inst);

    DEBUG_OUT() << "FRAMEADDR: reg " << targetReg << " gets address of FI "
                << fiId << " at position " << tracker.currentPos << std::endl;

    tracker.updateRegister(targetReg, RegisterState::FI_ADDRESS, fiId,
                           tracker.currentPos);

    // 打印该FI现在拥有的所有地址寄存器
    auto addressRegs = tracker.getFIAddressRegs(fiId);
    if (addressRegs.size() > 1) {
        DEBUG_OUT() << "FI " << fiId << " now has " << addressRegs.size()
                    << " address registers: ";
        for (unsigned reg : addressRegs) {
            DEBUG_OUT() << "r" << reg << " ";
        }
        DEBUG_OUT() << std::endl;
    }
}

void SpillCodeOptimizer::handleLoad(BasicBlockTracker& tracker,
                                    Instruction* inst) {
    unsigned targetReg = getTargetRegister(inst);

    // 获取内存操作数，判断是否从FI加载
    const auto& operands = inst->getOperands();
    if (operands.size() >= 2 && operands[1]->isMem()) {
        auto memOp = static_cast<MemoryOperand*>(operands[1].get());
        if (memOp->getBaseReg() && memOp->getBaseReg()->isReg()) {
            unsigned baseReg = memOp->getBaseReg()->getRegNum();

            // 检查基址寄存器是否持有FI地址
            if (tracker.regMap.count(baseReg) &&
                tracker.regMap[baseReg].state == RegisterState::FI_ADDRESS) {
                unsigned fiId = tracker.regMap[baseReg].fiId;

                DEBUG_OUT() << "LOAD: reg " << targetReg
                            << " gets value from FI " << fiId << " at position "
                            << tracker.currentPos << std::endl;

                tracker.updateRegister(targetReg, RegisterState::FI_VALUE, fiId,
                                       tracker.currentPos);

                // 打印该FI现在拥有的所有值寄存器
                auto valueRegs = tracker.getFIValueRegs(fiId);
                if (valueRegs.size() > 1) {
                    DEBUG_OUT() << "FI " << fiId << " now has "
                                << valueRegs.size() << " value registers: ";
                    for (unsigned reg : valueRegs) {
                        DEBUG_OUT() << "r" << reg << " ";
                    }
                    DEBUG_OUT() << std::endl;
                }
                return;
            }
        }
    }

    // 如果不是从FI加载，标记为OTHER
    tracker.updateRegister(targetReg, RegisterState::OTHER, 0,
                           tracker.currentPos);
}

void SpillCodeOptimizer::handleStore(BasicBlockTracker& tracker,
                                     Instruction* inst) {
    // Store指令不会更新目标寄存器，但会使FI的值状态失效
    const auto& operands = inst->getOperands();
    if (operands.size() >= 2 && operands[1]->isMem()) {
        auto memOp = static_cast<MemoryOperand*>(operands[1].get());
        if (memOp->getBaseReg() && memOp->getBaseReg()->isReg()) {
            unsigned baseReg = memOp->getBaseReg()->getRegNum();

            // 检查是否存储到FI
            if (tracker.regMap.count(baseReg) &&
                tracker.regMap[baseReg].state == RegisterState::FI_ADDRESS) {
                unsigned fiId = tracker.regMap[baseReg].fiId;

                DEBUG_OUT()
                    << "STORE: storing to FI " << fiId << " at position "
                    << tracker.currentPos << std::endl;

                // 使该FI的所有值寄存器失效（因为内存值已改变）
                if (tracker.fiMap.count(fiId)) {
                    auto valueRegs = tracker.getFIValueRegs(fiId);
                    for (unsigned valueReg : valueRegs) {
                        DEBUG_OUT()
                            << "Invalidating value register r" << valueReg
                            << " for FI " << fiId << std::endl;
                        tracker.clearRegister(valueReg);
                    }
                }
            }
        }
    }
}

void SpillCodeOptimizer::handleRegisterDef(BasicBlockTracker& tracker,
                                           Instruction* inst) {
    if (inst->isCopyInstr()) {
        // 获取源寄存器和目标寄存器
        unsigned sourceReg = getSourceRegister(inst);
        unsigned targetReg = getTargetRegister(inst);

        if (sourceReg != 0 && targetReg != 0) {
            // 先清除目标寄存器的旧状态
            tracker.clearRegister(targetReg);

            // 如果源寄存器有有效的状态信息，将其复制到目标寄存器
            if (tracker.regMap.count(sourceReg)) {
                const auto& sourceInfo = tracker.regMap[sourceReg];

                DEBUG_OUT()
                    << "COPY: reg " << targetReg << " inherits state from reg "
                    << sourceReg << " (FI " << sourceInfo.fiId << ", state: ";

                switch (sourceInfo.state) {
                    case RegisterState::FI_ADDRESS:
                        DEBUG_OUT() << "FI_ADDRESS)" << std::endl;
                        tracker.updateRegister(
                            targetReg, RegisterState::FI_ADDRESS,
                            sourceInfo.fiId, tracker.currentPos);
                        break;
                    case RegisterState::FI_VALUE:
                        DEBUG_OUT() << "FI_VALUE)" << std::endl;
                        tracker.updateRegister(
                            targetReg, RegisterState::FI_VALUE, sourceInfo.fiId,
                            tracker.currentPos);
                        break;
                    case RegisterState::OTHER:
                        DEBUG_OUT() << "OTHER)" << std::endl;
                        tracker.updateRegister(targetReg, RegisterState::OTHER,
                                               0, tracker.currentPos);
                        break;
                    case RegisterState::UNKNOWN:
                        break;
                }
            } else {
                // 源寄存器没有跟踪信息，目标寄存器标记为OTHER
                tracker.updateRegister(targetReg, RegisterState::OTHER, 0,
                                       tracker.currentPos);
            }
            return;  // 提前返回，不执行后面的通用处理逻辑
        }
    }

    // 获取指令定义的所有寄存器
    auto definedRegs = inst->getDefinedIntegerRegs();
    auto definedFloatRegs = inst->getDefinedFloatRegs();

    // 清除所有被定义寄存器的状态
    for (unsigned reg : definedRegs) {
        if (reg != 0) {  // x0寄存器除外
            tracker.clearRegister(reg);
        }
    }
    for (unsigned reg : definedFloatRegs) {
        tracker.clearRegister(reg);
    }
}

// 优化检查和应用
bool SpillCodeOptimizer::canOptimizeInstruction(Instruction* inst,
                                                BasicBlockTracker& tracker) {
    if (isFrameAddrInst(inst)) {
        // 检查是否可以复用已有的地址寄存器
        unsigned fiId = getFrameIndex(inst);
        unsigned reuseReg;
        return tracker.canReuseFIAddress(fiId, reuseReg);
    } else if (isLoadFromFI(inst)) {
        // 检查是否可以复用已有的值寄存器
        const auto& operands = inst->getOperands();
        if (operands.size() >= 2 && operands[1]->isMem()) {
            auto memOp = static_cast<MemoryOperand*>(operands[1].get());
            if (memOp->getBaseReg() && memOp->getBaseReg()->isReg()) {
                unsigned baseReg = memOp->getBaseReg()->getRegNum();
                if (tracker.regMap.count(baseReg) &&
                    tracker.regMap[baseReg].state ==
                        RegisterState::FI_ADDRESS) {
                    unsigned fiId = tracker.regMap[baseReg].fiId;
                    unsigned reuseReg;
                    return tracker.canReuseFIValue(fiId, reuseReg);
                }
            }
        }
    }
    return false;
}

void SpillCodeOptimizer::applyOptimization(Instruction* inst,
                                           BasicBlockTracker& tracker) {
    if (isFrameAddrInst(inst)) {
        unsigned fiId = getFrameIndex(inst);
        unsigned reuseReg;
        if (tracker.canReuseFIAddress(fiId, reuseReg)) {
            unsigned targetReg = getTargetRegister(inst);

            auto addressRegs = tracker.getFIAddressRegs(fiId);
            DEBUG_OUT() << "OPTIMIZATION: Reusing address register " << reuseReg
                        << " for FI " << fiId << " instead of reg " << targetReg
                        << " (available: " << addressRegs.size()
                        << " address regs)" << std::endl;

            // 将FRAMEADDR指令替换为MV指令
            replaceWithMoveInstruction(inst, targetReg, reuseReg, false);
        }
    } else if (isLoadFromFI(inst)) {
        const auto& operands = inst->getOperands();
        if (operands.size() >= 2 && operands[1]->isMem()) {
            auto memOp = static_cast<MemoryOperand*>(operands[1].get());
            if (memOp->getBaseReg() && memOp->getBaseReg()->isReg()) {
                unsigned baseReg = memOp->getBaseReg()->getRegNum();
                if (tracker.regMap.count(baseReg) &&
                    tracker.regMap[baseReg].state ==
                        RegisterState::FI_ADDRESS) {
                    unsigned fiId = tracker.regMap[baseReg].fiId;
                    unsigned reuseReg;
                    if (tracker.canReuseFIValue(fiId, reuseReg)) {
                        unsigned targetReg = getTargetRegister(inst);

                        auto valueRegs = tracker.getFIValueRegs(fiId);
                        DEBUG_OUT()
                            << "OPTIMIZATION: Reusing value register "
                            << reuseReg << " for FI " << fiId
                            << " instead of loading to reg " << targetReg
                            << " (available: " << valueRegs.size()
                            << " value regs)" << std::endl;

                        // 判断是整数还是浮点数加载，选择MV或FMV
                        bool isFloatLoad = (inst->getOpcode() == Opcode::FLW);
                        replaceWithMoveInstruction(inst, targetReg, reuseReg,
                                                   isFloatLoad);
                    }
                }
            }
        }
    }
}

// 替换为移动指令
void SpillCodeOptimizer::replaceWithMoveInstruction(Instruction* inst,
                                                    unsigned targetReg,
                                                    unsigned sourceReg,
                                                    bool isFloat) {
    // 清除原有操作数
    inst->clearOperands();

    // 设置新的操作码
    if (isFloat) {
        inst->setOpcode(Opcode::FMOV_S);  // 浮点移动指令
    } else {
        inst->setOpcode(Opcode::MV);  // 整数移动指令
    }

    auto T = isFloat ? RegisterType::Float : RegisterType::Integer;

    // 添加目标寄存器操作数
    auto targetOperand = std::make_unique<RegisterOperand>(targetReg, false, T);
    inst->addOperand(std::move(targetOperand));

    // 添加源寄存器操作数
    auto sourceOperand = std::make_unique<RegisterOperand>(sourceReg, false, T);
    inst->addOperand(std::move(sourceOperand));

    DEBUG_OUT() << "Replaced instruction with " << (isFloat ? "FMV" : "MV")
                << " r" << targetReg << ", r" << sourceReg << std::endl;
}

// 辅助函数实现
bool SpillCodeOptimizer::isFrameAddrInst(Instruction* inst) {
    return inst->getOpcode() == Opcode::FRAMEADDR;
}

bool SpillCodeOptimizer::isLoadFromFI(Instruction* inst) {
    return inst->getOpcode() == Opcode::LD || inst->getOpcode() == Opcode::FLW;
}

bool SpillCodeOptimizer::isStoreToFI(Instruction* inst) {
    return inst->getOpcode() == Opcode::SD || inst->getOpcode() == Opcode::FSW;
}

unsigned SpillCodeOptimizer::getFrameIndex(Instruction* inst) {
    const auto& operands = inst->getOperands();
    if (operands.size() >= 2 && operands[1]->isFrameIndex()) {
        return static_cast<FrameIndexOperand*>(operands[1].get())->getIndex();
    }
    return 0;
}

unsigned SpillCodeOptimizer::getTargetRegister(Instruction* inst) {
    const auto& operands = inst->getOperands();
    if (!operands.empty() && operands[0]->isReg()) {
        return operands[0]->getRegNum();
    }
    return 0;
}

unsigned SpillCodeOptimizer::getSourceRegister(Instruction* inst) {
    const auto& operands = inst->getOperands();
    if (operands.size() >= 2 && operands[1]->isReg()) {
        return operands[1]->getRegNum();
    }
    return 0;
}

}  // namespace riscv64
