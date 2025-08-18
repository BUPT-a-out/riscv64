#include "RAGreedy/RegisterRewriter.h"

#include <set>

namespace riscv64 {
void RegisterRewriter::rewrite() {
    mapStackSlotToFrameIndex();
    // 遍历所有基本块
    for (auto& BB : *function) {
        // 第一个循环：处理已经分配到物理寄存器的虚拟寄存器
        for (auto it = BB->begin(); it != BB->end(); ++it) {
            Instruction* inst = it->get();

            for (auto& operand : inst->getOperands()) {
                if (operand->isReg()) {
                    RegisterOperand* regOp =
                        static_cast<RegisterOperand*>(operand.get());
                    unsigned virtReg = regOp->getRegNum();

                    if (VRM->hasPhys(virtReg)) {
                        // 直接替换为物理寄存器
                        unsigned physReg = VRM->getPhys(virtReg);
                        regOp->setPhysicalReg(physReg);
                    }
                } else if (operand->isMem()) {
                    MemoryOperand* memOp =
                        static_cast<MemoryOperand*>(operand.get());
                    // 递归处理内存操作数中的基址寄存器和偏移量
                    if (memOp->getBaseReg()) {
                        RegisterOperand* regOp =
                            static_cast<RegisterOperand*>(memOp->getBaseReg());
                        unsigned virtReg = regOp->getRegNum();

                        if (VRM->hasPhys(virtReg)) {
                            // 直接替换为物理寄存器
                            unsigned physReg = VRM->getPhys(virtReg);
                            regOp->setPhysicalReg(physReg);
                        }
                    }
                }
            }
        }

        // 第二个循环：处理需要spill的虚拟寄存器
        for (auto it = BB->begin(); it != BB->end();) {
            Instruction* inst = it->get();

            // 分析当前指令的操作数
            auto defs = inst->getDefinedIntegerRegs();
            auto uses = inst->getUsedIntegerRegs();
            auto defFloats = inst->getDefinedFloatRegs();
            auto useFloats = inst->getUsedFloatRegs();

            // 收集需要处理的spill寄存器
            std::vector<unsigned> spilledUsedRegs;
            std::vector<unsigned> spilledDefinedRegs;
            bool hasFloatSpill = false;

            for (auto& operand : inst->getOperands()) {
                if (operand->isReg()) {
                    RegisterOperand* regOp =
                        static_cast<RegisterOperand*>(operand.get());
                    unsigned virtReg = regOp->getRegNum();

                    if (VRM->hasStackSlot(virtReg)) {
                        // 检查是否是浮点寄存器
                        if (assigningFloat) hasFloatSpill = true;

                        // 收集使用的spill寄存器
                        if ((assigningFloat &&
                             std::find(useFloats.begin(), useFloats.end(),
                                       virtReg) != useFloats.end()) ||
                            (!assigningFloat &&
                             std::find(uses.begin(), uses.end(), virtReg) !=
                                 uses.end())) {
                            spilledUsedRegs.push_back(virtReg);
                        }

                        // 收集定义的spill寄存器
                        if ((assigningFloat &&
                             std::find(defFloats.begin(), defFloats.end(),
                                       virtReg) != defFloats.end()) ||
                            (!assigningFloat &&
                             std::find(defs.begin(), defs.end(), virtReg) !=
                                 defs.end())) {
                            spilledDefinedRegs.push_back(virtReg);
                        }
                    }
                } else if (operand->isMem()) {
                    MemoryOperand* memOp =
                        static_cast<MemoryOperand*>(operand.get());
                    // 递归处理内存操作数中的基址寄存器和偏移量
                    if (memOp->getBaseReg()) {
                        RegisterOperand* regOp =
                            static_cast<RegisterOperand*>(memOp->getBaseReg());
                        unsigned virtReg = regOp->getRegNum();

                        if (VRM->hasStackSlot(virtReg)) {
                            // 检查是否是浮点寄存器
                            if (assigningFloat) hasFloatSpill = true;

                            // 收集使用的spill寄存器
                            if ((assigningFloat &&
                                 std::find(useFloats.begin(), useFloats.end(),
                                           virtReg) != useFloats.end()) ||
                                (!assigningFloat &&
                                 std::find(uses.begin(), uses.end(), virtReg) !=
                                     uses.end())) {
                                spilledUsedRegs.push_back(virtReg);
                            }

                            // 收集定义的spill寄存器
                            if ((assigningFloat &&
                                 std::find(defFloats.begin(), defFloats.end(),
                                           virtReg) != defFloats.end()) ||
                                (!assigningFloat &&
                                 std::find(defs.begin(), defs.end(), virtReg) !=
                                     defs.end())) {
                                spilledDefinedRegs.push_back(virtReg);
                            }
                        }
                    }
                }
            }

            // 处理使用的spill寄存器（在指令前插入load）
            for (unsigned virtReg : spilledUsedRegs) {
                StackSlot slot = VRM->getStackSlot(virtReg);

                // 获取frame index
                auto fi_id = stackSlot2FrameIndex[slot];

                // 分配数据寄存器和地址寄存器
                unsigned dataReg =
                    selectAvailablePhysicalDataReg(inst, assigningFloat);
                unsigned addrReg = 5;  // t0 用于地址计算

                // 1. 插入 FRAMEADDR 指令
                auto frameAddrInst =
                    std::make_unique<Instruction>(Opcode::FRAMEADDR);
                frameAddrInst->addOperand_(
                    std::make_unique<RegisterOperand>(addrReg, false));
                frameAddrInst->addOperand_(
                    std::make_unique<FrameIndexOperand>(fi_id));
                it = BB->insert(it, std::move(frameAddrInst));
                ++it;

                // 2. 插入 LOAD 指令
                if (assigningFloat) {
                    auto loadInst = std::make_unique<Instruction>(Opcode::FLW);
                    loadInst->addOperand_(
                        std::make_unique<RegisterOperand>(dataReg, false));
                    loadInst->addOperand_(std::make_unique<MemoryOperand>(
                        std::make_unique<RegisterOperand>(addrReg, false),
                        std::make_unique<ImmediateOperand>(0)));
                    it = BB->insert(it, std::move(loadInst));
                    ++it;
                } else {
                    auto loadInst = std::make_unique<Instruction>(Opcode::LD);
                    loadInst->addOperand_(
                        std::make_unique<RegisterOperand>(dataReg, false));
                    loadInst->addOperand_(std::make_unique<MemoryOperand>(
                        std::make_unique<RegisterOperand>(addrReg, false),
                        std::make_unique<ImmediateOperand>(0)));
                    it = BB->insert(it, std::move(loadInst));
                    ++it;
                }

                // 3. 更新原指令中的寄存器引用
                updateRegisterInInstruction(inst, virtReg, dataReg,
                                            assigningFloat);
            }

            ++it;  // 移动到当前指令的下一个位置

            // 处理定义的spill寄存器（在指令后插入store）
            for (unsigned virtReg : spilledDefinedRegs) {
                StackSlot slot = VRM->getStackSlot(virtReg);

                // 获取frame index

                auto fi_id = stackSlot2FrameIndex[slot];

                // 分配数据寄存器和地址寄存器
                unsigned dataReg = assigningFloat ? 33 : 6;  // ft1/t1
                unsigned addrReg = 5;  // t0 用于地址计算

                // 更新原指令中的寄存器引用
                updateRegisterInInstruction(inst, virtReg, dataReg,
                                            assigningFloat);

                // 1. 插入 FRAMEADDR 指令
                auto frameAddrInst =
                    std::make_unique<Instruction>(Opcode::FRAMEADDR);
                frameAddrInst->addOperand_(
                    std::make_unique<RegisterOperand>(addrReg, false));
                frameAddrInst->addOperand_(
                    std::make_unique<FrameIndexOperand>(fi_id));
                it = BB->insert(it, std::move(frameAddrInst));
                ++it;

                // 2. 插入 STORE 指令
                if (assigningFloat) {
                    auto storeInst = std::make_unique<Instruction>(Opcode::FSW);
                    storeInst->addOperand_(
                        std::make_unique<RegisterOperand>(dataReg, false));
                    storeInst->addOperand_(std::make_unique<MemoryOperand>(
                        std::make_unique<RegisterOperand>(addrReg, false),
                        std::make_unique<ImmediateOperand>(0)));
                    it = BB->insert(it, std::move(storeInst));
                    ++it;
                } else {
                    auto storeInst = std::make_unique<Instruction>(Opcode::SD);
                    storeInst->addOperand_(
                        std::make_unique<RegisterOperand>(dataReg, false));
                    storeInst->addOperand_(std::make_unique<MemoryOperand>(
                        std::make_unique<RegisterOperand>(addrReg, false),
                        std::make_unique<ImmediateOperand>(0)));
                    it = BB->insert(it, std::move(storeInst));
                    ++it;
                }
            }
        }
    }
}

// 辅助函数：选择可用的物理寄存器
unsigned RegisterRewriter::selectAvailablePhysicalDataReg(Instruction* inst,
                                                          bool isFloat) {
    // 整数优先使用临时寄存器 t1-t6 (避开t0用于地址计算)
    std::vector<unsigned> tempIntegerPriority = {6,  7,  28,
                                                 29, 30, 31};  // t1-t2, t3-t6

    // 浮点优先使用临时寄存器 ft0-ft11
    std::vector<unsigned> tempFloatPriority = {
        32, 33, 34, 35, 36, 37, 38, 39, 60, 61, 62, 63};  // ft0-ft7, ft8-ft11

    auto regPriority = isFloat ? tempFloatPriority : tempIntegerPriority;
    auto usedInInst =
        isFloat ? inst->getUsedFloatRegs() : inst->getUsedIntegerRegs();

    // 选择第一个不冲突且未使用的物理寄存器
    for (unsigned physReg : regPriority) {
        if (std::find(usedInInst.begin(), usedInInst.end(), physReg) ==
            usedInInst.end()) {
            return physReg;
        }
    }

    // 如果没有可用寄存器，返回默认值
    // 其实不可能
    return isFloat ? 32 : 6;  // ft0 或 t1
}

// 辅助函数：更新指令中的寄存器引用
void RegisterRewriter::updateRegisterInInstruction(Instruction* inst,
                                                   unsigned oldReg,
                                                   unsigned newPhysReg,
                                                   bool isFloat) {
    const auto& operands = inst->getOperands();
    for (const auto& operand : operands) {
        if (operand->isReg() && operand->isFloatRegister() == isFloat) {
            RegisterOperand* regOp =
                static_cast<RegisterOperand*>(operand.get());
            if (regOp->getRegNum() == oldReg) {
                regOp->setPhysicalReg(newPhysReg);
            }
        } else if (operand->isMem()) {
            auto baseReg =
                static_cast<MemoryOperand*>(operand.get())->getBaseReg();
            if (baseReg && baseReg->isReg() &&
                baseReg->isFloatRegister() == isFloat &&
                baseReg->getRegNum() == oldReg) {
                baseReg->setPhysicalReg(newPhysReg);
            }
        }
    }
}

void RegisterRewriter::mapStackSlotToFrameIndex() {
    for (StackSlot s : VRM->getAllStackSlots()) {
        auto sfm = function->getStackFrameManager();
        auto fi_id = sfm->createSpillObject(s);
        stackSlot2FrameIndex[s] = fi_id;
    }
}

unsigned RegisterRewriter::allocateTemporaryRegister(unsigned virtReg) {
    // TODO：分析活跃寄存器，scavenge
    if (assigningFloat) {
        return 32;  // ft0临时浮点寄存器
    } else {
        return 5;  // t0临时整数寄存器
    }
}

void RegisterRewriter::print(std::ostream& OS) const {
    OS << "Register Rewriter for Function: " << function->getName() << "\n";
    OS << "====================================================\n";

    OS << "Rewritten Instructions Preview:\n";
    int instCount = 0;
    for (const auto& BB : *function) {
        OS << "Basic Block " << BB->getLabel() << ":\n";
        for (const auto& inst : *BB) {
            if (instCount++ > 10) {
                OS << "  ... (and more)\n";
                break;
            }
            OS << "  " << inst->toString() << "\n";
        }
        if (instCount > 10) break;
    }
}

// 使用示例
void performRegisterRewriting(Function* function, VirtRegMap* VRM) {
    RegisterRewriter rewriter(function, VRM);

    DEBUG_OUT() << "Starting register rewriting...\n";
    rewriter.rewrite();

    DEBUG_OUT() << "Register rewriting completed.\n";
    rewriter.print(std::cout);
}

}  // namespace riscv64
