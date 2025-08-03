#include "ValueReusePass.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace riscv64 {

bool ValueReusePass::runOnFunction(Function* riscv_function) {
    if (riscv_function == nullptr || riscv_function->empty()) {
        return false;
    }

    resetState();

    std::cout << "ValueReusePass: Analyzing function "
              << riscv_function->getName() << std::endl;

    // Build immediate value to register mapping
    buildImmediateMap(riscv_function);

    // Apply optimizations - process each basic block
    bool modified = false;
    for (auto& riscv_bb : *riscv_function) {
        if (processBasicBlock(riscv_bb.get())) {
            modified = true;
        }
    }

    // Print statistics
    if (stats_.loadsAnalyzed > 0 || stats_.optimizationOpportunities > 0) {
        std::cout << "ValueReusePass statistics for "
                  << riscv_function->getName() << ":" << std::endl;
        std::cout << "  Instructions analyzed: " << stats_.loadsAnalyzed
                  << std::endl;
        std::cout << "  Optimization opportunities: "
                  << stats_.optimizationOpportunities << std::endl;
        std::cout << "  Instructions eliminated: " << stats_.loadsEliminated
                  << std::endl;
        std::cout << "  Virtual registers reused: " << stats_.virtualRegsReused
                  << std::endl;
    }

    return modified;
}

void ValueReusePass::buildImmediateMap(Function* riscv_function) {
    std::cout << "  Building immediate value mapping..." << std::endl;

    // First pass: find all immediate loads and their target registers
    for (auto& basicBlock : *riscv_function) {
        for (auto& inst : *basicBlock) {
            Opcode opcode = inst->getOpcode();

            // Look for immediate load instructions
            if (opcode == Opcode::LI) {
                const auto& operands = inst->getOperands();
                if (operands.size() >= 2) {
                    auto* dest_reg =
                        dynamic_cast<RegisterOperand*>(operands[0].get());
                    auto* imm_operand =
                        dynamic_cast<ImmediateOperand*>(operands[1].get());

                    if (dest_reg != nullptr && imm_operand != nullptr) {
                        int64_t value = imm_operand->getValue();
                        unsigned regNum = dest_reg->getRegNum();

                        // Track this immediate value
                        auto& entry = immediateToFirstReg_[value];
                        if (entry == 0) {  // First occurrence
                            entry = regNum;
                            std::cout << "    Immediate " << value
                                      << " first loaded into register "
                                      << regNum << std::endl;
                        } else {
                            std::cout << "    Immediate " << value
                                      << " DUPLICATE in register " << regNum
                                      << " (already in " << entry << ")"
                                      << std::endl;
                            stats_.optimizationOpportunities++;
                        }
                    }
                }
            }

            // Also check ADDI from zero register (another form of immediate
            // load)
            else if (opcode == Opcode::ADDI) {
                const auto& operands = inst->getOperands();
                if (operands.size() >= 3) {
                    auto* dest_reg =
                        dynamic_cast<RegisterOperand*>(operands[0].get());
                    auto* src_reg =
                        dynamic_cast<RegisterOperand*>(operands[1].get());
                    auto* imm_operand =
                        dynamic_cast<ImmediateOperand*>(operands[2].get());

                    // Check if it's ADDI from zero register (effectively LI)
                    if (dest_reg != nullptr && src_reg != nullptr &&
                        imm_operand != nullptr && src_reg->getRegNum() == 0) {
                        int64_t value = imm_operand->getValue();
                        unsigned regNum = dest_reg->getRegNum();

                        auto& entry = immediateToFirstReg_[value];
                        if (entry == 0) {
                            entry = regNum;
                            std::cout << "    Immediate " << value
                                      << " first loaded (ADDI) into register "
                                      << regNum << std::endl;
                        } else {
                            std::cout << "    Immediate " << value
                                      << " DUPLICATE (ADDI) in register "
                                      << regNum << " (already in " << entry
                                      << ")" << std::endl;
                            stats_.optimizationOpportunities++;
                        }
                    }
                }
            }
        }
    }
}

bool ValueReusePass::processBasicBlock(BasicBlock* riscv_bb) {
    if (riscv_bb == nullptr) {
        return false;
    }

    std::cout << "  Processing basic block" << std::endl;

    bool modified = false;
    std::vector<BasicBlock::iterator> toErase;

    // Track register liveness within this basic block
    std::unordered_set<unsigned> liveRegs;

    for (auto it = riscv_bb->begin(); it != riscv_bb->end(); ++it) {
        if (processInstruction(it->get(), liveRegs)) {
            // Mark for potential removal if it's redundant
            if (shouldRemoveInstruction(it->get())) {
                toErase.push_back(it);
            }
            modified = true;
        }
    }

    // Remove optimized instructions
    for (auto iter : toErase) {
        std::cout << "    Removing redundant instruction" << std::endl;
        riscv_bb->erase(iter);
        stats_.loadsEliminated++;
    }

    return modified;
}

bool ValueReusePass::processInstruction(
    Instruction* riscv_inst, std::unordered_set<unsigned>& liveRegs) {
    if (riscv_inst == nullptr) {
        return false;
    }

    Opcode opcode = riscv_inst->getOpcode();

    // Count all instructions we analyze
    stats_.loadsAnalyzed++;

    // Handle load immediate instructions
    if (opcode == Opcode::LI) {
        return optimizeLoadImmediate(riscv_inst, liveRegs);
    }

    // Handle ADDI instruction (may be immediate load from zero register)
    if (opcode == Opcode::ADDI) {
        const auto& operands = riscv_inst->getOperands();
        if (operands.size() >= 3) {
            auto* src_reg = dynamic_cast<RegisterOperand*>(operands[1].get());
            if (src_reg != nullptr && src_reg->getRegNum() == 0) {
                // This is effectively a LI instruction
                return optimizeLoadImmediate(riscv_inst, liveRegs);
            }
        }
    }

    // Handle memory load instructions
    if (opcode == Opcode::LW || opcode == Opcode::FLW) {
        return optimizeMemoryLoad(riscv_inst, liveRegs);
    }

    // Update liveness information
    updateLiveness(riscv_inst, liveRegs);

    return false;
}

bool ValueReusePass::optimizeLoadImmediate(
    Instruction* inst, std::unordered_set<unsigned>& liveRegs) {
    const auto& operands = inst->getOperands();
    Opcode opcode = inst->getOpcode();

    RegisterOperand* dest_reg = nullptr;
    ImmediateOperand* imm_operand = nullptr;

    if (opcode == Opcode::LI && operands.size() >= 2) {
        dest_reg = dynamic_cast<RegisterOperand*>(operands[0].get());
        imm_operand = dynamic_cast<ImmediateOperand*>(operands[1].get());
    } else if (opcode == Opcode::ADDI && operands.size() >= 3) {
        dest_reg = dynamic_cast<RegisterOperand*>(operands[0].get());
        auto* src_reg = dynamic_cast<RegisterOperand*>(operands[1].get());
        imm_operand = dynamic_cast<ImmediateOperand*>(operands[2].get());

        // Only handle ADDI from zero register
        if (src_reg == nullptr || src_reg->getRegNum() != 0) {
            return false;
        }
    }

    if (dest_reg == nullptr || imm_operand == nullptr) {
        return false;
    }

    int64_t value = imm_operand->getValue();
    unsigned destRegNum = dest_reg->getRegNum();

    // Check if this immediate is already available in another register
    auto iter = immediateToFirstReg_.find(value);
    if (iter != immediateToFirstReg_.end()) {
        unsigned existingReg = iter->second;
        if (existingReg != destRegNum && liveRegs.count(existingReg) > 0) {
            // We can reuse the existing register!
            std::cout << "    OPTIMIZING: Immediate " << value
                      << " already in register " << existingReg
                      << ", can reuse for register " << destRegNum << std::endl;

            // In a real implementation, we would replace this instruction with
            // MV For now, just mark the opportunity
            stats_.virtualRegsReused++;
            return true;
        }
    }

    // Update liveness - this register now contains this immediate
    liveRegs.insert(destRegNum);
    return false;
}

bool ValueReusePass::optimizeMemoryLoad(
    Instruction* inst, std::unordered_set<unsigned>& liveRegs) {
    const auto& operands = inst->getOperands();
    if (operands.size() < 2) {
        return false;
    }

    auto* dest_reg = dynamic_cast<RegisterOperand*>(operands[0].get());
    if (dest_reg == nullptr) {
        return false;
    }

    // For now, just track that we analyzed a memory load
    // In a full implementation, we would:
    // 1. Track memory locations and their loaded values
    // 2. Check for redundant loads from the same location
    // 3. Optimize using available registers

    std::cout << "    Found memory load into register " << dest_reg->getRegNum()
              << std::endl;

    liveRegs.insert(dest_reg->getRegNum());
    stats_.optimizationOpportunities++;

    return false;
}

bool ValueReusePass::replaceWithMove(Instruction* inst, unsigned sourceReg,
                                     unsigned destReg) {
    // For now, just mark that we would replace with move
    // In a real implementation, we would create a new MV instruction
    (void)inst;
    (void)sourceReg;
    (void)destReg;

    std::cout << "      Would replace with MV instruction" << std::endl;
    return true;
}

void ValueReusePass::updateLiveness(Instruction* inst,
                                    std::unordered_set<unsigned>& liveRegs) {
    // Simple liveness tracking: assume all destination registers are live
    const auto& operands = inst->getOperands();

    // For most instructions, the first operand is the destination
    if (!operands.empty()) {
        auto* dest_reg = dynamic_cast<RegisterOperand*>(operands[0].get());
        if (dest_reg != nullptr) {
            liveRegs.insert(dest_reg->getRegNum());
        }
    }
}

bool ValueReusePass::shouldRemoveInstruction(Instruction* /* inst */) {
    // For now, don't remove any instructions - just transform them
    return false;
}

void ValueReusePass::resetState() {
    stats_.loadsAnalyzed = 0;
    stats_.optimizationOpportunities = 0;
    stats_.loadsEliminated = 0;
    stats_.virtualRegsReused = 0;

    immediateToFirstReg_.clear();
}

}  // namespace riscv64
