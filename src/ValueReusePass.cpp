#include "ValueReusePass.h"

#include <algorithm>
#include <memory>
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
    // Use a global liveness set that persists across blocks based on domination
    std::unordered_set<unsigned> globalLiveRegs;
    bool modified = false;
    for (auto& riscv_bb : *riscv_function) {
        if (processBasicBlockGlobal(riscv_bb.get(), globalLiveRegs)) {
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
    std::unordered_set<unsigned> localLiveRegs;
    return processBasicBlockGlobal(riscv_bb, localLiveRegs);
}

bool ValueReusePass::processBasicBlockGlobal(
    BasicBlock* riscv_bb, std::unordered_set<unsigned>& globalLiveRegs) {
    if (riscv_bb == nullptr) {
        return false;
    }

    std::cout << "  Processing basic block: " << riscv_bb->getLabel()
              << std::endl;

    bool modified = false;
    std::vector<BasicBlock::iterator> toErase;

    for (auto it = riscv_bb->begin(); it != riscv_bb->end(); ++it) {
        if (processInstruction(it->get(), globalLiveRegs)) {
            // Mark for potential removal if it's redundant
            if (shouldRemoveInstruction(it->get())) {
                toErase.push_back(it);
            }
            modified = true;
        }
    }

    // Remove optimized instructions marked by processInstruction
    for (auto iter : toErase) {
        std::cout << "    Removing redundant instruction" << std::endl;
        riscv_bb->erase(iter);
    }

    // Remove instructions marked by optimization passes
    if (!instructionsToRemove_.empty()) {
        std::cout << "    Removing " << instructionsToRemove_.size()
                  << " optimized memory load instructions" << std::endl;

        // Remove in reverse order to maintain iterator validity
        for (auto inst : instructionsToRemove_) {
            for (auto it = riscv_bb->begin(); it != riscv_bb->end(); ++it) {
                if (it->get() == inst) {
                    riscv_bb->erase(it);
                    break;
                }
            }
        }
        instructionsToRemove_.clear();
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

    // Parse memory operand to get base register and offset
    auto* memory_operand = dynamic_cast<MemoryOperand*>(operands[1].get());
    if (memory_operand == nullptr) {
        // Not a memory operand, can't optimize
        liveRegs.insert(dest_reg->getRegNum());
        return false;
    }

    unsigned base_reg = memory_operand->getBaseReg()->getRegNum();
    int64_t offset = memory_operand->getOffset()->getValue();

    // Find the corresponding LA instruction that loaded the address
    Instruction* la_inst = findCorrespondingLA(inst, inst->getParent());
    if (la_inst == nullptr) {
        // No corresponding LA instruction found, can't optimize
        std::cout << "    Found memory load from address (base=" << base_reg
                  << ", offset=" << offset << ") into register "
                  << dest_reg->getRegNum()
                  << " - no corresponding LA instruction" << std::endl;
        liveRegs.insert(dest_reg->getRegNum());
        return false;
    }

    // Extract global variable name from LA instruction
    std::string global_var_name = extractGlobalVarFromLA(la_inst);
    if (global_var_name.empty()) {
        // Not a global variable LA instruction, can't optimize
        std::cout << "    Found memory load from address (base=" << base_reg
                  << ", offset=" << offset << ") into register "
                  << dest_reg->getRegNum() << " - not a global variable"
                  << std::endl;
        liveRegs.insert(dest_reg->getRegNum());
        return false;
    }

    std::cout << "    Found global variable load: " << global_var_name
              << " into register " << dest_reg->getRegNum() << std::endl;

    // Check if this global variable has been loaded before
    auto existing_reg_iter = globalVarToFirstReg_.find(global_var_name);
    if (existing_reg_iter != globalVarToFirstReg_.end()) {
        unsigned existing_reg = existing_reg_iter->second;

        // Check if the existing register is still live
        if (liveRegs.count(existing_reg) > 0) {
            // We can reuse the value from the existing register!
            std::cout << "    OPTIMIZING: Global variable " << global_var_name
                      << " already loaded in register " << existing_reg
                      << ", can eliminate load into register "
                      << dest_reg->getRegNum() << std::endl;

            // Generate MOV instruction to copy the existing value to the target
            // register
            auto move_inst =
                std::make_unique<Instruction>(Opcode::MV, inst->getParent());
            move_inst->addOperand(std::make_unique<RegisterOperand>(
                dest_reg->getRegNum(), dest_reg->isVirtual(),
                dest_reg->getRegisterType()));
            move_inst->addOperand(std::make_unique<RegisterOperand>(
                existing_reg, true,
                dest_reg->getRegisterType()));  // Assume source is virtual

            // Insert MOV instruction before the current LW instruction
            auto bb = inst->getParent();
            auto it =
                std::find_if(bb->begin(), bb->end(),
                             [inst](const std::unique_ptr<Instruction>& instr) {
                                 return instr.get() == inst;
                             });
            if (it != bb->end()) {
                bb->insert(it, std::move(move_inst));
                std::cout << "    Generated MOV instruction from register "
                          << existing_reg << " to register "
                          << dest_reg->getRegNum() << std::endl;
            }

            // Mark both LA and LW instructions for removal
            instructionsToRemove_.push_back(la_inst);
            instructionsToRemove_.push_back(inst);
            stats_.loadsEliminated++;
            stats_.virtualRegsReused++;

            liveRegs.insert(dest_reg->getRegNum());
            return true;
        } else {
            // The existing register is no longer live, update the mapping
            std::cout << "    Previous register " << existing_reg
                      << " no longer live, updating mapping" << std::endl;
            globalVarToFirstReg_[global_var_name] = dest_reg->getRegNum();
        }
    } else {
        // First time loading this global variable
        std::cout << "    First load of global variable " << global_var_name
                  << ", recording in register " << dest_reg->getRegNum()
                  << std::endl;
        globalVarToFirstReg_[global_var_name] = dest_reg->getRegNum();
    }

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
    globalVarToFirstReg_.clear();
    laInstToGlobalVar_.clear();
    instructionsToRemove_.clear();
}

std::string ValueReusePass::extractGlobalVarFromLA(Instruction* inst) {
    if (inst == nullptr || inst->getOpcode() != Opcode::LA) {
        return "";
    }

    const auto& operands = inst->getOperands();
    if (operands.size() < 2) {
        return "";
    }

    // Second operand should be a LabelOperand containing the global variable
    // name
    auto* label_operand = dynamic_cast<LabelOperand*>(operands[1].get());
    if (label_operand == nullptr) {
        return "";
    }

    return label_operand->getLabelName();
}

Instruction* ValueReusePass::findCorrespondingLA(Instruction* lw_inst,
                                                 BasicBlock* bb) {
    if (lw_inst == nullptr || bb == nullptr) {
        return nullptr;
    }

    // Get the base register from the LW instruction
    const auto& operands = lw_inst->getOperands();
    if (operands.size() < 2) {
        return nullptr;
    }

    auto* memory_operand = dynamic_cast<MemoryOperand*>(operands[1].get());
    if (memory_operand == nullptr) {
        return nullptr;
    }

    unsigned base_reg = memory_operand->getBaseReg()->getRegNum();

    // Search backwards in the basic block for an LA instruction that loads into
    // base_reg
    auto it = std::find_if(bb->begin(), bb->end(),
                           [lw_inst](const std::unique_ptr<Instruction>& inst) {
                               return inst.get() == lw_inst;
                           });

    if (it == bb->end()) {
        return nullptr;
    }

    // Search backwards from the LW instruction
    for (auto rev_it = std::make_reverse_iterator(it); rev_it != bb->rend();
         ++rev_it) {
        Instruction* candidate = rev_it->get();
        if (candidate->getOpcode() == Opcode::LA) {
            const auto& la_operands = candidate->getOperands();
            if (la_operands.size() >= 2) {
                auto* dest_reg =
                    dynamic_cast<RegisterOperand*>(la_operands[0].get());
                auto* label_operand =
                    dynamic_cast<LabelOperand*>(la_operands[1].get());
                if (dest_reg != nullptr && label_operand != nullptr &&
                    dest_reg->getRegNum() == base_reg) {
                    return candidate;
                }
            }
        }
    }

    return nullptr;
}

}  // namespace riscv64
