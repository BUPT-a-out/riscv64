// DeadCodeElimination.cpp
// Implementation of a simple post-regalloc dead code elimination pass.

#include "DeadCodeElimination.h"

#include <algorithm>
#include <iostream>
#include <list>
#include <stack>

namespace riscv64 {

bool DeadCodeElimination::runOnFunction(Function* func, bool forPhys) {
    if (func == nullptr || func->empty()) {
        return false;
    }

    info_.clear();
    computeDefUse(func, forPhys);
    livenessFixpoint(func, forPhys);
    bool changed = eliminate(func, forPhys);
    return changed;
}

void DeadCodeElimination::computeDefUse(Function* func, bool forPhys) {
    for (auto& bbPtr : *func) {
        auto* basicBlock = bbPtr.get();
        auto& blk = info_[basicBlock];
        blk.use.clear();
        blk.def.clear();

        // Iterate instructions in forward order to record first-use-before-def.
        for (const auto& instPtr : *basicBlock) {
            Instruction* inst = instPtr.get();
            // Uses
            for (unsigned usedReg : getUsedRegs(inst)) {
                if (!forPhys && !isVirtualReg(usedReg)) {
                    continue;  // ignore physical
                }
                if (forPhys && (isVirtualReg(usedReg) || usedReg <= 4 || usedReg == 8)) {
                    continue;  // ignore virtual or preserved reg
                }
                if (blk.def.find(usedReg) == blk.def.end()) {
                    blk.use.insert(usedReg);
                }
            }
            // Defs
            for (unsigned defReg : getDefinedRegs(inst)) {
                if (!forPhys && !isVirtualReg(defReg)) {
                    continue;
                }
                if (forPhys && (isVirtualReg(defReg) || defReg <= 4)) {
                    continue;
                }
                blk.def.insert(defReg);
            }
        }
    }
}

void DeadCodeElimination::livenessFixpoint(Function* func, bool forPhys) {
    // Initialize liveIn/liveOut empty vectors implicitly by map default.
    // Compute a post-order then iterate in reverse (RPO) until convergence.
    auto postOrder = func->getPostOrder();
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = postOrder.rbegin(); it != postOrder.rend(); ++it) {
            BasicBlock* basicBlock = *it;
            auto& blk = info_[basicBlock];

            // liveOut = union of successors' liveIn
            std::unordered_set<unsigned> newLiveOut;
            auto successors = basicBlock->getSuccessors();
            if (successors.empty()) {
                if (forPhys) {
                    newLiveOut.insert(10);// Exit block: 10 (a0) is used.
                    newLiveOut.insert(42);// fa0

                    newLiveOut.insert(9); // s1
                    newLiveOut.insert(41); // fs1

                    for (int i = 18; i <= 27; i++) { // s2-s11
                        newLiveOut.insert(i); // 
                    }

                    for (int i = 40; i <= 59; i++) {
                        newLiveOut.insert(i); // fs2-fs11
                    }
                }
            }
            for (BasicBlock* succ : successors) {
                if (succ == nullptr) {
                    continue;
                }
                auto& succBlk = info_[succ];
                newLiveOut.insert(succBlk.liveIn.begin(), succBlk.liveIn.end());
            }

            // liveIn = use U (liveOut - def)
            std::unordered_set<unsigned> newLiveIn = blk.use;  // copy
            for (unsigned outReg : newLiveOut) {
                if (blk.def.find(outReg) == blk.def.end()) {
                    newLiveIn.insert(outReg);
                }
            }

            if (newLiveOut != blk.liveOut || newLiveIn != blk.liveIn) {
                blk.liveOut.swap(newLiveOut);
                blk.liveIn.swap(newLiveIn);
                changed = true;
            }
        }
    }
}

bool DeadCodeElimination::eliminate(Function* func, bool forPhys) {
    bool changed = false;
    // For each block, walk instructions backwards tracking per-instruction live
    for (auto& bbPtr : *func) {
        auto* basicBlock = bbPtr.get();
        auto& blk = info_[basicBlock];
        std::unordered_set<unsigned> live = blk.liveOut;  // live after last

        std::vector<Instruction*> toRemove;  // collect first
        for (auto it = basicBlock->rbegin(); it != basicBlock->rend(); ++it) {
            Instruction* inst = it->get();

            // Gather defs & uses
            auto defs = getDefinedRegs(inst);
            auto uses = getUsedRegs(inst);

            bool removable = false;
            if (!defs.empty() && defs.size() == 1) {  // single-def instruction
                unsigned definedReg = defs[0];
                // s0 and reserved
                if ((forPhys && !isVirtualReg(definedReg) && definedReg >= 5 && definedReg != 8) ||
                    (!forPhys && isVirtualReg(definedReg))) {
                    if (live.find(definedReg) == live.end() &&
                        !hasSideEffects(inst)) {
                        removable = true;
                    }
                }
            }

            if (removable) {
                toRemove.push_back(inst);
            } else {
                // Normal liveness update: first remove defs from live, then add
                // uses
                for (unsigned definedReg : defs) {
                    // kill after definition
                    live.erase(definedReg);
                }
                for (unsigned usedReg : uses) {
                    live.insert(usedReg);
                }
            }
        }

        if (!toRemove.empty()) {
            changed = true;
            // Physically remove (forward iteration w/ erase)
            for (Instruction* inst : toRemove) {
                DCE_DEBUG()
                    << "[DCE] Removing dead instruction: " << inst->toString()
                    << "\n";
                basicBlock->removeInstruction(inst);
            }
        }
    }
    return changed;
}

std::vector<unsigned> DeadCodeElimination::getDefinedRegs(
    Instruction* inst) const {
    std::vector<unsigned> out;
    if (inst == nullptr) {
        return out;
    }
    // Integer + Float defs
    auto ints = inst->getDefinedIntegerRegs();
    out.insert(out.end(), ints.begin(), ints.end());
    auto flts = inst->getDefinedFloatRegs();
    out.insert(out.end(), flts.begin(), flts.end());
    return out;
}

std::vector<unsigned> DeadCodeElimination::getUsedRegs(
    Instruction* inst) const {
    std::vector<unsigned> out;
    if (inst == nullptr) {
        return out;
    }

    auto ints = inst->getUsedIntegerRegs();
    out.insert(out.end(), ints.begin(), ints.end());
    auto flts = inst->getUsedFloatRegs();
    out.insert(out.end(), flts.begin(), flts.end());
    return out;
}

bool DeadCodeElimination::hasSideEffects(Instruction* inst) const {
    if (inst == nullptr) {
        return true;  // be conservative
    }
    auto opcodeVal = inst->getOpcode();
    // Any control-flow, call, system, or store considered side-effecting.
    switch (opcodeVal) {
        case CALL:
        case TAIL:
        case RET:
        case JAL:
        case JALR:
        case JR:
        case J:
        case BEQ:
        case BNE:
        case BLT:
        case BGE:
        case BLTU:
        case BGEU:
        case BEQZ:
        case BNEZ:
        case BLEZ:
        case BGEZ:
        case BLTZ:
        case BGTZ:
        case BGT:
        case BLE:
        case BGTU:
        case BLEU:
        case ECALL:
        case EBREAK:
        case FENCE:
        case SB:
        case SH:
        case SW:
        case SD:
        case FSW:
        case FSD:
            return true;
        default:
            break;
    }
    // Treat pseudo COPY as not side-effecting (can be removed if dead).
    return false;
}

}  // namespace riscv64
