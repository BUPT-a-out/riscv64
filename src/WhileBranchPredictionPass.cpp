#include "WhileBranchPredictionPass.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <utility>

#include "Instructions/BasicBlock.h"
#include "Instructions/Instruction.h"
#include "Instructions/MachineOperand.h"
#include "Visit.h"

// Debug output macro - only outputs when A_OUT_DEBUG is defined
#ifdef A_OUT_DEBUG
#define DEBUG_OUT() std::cout
#else
#define DEBUG_OUT() \
    if constexpr (false) std::cout
#endif

namespace riscv64 {

void WhileBranchPredictionPass::runOnFunction(
    Function* func, const midend::Function* mid_func,
    const midend::LoopInfo* loopInfo) {
    Visitor::createCFG(func);

    if (!func || !mid_func || !loopInfo || func->empty()) return;

    // Collect raw pointer order so we can reorder later.
    std::vector<BasicBlock*> order;
    for (auto& bb_ptr : *func) order.push_back(bb_ptr.get());

    bool changed = false;
    for (size_t idx = 0; idx < order.size(); ++idx) {
        BasicBlock* bb = order[idx];
        if (bb->size() < 2) continue;

        // Map backend BB to midend BB; if fails, skip.
        midend::BasicBlock* mid_bb = nullptr;
        try {
            mid_bb = func->getBasicBlock(bb);
        } catch (...) {
            continue;
        }

        // Check if this mid_bb is a loop header (in some loop) and has 2
        // successors.
        auto* parentLoop = loopInfo->getLoopFor(mid_bb);
        if (!parentLoop || !loopInfo->isLoopHeader(mid_bb)) continue;

        auto succs = mid_bb->getSuccessors();
        if (succs.size() != 2) continue;  // need exactly two successors

        // Classify continuation vs exit: successor inside same loop vs outside.
        midend::BasicBlock* cont_mid = nullptr;
        midend::BasicBlock* exit_mid = nullptr;
        for (auto* s : succs) {
            if (parentLoop->contains(s))
                cont_mid = s;
            else
                exit_mid = s;
        }
        if (!cont_mid || !exit_mid) continue;  // ambiguous
        if (!parentLoop->contains(cont_mid) ||
            !parentLoop->contains(exit_mid)) {
            // Both successors are inside the loop, skip.
            continue;
        }

        // Get last two instructions.
        auto rit = bb->rbegin();
        Instruction* last = rit->get();
        ++rit;
        Instruction* secondLast = rit->get();

        // We want pattern: secondLast is simple conditional (BNEZ etc), last is
        // J.
        if (last->getOpcode() != Opcode::J) continue;

        Opcode condOp = secondLast->getOpcode();
        bool isSimpleCond = (condOp == Opcode::BNEZ || condOp == Opcode::BEQZ ||
                             condOp == Opcode::BLEZ || condOp == Opcode::BGEZ ||
                             condOp == Opcode::BLTZ || condOp == Opcode::BGTZ);
        if (!isSimpleCond) continue;
        // Extract labels.
        auto* condTargetOp =
            dynamic_cast<LabelOperand*>(secondLast->getOperand(1));
        auto* jTargetOp = dynamic_cast<LabelOperand*>(last->getOperand(0));
        if (!condTargetOp || !jTargetOp) continue;
        std::string condLabel =
            condTargetOp->getLabelName();  // current branch target
        std::string jLabel = jTargetOp->getLabelName();  // current jump target

        // Map labels back to backend BB pointers.
        BasicBlock* condTargetBB = nullptr;
        BasicBlock* jumpTargetBB = nullptr;
        for (auto* candidate : order) {
            if (candidate->getLabel() == condLabel)
                condTargetBB = candidate;
            else if (candidate->getLabel() == jLabel)
                jumpTargetBB = candidate;
        }
        if (!condTargetBB || !jumpTargetBB) continue;

        // Determine which backend BB corresponds to continuation mid BB.
        BasicBlock* loopBodyBB = nullptr;
        BasicBlock* exitBB = nullptr;
        try {
            if (func->getBasicBlock(condTargetBB) == cont_mid)
                loopBodyBB = condTargetBB;
            if (func->getBasicBlock(condTargetBB) == exit_mid)
                exitBB = condTargetBB;
        } catch (...) {
        }
        try {
            if (func->getBasicBlock(jumpTargetBB) == cont_mid)
                loopBodyBB = jumpTargetBB;
            if (func->getBasicBlock(jumpTargetBB) == exit_mid)
                exitBB = jumpTargetBB;
        } catch (...) {
        }
        if (!loopBodyBB || !exitBB) continue;  // pattern mismatch

        // Invert the branch and retarget to merge label.
        Opcode newOp = invertSimpleBranch(condOp);
        if (newOp == condOp) continue;  // unsupported inversion

        // After inversion, branch should target exitBB.
        secondLast->setOpcode(newOp);
        secondLast->setOperand(
            1, std::make_unique<LabelOperand>(exitBB->getLabel()));

        // Remove the unconditional jump instruction.
        // Iterate to find and erase.
        for (auto& inst : *bb) {
            if (inst.get() == last) {
                // bb->erase(it);
                inst->setOperand(
                    0, std::make_unique<LabelOperand>(loopBodyBB->getLabel()));
                break;
            }
        }

        // Reorder basic blocks so that loopBB comes immediately after bb.
        if (idx + 1 >= order.size() || order[idx + 1] != loopBodyBB) {
            // Remove loopBB from current position and insert.
            auto itFound = std::find(order.begin(), order.end(), loopBodyBB);
            if (itFound != order.end()) {
                order.erase(itFound);
                order.insert(order.begin() + static_cast<long>(idx + 1),
                             loopBodyBB);
            }
        }

        changed = true;
        DEBUG_OUT() << "[WhileBranchPrediction] Loop header '" << bb->getLabel()
                    << "': fallthrough -> '" << loopBodyBB->getLabel()
                    << "', branch -> '" << exitBB->getLabel() << "'"
                    << std::endl;
    }

    if (!changed) return;

    // Rebuild function basic_blocks order following new 'order'.
    auto& blocks = func->basic_blocks;  // friend access
    std::map<BasicBlock*, std::unique_ptr<BasicBlock>> temp;
    for (auto& up : blocks) temp[up.get()] = std::move(up);
    blocks.clear();
    for (auto* b : order) {
        auto it = temp.find(b);
        if (it != temp.end()) blocks.push_back(std::move(it->second));
    }
}

}  // namespace riscv64
