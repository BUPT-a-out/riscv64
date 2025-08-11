#include "Instructions/BasicBlock.h"
#include "Visit.h"

namespace riscv64 {

namespace CFG {
BasicBlock* getBBForLabel(const std::string& label, Function* func) {
    for (const auto& bb : *func) {
        if (bb->getLabel() == label) {
            return bb.get();
        }
    }
    throw std::runtime_error("No basic block found for label: " + label);
}

void handleUnconditonalJump(Function* func, BasicBlock* bb, Instruction* inst) {
    BasicBlock* successor = nullptr;
    // 无条件跳转，取第 1 个操作数作为目标基本块
    auto* target = dynamic_cast<LabelOperand*>(inst->getOperand(0));
    if (target == nullptr) {
        throw std::runtime_error("Invalid target for unconditional jump");
    }

    successor = getBBForLabel(target->getLabelName(),
                              func);  // TODO(rikka): 这里或许会有重名的问题
    if (successor == nullptr) {
        throw std::runtime_error("No basic block found for label: " +
                                 target->getLabelName());
    }
    bb->addSuccessor(successor);
    successor->addPredecessor(bb);
}

void handleConditionalJump(Function* func, BasicBlock* bb, Instruction* inst) {
    BasicBlock* successor = nullptr;
    // 条件跳转，取第 1 个操作数作为条件，第 2 个操作数作为目标基本块
    if (inst->getOperand(0)->isImm()) {
        auto* immCondition =
            dynamic_cast<ImmediateOperand*>(inst->getOperand(0));
        if (immCondition->getValue() != 0) {
            // 如果条件是立即数且不为0，直接跳转到目标基本块
            auto* target = dynamic_cast<LabelOperand*>(inst->getOperand(1));
            if (target == nullptr) {
                throw std::runtime_error(
                    "Invalid target for unconditional jump");
            }
            successor = getBBForLabel(target->getLabelName(), func);
            bb->addSuccessor(successor);
            successor->addPredecessor(bb);
            return;
        }
    }

    auto* condition = dynamic_cast<RegisterOperand*>(inst->getOperand(0));
    auto* target = dynamic_cast<LabelOperand*>(inst->getOperand(1));
    if (condition == nullptr || target == nullptr) {
        throw std::runtime_error("Invalid operands for conditional branch");
    }

    successor = getBBForLabel(target->getLabelName(), func);
    if (successor == nullptr) {
        throw std::runtime_error("No basic block found for label: " +
                                 target->getLabelName());
    }
    bb->addSuccessor(successor);
    successor->addPredecessor(bb);
}

void handleComparisonJump(Function* func, BasicBlock* bb, Instruction* inst) {
    BasicBlock* successor = nullptr;
    // 比较跳转，取第 1 和 2 个操作数为比较对象，第 3 个为跳转目标
    auto* lhs = dynamic_cast<RegisterOperand*>(inst->getOperand(0));
    auto* rhs = dynamic_cast<RegisterOperand*>(inst->getOperand(1));
    auto* target = dynamic_cast<LabelOperand*>(inst->getOperand(2));
    if (lhs == nullptr || rhs == nullptr || target == nullptr) {
        throw std::runtime_error("Invalid operands for conditional branch");
    }
    successor = getBBForLabel(target->getLabelName(), func);
    if (successor == nullptr) {
        throw std::runtime_error("No basic block found for label: " +
                                 target->getLabelName());
    }

    bb->addSuccessor(successor);
    successor->addPredecessor(bb);
}

void print(Function* func) {
    // 调试：打印出 CFG 信息
    std::cout << "Function: " << func->getName() << "\n";
    for (const auto& bb : *func) {
        std::cout << "  BasicBlock: " << bb->getLabel() << "\n";
        std::cout << "    Successors: ";
        for (const auto* succ : bb->getSuccessors()) {
            std::cout << succ->getLabel() << " ";
        }
        std::cout << "\n    Predecessors: ";
        for (const auto* pred : bb->getPredecessors()) {
            std::cout << pred->getLabel() << " ";
        }
        std::cout << "\n";
    }
}

}  // namespace CFG

void Visitor::createCFG(Function* func) {
    // 创建基本块之间的控制流图
    for (const auto& bb : *func) {
        for (const auto& inst : *bb) {
            switch (inst->getOpcode()) {
                case Opcode::J:
                    CFG::handleUnconditonalJump(func, bb.get(), inst.get());
                    break;

                case Opcode::BNEZ:
                case Opcode::BEQZ:
                case Opcode::BLEZ:
                case Opcode::BGEZ:
                    CFG::handleConditionalJump(func, bb.get(), inst.get());
                    break;

                case Opcode::BEQ:
                case Opcode::BNE:
                case Opcode::BLT:
                case Opcode::BGE:
                case Opcode::BLTU:
                case Opcode::BGEU:
                case Opcode::BGT:
                case Opcode::BLE:
                case Opcode::BGTU:
                case Opcode::BLEU:
                    CFG::handleComparisonJump(func, bb.get(), inst.get());
                    break;

                default:
                    // 对于其他指令，没有跳转目标
                    break;
            }
        }
    }
}

}  // namespace riscv64