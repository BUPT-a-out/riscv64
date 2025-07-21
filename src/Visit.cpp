#include "Visit.h"

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>

#include "ABI.h"
#include "CodeGen.h"
#include "IR/Instructions.h"
#include "IR/Module.h"
#include "Instructions/All.h"
#include "StackFrameManager.h"

namespace riscv64 {

Visitor::Visitor(CodeGenerator* code_gen) : codeGen_(code_gen) {}

// 访问 module
Module Visitor::visit(const midend::Module* module) {
    Module riscv_module;

    for (const auto& global : module->globals()) {
        visit(global);
    }
    for (auto* const func : *module) {
        visit(func, &riscv_module);
    }

    return riscv_module;
}

// 访问函数
void Visitor::visit(const midend::Function* func, Module* parent_module) {
    // 其他操作...
    auto riscv_func = std::make_unique<Function>(func->getName());
    auto* func_ptr = riscv_func.get();
    parent_module->addFunction(std::move(riscv_func));

    for (const auto& bb : *func) {
        codeGen_->mapBBToLabel(bb, bb->getName());
    }

    for (const auto& bb : *func) {
        visit(bb, func_ptr);
    }

    // 此时 func_ptr 已经包含了所有基本块，开始维护 CFG
    createCFG(func_ptr);
    // 调试：打印出 CFG 信息
    std::cout << "Function: " << func_ptr->getName() << "\n";
    for (const auto& bb : *func_ptr) {
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

BasicBlock* getBBForLabel(const std::string& label, Function* func) {
    for (const auto& bb : *func) {
        if (bb->getLabel() == label) {
            return bb.get();
        }
    }
    throw std::runtime_error("No basic block found for label: " + label);
}

void Visitor::createCFG(Function* func) {
    // 创建基本块之间的控制流图
    for (const auto& bb : *func) {
        for (const auto& inst : *bb) {
            BasicBlock* successor = nullptr;
            BasicBlock* predecessor = nullptr;
            switch (inst->getOpcode()) {
                case Opcode::J: {
                    // 无条件跳转，取第 1 个操作数作为目标基本块
                    auto* target =
                        dynamic_cast<LabelOperand*>(inst->getOperand(0));
                    if (target == nullptr) {
                        throw std::runtime_error(
                            "Invalid target for unconditional jump");
                    }

                    successor = getBBForLabel(
                        target->getLabelName(),
                        func);  // TODO(rikka): 这里或许会有重名的问题
                    if (successor == nullptr) {
                        throw std::runtime_error(
                            "No basic block found for label: " +
                            target->getLabelName());
                    }
                    bb->addSuccessor(successor);
                    successor->addPredecessor(bb.get());
                    break;
                }

                case Opcode::BNEZ:
                case Opcode::BEQZ:
                case Opcode::BLEZ:
                case Opcode::BGEZ: {
                    // 条件跳转，取第 1 个操作数作为条件，第 2
                    // 个操作数作为目标基本块
                    auto* condition =
                        dynamic_cast<RegisterOperand*>(inst->getOperand(0));
                    auto* target =
                        dynamic_cast<LabelOperand*>(inst->getOperand(1));
                    if (condition == nullptr || target == nullptr) {
                        throw std::runtime_error(
                            "Invalid operands for conditional branch");
                    }

                    successor = getBBForLabel(target->getLabelName(), func);
                    if (successor == nullptr) {
                        throw std::runtime_error(
                            "No basic block found for label: " +
                            target->getLabelName());
                    }
                    bb->addSuccessor(successor);
                    successor->addPredecessor(bb.get());
                    break;
                }

                case Opcode::BEQ:
                case Opcode::BNE:
                case Opcode::BLT:
                case Opcode::BGE:
                case Opcode::BLTU:
                case Opcode::BGEU:
                case Opcode::BGT:
                case Opcode::BLE:
                case Opcode::BGTU:
                case Opcode::BLEU: {
                    // 第 1 和 2 个操作数为比较对象，第 3 个为跳转目标
                    auto* lhs =
                        dynamic_cast<RegisterOperand*>(inst->getOperand(0));
                    auto* rhs =
                        dynamic_cast<RegisterOperand*>(inst->getOperand(1));
                    auto* target =
                        dynamic_cast<LabelOperand*>(inst->getOperand(2));
                    if (lhs == nullptr || rhs == nullptr || target == nullptr) {
                        throw std::runtime_error(
                            "Invalid operands for conditional branch");
                    }
                    successor = getBBForLabel(target->getLabelName(), func);
                    if (successor == nullptr) {
                        throw std::runtime_error(
                            "No basic block found for label: " +
                            target->getLabelName());
                    }

                    bb->addSuccessor(successor);
                    successor->addPredecessor(bb.get());
                    break;
                }

                default:
                    // 对于其他指令，没有跳转目标
                    break;
            }
        }
    }
}

// 访问基本块
void Visitor::visit(const midend::BasicBlock* bb, Function* parent_func) {
    // 其他操作...
    auto riscv_bb = std::make_unique<BasicBlock>(parent_func, bb->getName());
    auto* bb_ptr = riscv_bb.get();
    parent_func->addBasicBlock(std::move(riscv_bb));

    for (const auto& inst : *bb) {
        visit(inst, bb_ptr);
    }
}

// 访问指令
std::unique_ptr<MachineOperand> Visitor::visit(const midend::Instruction* inst,
                                               BasicBlock* parent_bb) {
    // 检查是否已经处理过
    auto foundReg = findRegForValue(inst);
    if (foundReg.has_value()) {
        return std::make_unique<RegisterOperand>(foundReg.value()->getRegNum(),
                                                 foundReg.value()->isVirtual());
    }

    switch (inst->getOpcode()) {
        case midend::Opcode::Add:
        case midend::Opcode::Sub:
        case midend::Opcode::Mul:
        case midend::Opcode::Div:
        case midend::Opcode::Rem:
        case midend::Opcode::And:
        case midend::Opcode::Or:
        case midend::Opcode::Xor:
        case midend::Opcode::Shl:
        case midend::Opcode::Shr:
        case midend::Opcode::ICmpSGT:
        case midend::Opcode::ICmpSLT:
        case midend::Opcode::ICmpEQ:
        case midend::Opcode::ICmpNE:
            // 处理算术指令，此处直接生成
            // TODO(rikka): 关于 0, 1, 2^n(左移) 的判断优化等，后期写一个 Pass
            // 来优化
            return visitBinaryOp(inst, parent_bb);
            break;
        case midend::Opcode::Load:
            return visitLoadInst(inst, parent_bb);
            break;
        case midend::Opcode::Alloca:
            return visitAllocaInst(inst, parent_bb);
            break;
        case midend::Opcode::Br:
            visitBranchInst(inst, parent_bb);
            break;
        case midend::Opcode::Store:
            visitStoreInst(inst, parent_bb);
            break;
        case midend::Opcode::Ret:
            // 处理返回指令
            visitRetInstruction(inst, parent_bb);
            break;
        case midend::Opcode::PHI:
            return visitPhiInst(inst, parent_bb);
            break;
        case midend::Opcode::Call:
            return visitCallInst(inst, parent_bb);
            break;
        case midend::Opcode::GetElementPtr:
            return visitGEPInst(inst, parent_bb);
            break;
        default:
            // 其他指令类型
            throw std::runtime_error("Unsupported instruction: " +
                                     inst->toString());
    }
    return nullptr;  // 对于不产生值的指令，返回 nullptr
}

std::unique_ptr<RegisterOperand> Visitor::immToReg(
    std::unique_ptr<MachineOperand> operand, BasicBlock* parent_bb) {
    // 将立即数存到寄存器中，如果已经是寄存器则直接返回
    if (operand->getType() == OperandType::Register) {
        auto* register_operand = dynamic_cast<RegisterOperand*>(operand.get());
        return std::make_unique<RegisterOperand>(register_operand->getRegNum(),
                                                 register_operand->isVirtual());
    }

    auto* imm_operand = dynamic_cast<ImmediateOperand*>(operand.get());
    if (imm_operand == nullptr) {
        throw std::runtime_error("Invalid immediate operand type");
    }

    if (imm_operand->getValue() == 0) {
        // 如果立即数是 0，直接返回 zero
        return std::make_unique<RegisterOperand>("zero");
    }

    // 生成一个新的寄存器，并将立即数加载到该寄存器中
    auto instruction = std::make_unique<Instruction>(Opcode::LI, parent_bb);
    auto new_reg = codeGen_->allocateReg();  // 分配一个新的寄存器
    auto reg_num = new_reg->getRegNum();
    auto is_virtual = new_reg->isVirtual();
    instruction->addOperand(std::move(new_reg));  // rd
    instruction->addOperand(
        std::make_unique<ImmediateOperand>(imm_operand->getValue()));  // imm
    parent_bb->addInstruction(std::move(instruction));

    return std::make_unique<RegisterOperand>(reg_num, is_virtual);
}

std::unique_ptr<MachineOperand> Visitor::visitGEPInst(
    const midend::Instruction* inst, BasicBlock* parent_bb) {
    if (inst->getOpcode() != midend::Opcode::GetElementPtr) {
        throw std::runtime_error("Not a GEP instruction: " + inst->toString());
    }

    const auto* gep_inst = dynamic_cast<const midend::GetElementPtrInst*>(inst);
    if (gep_inst == nullptr) {
        throw std::runtime_error("Not a GEP instruction: " + inst->toString());
    }

    // 获取基地址
    const auto* base_ptr = gep_inst->getPointerOperand();
    auto base_addr = visit(base_ptr, parent_bb);
    if (base_addr->getType() != OperandType::FrameIndex) {
        throw std::runtime_error(
            "Base address must be a frame index operand, got: " +
            base_addr->toString());
    }

    // 生成 frameaddr 指令来获取基地址
    auto get_base_addr_inst =
        std::make_unique<Instruction>(Opcode::FRAMEADDR, parent_bb);
    auto base_addr_reg = codeGen_->allocateReg();
    get_base_addr_inst->addOperand(std::make_unique<RegisterOperand>(
        base_addr_reg->getRegNum(), base_addr_reg->isVirtual()));  // rd
    get_base_addr_inst->addOperand(std::move(base_addr));          // FI
    parent_bb->addInstruction(std::move(get_base_addr_inst));

    // 计算偏移量
    if (gep_inst->getNumIndices() > 3) {
        throw std::runtime_error(
            "GEP instruction must have <= 3 indices: " + inst->toString() +
            ", got " + std::to_string(gep_inst->getNumIndices()));
    }

    // 获取指针指向的类型
    auto* pointed_type = gep_inst->getSourceElementType();

    // 初始化偏移量为0
    auto offset_reg = codeGen_->allocateReg();
    auto li_zero_inst = std::make_unique<Instruction>(Opcode::LI, parent_bb);
    li_zero_inst->addOperand(std::make_unique<RegisterOperand>(
        offset_reg->getRegNum(), offset_reg->isVirtual()));
    li_zero_inst->addOperand(std::make_unique<ImmediateOperand>(0));
    parent_bb->addInstruction(std::move(li_zero_inst));

    // 辅助函数：计算类型的字节大小
    std::function<size_t(const midend::Type*)> calculateTypeSize =
        [&](const midend::Type* type) -> size_t {
        if (type->isPointerType()) {
            return 8;  // 64位指针
        } else if (type->isIntegerType()) {
            return type->getBitWidth() / 8;
        } else if (type->isArrayType()) {
            auto* array_type = static_cast<const midend::ArrayType*>(type);
            auto element_size = calculateTypeSize(array_type->getElementType());
            return element_size * array_type->getNumElements();
        } else {
            return 4;  // 默认大小
        }
    };

    // 第一个索引通常是0，用于"穿透"指针，我们可以跳过它
    // 如果第一个索引不是0，我们需要计算指针本身的偏移
    auto* current_type = pointed_type;

    for (unsigned i = 0; i < gep_inst->getNumIndices(); ++i) {
        auto* index_value = gep_inst->getIndex(i);
        auto index_operand = visit(index_value, parent_bb);

        // 对于第一个索引，如果不是0，需要计算指针偏移
        if (i == 0) {
            // 检查是否为常量0，如果是则跳过
            if (auto* const_int =
                    midend::dyn_cast<midend::ConstantInt>(index_value)) {
                if (const_int->getSignedValue() == 0) {
                    continue;  // 跳过第一个索引为0的情况
                }
            }

            // 如果第一个索引不是0，计算指针偏移
            size_t type_size = calculateTypeSize(current_type);

            // 计算偏移：index * type_size
            auto index_reg = immToReg(std::move(index_operand), parent_bb);
            auto size_reg = codeGen_->allocateReg();
            auto li_size_inst =
                std::make_unique<Instruction>(Opcode::LI, parent_bb);
            li_size_inst->addOperand(std::make_unique<RegisterOperand>(
                size_reg->getRegNum(), size_reg->isVirtual()));
            li_size_inst->addOperand(
                std::make_unique<ImmediateOperand>(type_size));
            parent_bb->addInstruction(std::move(li_size_inst));

            auto mul_reg = codeGen_->allocateReg();
            auto mul_inst =
                std::make_unique<Instruction>(Opcode::MUL, parent_bb);
            mul_inst->addOperand(std::make_unique<RegisterOperand>(
                mul_reg->getRegNum(), mul_reg->isVirtual()));
            mul_inst->addOperand(std::move(index_reg));
            mul_inst->addOperand(std::move(size_reg));
            parent_bb->addInstruction(std::move(mul_inst));

            // 累加到总偏移量
            auto new_offset_reg = codeGen_->allocateReg();
            auto add_inst =
                std::make_unique<Instruction>(Opcode::ADD, parent_bb);
            add_inst->addOperand(std::make_unique<RegisterOperand>(
                new_offset_reg->getRegNum(), new_offset_reg->isVirtual()));
            add_inst->addOperand(std::make_unique<RegisterOperand>(
                offset_reg->getRegNum(), offset_reg->isVirtual()));
            add_inst->addOperand(std::move(mul_reg));
            parent_bb->addInstruction(std::move(add_inst));

            offset_reg = std::move(new_offset_reg);
            continue;
        }

        // 处理第二个及以后的索引
        if (current_type->isArrayType()) {
            const auto* array_type =
                dynamic_cast<const midend::ArrayType*>(current_type);

            // 关键修复：计算当前维度的步长
            // 步长 = 内层所有维度的元素总大小
            size_t stride = calculateTypeSize(array_type->getElementType());

            // 计算偏移：index * stride
            auto index_reg = immToReg(std::move(index_operand), parent_bb);

            std::unique_ptr<RegisterOperand> element_offset_reg;
            if (stride == 1) {
                // 如果步长是1，直接使用索引
                element_offset_reg = std::move(index_reg);
            } else if ((stride & (stride - 1)) == 0) {
                // 如果步长是2的幂，使用左移
                int shift_amount = 0;
                auto temp = static_cast<unsigned int>(stride);
                while (temp > 1) {
                    temp >>= 1;
                    shift_amount++;
                }

                element_offset_reg = codeGen_->allocateReg();
                auto slli_inst =
                    std::make_unique<Instruction>(Opcode::SLLI, parent_bb);
                slli_inst->addOperand(std::make_unique<RegisterOperand>(
                    element_offset_reg->getRegNum(),
                    element_offset_reg->isVirtual()));
                slli_inst->addOperand(std::move(index_reg));
                slli_inst->addOperand(
                    std::make_unique<ImmediateOperand>(shift_amount));
                parent_bb->addInstruction(std::move(slli_inst));
            } else {
                // 一般情况，使用乘法
                auto size_reg = codeGen_->allocateReg();
                auto li_size_inst =
                    std::make_unique<Instruction>(Opcode::LI, parent_bb);
                li_size_inst->addOperand(std::make_unique<RegisterOperand>(
                    size_reg->getRegNum(), size_reg->isVirtual()));
                li_size_inst->addOperand(
                    std::make_unique<ImmediateOperand>(stride));
                parent_bb->addInstruction(std::move(li_size_inst));

                element_offset_reg = codeGen_->allocateReg();
                auto mul_inst =
                    std::make_unique<Instruction>(Opcode::MUL, parent_bb);
                mul_inst->addOperand(std::make_unique<RegisterOperand>(
                    element_offset_reg->getRegNum(),
                    element_offset_reg->isVirtual()));
                mul_inst->addOperand(std::move(index_reg));
                mul_inst->addOperand(std::move(size_reg));
                parent_bb->addInstruction(std::move(mul_inst));
            }

            // 累加到总偏移量
            auto new_offset_reg = codeGen_->allocateReg();
            auto add_inst =
                std::make_unique<Instruction>(Opcode::ADD, parent_bb);
            add_inst->addOperand(std::make_unique<RegisterOperand>(
                new_offset_reg->getRegNum(), new_offset_reg->isVirtual()));
            add_inst->addOperand(std::make_unique<RegisterOperand>(
                offset_reg->getRegNum(), offset_reg->isVirtual()));
            add_inst->addOperand(std::move(element_offset_reg));
            parent_bb->addInstruction(std::move(add_inst));

            offset_reg = std::move(new_offset_reg);
            current_type = array_type->getElementType();
        } else {
            throw std::runtime_error("Unsupported type for GEP indexing: " +
                                     current_type->toString());
        }
    }

    // 计算最终地址：基地址 + 总偏移量
    auto final_addr_reg = codeGen_->allocateReg();
    auto final_add_inst = std::make_unique<Instruction>(Opcode::ADD, parent_bb);
    final_add_inst->addOperand(std::make_unique<RegisterOperand>(
        final_addr_reg->getRegNum(), final_addr_reg->isVirtual()));
    final_add_inst->addOperand(std::make_unique<RegisterOperand>(
        base_addr_reg->getRegNum(), base_addr_reg->isVirtual()));
    final_add_inst->addOperand(std::make_unique<RegisterOperand>(
        offset_reg->getRegNum(), offset_reg->isVirtual()));
    parent_bb->addInstruction(std::move(final_add_inst));

    // 建立GEP指令结果到寄存器的映射
    codeGen_->mapValueToReg(inst, final_addr_reg->getRegNum(),
                            final_addr_reg->isVirtual());

    return std::make_unique<RegisterOperand>(final_addr_reg->getRegNum(),
                                             final_addr_reg->isVirtual());
}

std::unique_ptr<MachineOperand> Visitor::visitCallInst(
    const midend::Instruction* inst, BasicBlock* parent_bb) {
    if (inst->getOpcode() != midend::Opcode::Call) {
        throw std::runtime_error("Not a call instruction: " + inst->toString());
    }

    // 处理函数调用指令
    const auto* call_inst = dynamic_cast<const midend::CallInst*>(inst);
    if (call_inst == nullptr) {
        throw std::runtime_error("Not a call instruction: " + inst->toString());
    }

    auto* called_func = call_inst->getCalledFunction();
    if (called_func == nullptr) {
        throw std::runtime_error("Called function not found for: " +
                                 inst->toString());
    }

    // 存入寄存器
    for (size_t arg_i = 0; arg_i < called_func->getNumArgs(); ++arg_i) {
        auto* dest_arg = called_func->getArg(arg_i);
        if (dest_arg == nullptr) {
            throw std::runtime_error(
                "Argument " + std::to_string(arg_i) +
                " is null in call instruction: " + inst->toString());
        }

        auto* source_operand = call_inst->getArgOperand(arg_i);
        if (source_operand == nullptr) {
            throw std::runtime_error(
                "Source operand for argument " + std::to_string(arg_i) +
                " is null in call instruction: " + inst->toString());
        }

        // 将参数转换为寄存器（真实的寄存器）
        auto dest_arg_operand = funcArgToReg(dest_arg);

        // Cast to RegisterOperand since function arguments should be in
        // registers
        auto* reg_operand =
            dynamic_cast<RegisterOperand*>(dest_arg_operand.get());
        if (reg_operand == nullptr) {
            throw std::runtime_error(
                "Function argument must be a register operand");
        }
        auto dest_reg = std::make_unique<RegisterOperand>(
            reg_operand->getRegNum(), reg_operand->isVirtual());

        // 将参数存储到寄存器中
        storeOperandToReg(visit(source_operand, parent_bb), std::move(dest_reg),
                          parent_bb);
    }

    // 生成调用指令
    auto riscv_call_inst =
        std::make_unique<Instruction>(Opcode::CALL, parent_bb);
    riscv_call_inst->addOperand(
        std::make_unique<LabelOperand>(called_func->getName()));  // 函数名
    parent_bb->addInstruction(std::move(riscv_call_inst));

    // 如果被调用函数有返回值，则将 a0 的值保存到一个新的寄存器并返回
    if (!called_func->getReturnType()->isVoidType()) {
        auto new_reg = codeGen_->allocateReg();
        auto dest_reg = std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                          new_reg->isVirtual());
        storeOperandToReg(std::make_unique<RegisterOperand>("a0"),
                          std::move(dest_reg), parent_bb);
        codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                new_reg->isVirtual());
        return new_reg;
    }

    return nullptr;
}

std::unique_ptr<MachineOperand> Visitor::visitPhiInst(
    const midend::Instruction* inst, BasicBlock* parent_bb) {
    if (inst->getOpcode() != midend::Opcode::PHI) {
        throw std::runtime_error("Not a PHI instruction: " + inst->toString());
    }

    const auto* phi_inst = dynamic_cast<const midend::PHINode*>(inst);
    if (phi_inst == nullptr) {
        throw std::runtime_error("Not a PHI instruction: " + inst->toString());
    }

    auto value_1 = visit(phi_inst->getIncomingValue(0), parent_bb);
    auto value_2 = visit(phi_inst->getIncomingValue(1), parent_bb);

    auto* parent_func = parent_bb->getParent();

    auto* incoming_block_1 = parent_func->getBasicBlockByLabel(
        phi_inst->getIncomingBlock(0)->getName());
    auto* incoming_block_2 = parent_func->getBasicBlockByLabel(
        phi_inst->getIncomingBlock(1)->getName());

    if (incoming_block_1 == nullptr || incoming_block_2 == nullptr) {
        throw std::runtime_error(
            "Incoming blocks not found for PHI instruction: " +
            inst->toString());
    }

    // 在最后一条跳转指令之前，插入 mov 指令
    auto new_reg = codeGen_->allocateReg();

    // 为 PHI 指令分配的寄存器创建副本用于插入
    auto dest_reg_1 = std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                        new_reg->isVirtual());
    auto dest_reg_2 = std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                        new_reg->isVirtual());

    storeOperandToReg(std::move(value_1), std::move(dest_reg_1),
                      incoming_block_1, std::prev(incoming_block_1->end()));
    storeOperandToReg(std::move(value_2), std::move(dest_reg_2),
                      incoming_block_2, std::prev(incoming_block_2->end()));

    // 映射 PHI 指令到分配的寄存器
    codeGen_->mapValueToReg(inst, new_reg->getRegNum(), new_reg->isVirtual());

    return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                             new_reg->isVirtual());
}

void Visitor::visitBranchInst(const midend::Instruction* inst,
                              BasicBlock* parent_bb) {
    if (inst->getOpcode() != midend::Opcode::Br) {
        throw std::runtime_error("Not a branch instruction: " +
                                 inst->toString());
    }

    const auto* branch_inst = dynamic_cast<const midend::BranchInst*>(inst);
    if (branch_inst == nullptr) {
        throw std::runtime_error("Not a branch instruction: " +
                                 inst->toString());
    }

    if (branch_inst->isUnconditional()) {
        // 处理无条件跳转
        auto* target_bb = branch_inst->getTargetBB();
        auto instruction = std::make_unique<Instruction>(Opcode::J, parent_bb);
        instruction->addOperand(std::make_unique<LabelOperand>(target_bb));
        parent_bb->addInstruction(std::move(instruction));
    } else {
        // 处理条件跳转
        // TODO(rikka): 根据 br 指令的 cond 类型生成不同的跳转指令
        auto condition = visit(branch_inst->getCondition(), parent_bb);
        auto* true_bb = branch_inst->getTrueBB();
        auto* false_bb = branch_inst->getFalseBB();

        // 生成条件跳转指令
        auto instruction =
            std::make_unique<Instruction>(Opcode::BNEZ, parent_bb);
        instruction->addOperand(std::move(condition));  // 条件
        instruction->addOperand(
            std::make_unique<LabelOperand>(true_bb));  // 真分支标签
        parent_bb->addInstruction(std::move(instruction));

        // 生成无条件跳转到假分支的指令
        auto false_instruction =
            std::make_unique<Instruction>(Opcode::J, parent_bb);
        false_instruction->addOperand(
            std::make_unique<LabelOperand>(false_bb));  // 跳转到假分支标签
        parent_bb->addInstruction(std::move(false_instruction));
    }
}

// 处理 alloca 指令
std::unique_ptr<MachineOperand> Visitor::visitAllocaInst(
    const midend::Instruction* inst, BasicBlock* parent_bb) {
    if (inst->getOpcode() != midend::Opcode::Alloca) {
        throw std::runtime_error("Not an alloca instruction: " +
                                 inst->toString());
    }

    const auto* alloca_inst = midend::dyn_cast<midend::AllocaInst>(inst);
    if (alloca_inst == nullptr) {
        throw std::runtime_error("Not an alloca instruction: " +
                                 inst->toString());
    }

    // 检查是否已经为这个alloca分配过FI
    auto* sfm = parent_bb->getParent()->getStackFrameManager();
    int existing_id = sfm->getAllocaStackSlotId(inst);
    if (existing_id != -1) {
        // 已经分配过，直接返回现有的FrameIndexOperand
        return std::make_unique<FrameIndexOperand>(existing_id);
    }

    // 处理 alloca 指令 - 正确计算类型大小
    auto* allocated_type = alloca_inst->getAllocatedType();

    // 辅助函数：递归计算类型的字节大小
    std::function<size_t(const midend::Type*)> calculateTypeSize =
        [&](const midend::Type* type) -> size_t {
        if (type->isPointerType()) {
            return 8;  // 64位指针
        } else if (type->isIntegerType()) {
            auto bit_width = type->getBitWidth();
            if (bit_width == 0) {
                // 对于某些整数类型，使用默认大小
                return 4;  // 默认32位整数
            }
            return (bit_width + 7) / 8;  // 向上舍入到字节
        } else if (type->isFloatType()) {
            return 4;  // float类型
        } else if (type->isArrayType()) {
            auto* array_type = static_cast<const midend::ArrayType*>(type);
            auto element_size = calculateTypeSize(array_type->getElementType());
            auto num_elements = array_type->getNumElements();
            return element_size * num_elements;
        } else {
            // 对于其他未知类型，使用默认大小
            return 4;
        }
    };

    // 计算基本类型的大小
    auto type_size = calculateTypeSize(allocated_type);

    // 处理数组分配
    int64_t array_size = 1;  // 默认数组大小为 1
    if (alloca_inst->isArrayAllocation()) {
        auto* array_size_value = alloca_inst->getArraySize();
        if (auto* array_size_const =
                midend::dyn_cast<midend::ConstantInt>(array_size_value)) {
            array_size = array_size_const->getSignedValue();
        } else {
            throw std::runtime_error("Array size must be a constant integer");
        }
    }

    // 计算总大小
    auto total_size = type_size * array_size;

    if (total_size == 0) {
        throw std::runtime_error("Invalid alloca size: " +
                                 std::to_string(total_size));
    }

    // 创建栈对象
    int id = sfm->getNewStackObjectIdentifier();
    auto stackObject =
        std::make_unique<StackObject>(static_cast<int>(total_size),  // 总大小
                                      4,  // 对齐要求(4字节对齐)
                                      id  // 标识符
        );

    sfm->addStackObject(std::move(stackObject));
    sfm->mapAllocaToStackSlot(inst, id);

    // 为alloca指令本身也建立映射，这样后续引用这个指令时能找到对应的FI
    codeGen_->mapValueToReg(
        inst, id,
        false);  // 使用id作为"寄存器号"，false表示这不是真正的寄存器

    // 返回分配的FrameIndexOperand
    return std::make_unique<FrameIndexOperand>(id);
}

// 处理 store 指令
void Visitor::visitStoreInst(const midend::Instruction* inst,
                             BasicBlock* parent_bb) {
    if (inst->getOpcode() != midend::Opcode::Store) {
        throw std::runtime_error("Not a store instruction: " +
                                 inst->toString());
    }
    if (inst->getNumOperands() != 2) {
        throw std::runtime_error(
            "Store instruction must have two operands, got " +
            std::to_string(inst->getNumOperands()));
    }

    const auto* store_inst = dynamic_cast<const midend::StoreInst*>(inst);

    // 获取存储的值
    auto value_operand =
        immToReg(visit(store_inst->getValueOperand(), parent_bb), parent_bb);

    // 获取指针操作数
    auto* pointer_operand = store_inst->getPointerOperand();

    // 处理指针操作数 - 可能是 alloca 指令或者 GEP 指令
    if (auto* alloca_inst =
            midend::dyn_cast<midend::AllocaInst>(pointer_operand)) {
        // 直接是 alloca 指令
        auto* sfm = parent_bb->getParent()->getStackFrameManager();
        int frame_id = sfm->getAllocaStackSlotId(alloca_inst);

        if (frame_id == -1) {
            // 如果还没有为这个alloca分配FI，现在分配
            visitAllocaInst(alloca_inst, parent_bb);
            frame_id = sfm->getAllocaStackSlotId(alloca_inst);
        }

        if (frame_id == -1) {
            throw std::runtime_error(
                "Cannot find frame index for alloca instruction in store");
        }

        // 生成frameaddr指令来获取栈地址
        auto frame_addr_reg = codeGen_->allocateReg();
        auto store_frame_addr_inst =
            std::make_unique<Instruction>(Opcode::FRAMEADDR, parent_bb);
        store_frame_addr_inst->addOperand(std::make_unique<RegisterOperand>(
            frame_addr_reg->getRegNum(), frame_addr_reg->isVirtual()));  // rd
        store_frame_addr_inst->addOperand(
            std::make_unique<FrameIndexOperand>(frame_id));  // FI
        parent_bb->addInstruction(std::move(store_frame_addr_inst));

        // 生成存储指令
        auto sw_inst = std::make_unique<Instruction>(Opcode::SW, parent_bb);
        sw_inst->addOperand(std::move(value_operand));  // source register
        sw_inst->addOperand(std::make_unique<MemoryOperand>(
            std::move(frame_addr_reg),
            std::make_unique<ImmediateOperand>(0)));  // memory address
        parent_bb->addInstruction(std::move(sw_inst));

    } else if (auto* gep_inst = midend::dyn_cast<midend::GetElementPtrInst>(
                   pointer_operand)) {
        // 是 GEP 指令的结果
        // 访问 GEP 指令，它会返回计算出的地址寄存器
        auto address_operand = visit(gep_inst, parent_bb);

        // address_operand 应该是一个寄存器操作数，包含计算出的地址
        auto* address_reg =
            dynamic_cast<RegisterOperand*>(address_operand.get());
        if (address_reg == nullptr) {
            throw std::runtime_error(
                "GEP instruction should return a register operand");
        }

        // 生成存储指令，直接使用 GEP 计算出的地址
        auto sw_inst = std::make_unique<Instruction>(Opcode::SW, parent_bb);
        sw_inst->addOperand(std::move(value_operand));  // source register
        sw_inst->addOperand(std::make_unique<MemoryOperand>(
            std::make_unique<RegisterOperand>(address_reg->getRegNum(),
                                              address_reg->isVirtual()),
            std::make_unique<ImmediateOperand>(0)));  // memory address
        parent_bb->addInstruction(std::move(sw_inst));

    } else {
        // 其他类型的指针操作数
        throw std::runtime_error(
            "Unsupported pointer operand type in store instruction: " +
            pointer_operand->toString());
    }
}

// 处理 load 指令
std::unique_ptr<MachineOperand> Visitor::visitLoadInst(
    const midend::Instruction* inst, BasicBlock* parent_bb) {
    if (inst->getOpcode() != midend::Opcode::Load) {
        throw std::runtime_error("Not a load instruction: " + inst->toString());
    }
    if (inst->getNumOperands() != 1) {
        throw std::runtime_error(
            "Load instruction must have one operand, got " +
            std::to_string(inst->getNumOperands()));
    }

    const auto* load_inst = dynamic_cast<const midend::LoadInst*>(inst);

    // 获取指针操作数
    auto* pointer_operand = load_inst->getPointerOperand();

    // 处理指针操作数 - 可能是 alloca 指令或者 GEP 指令
    if (auto* alloca_inst =
            midend::dyn_cast<midend::AllocaInst>(pointer_operand)) {
        // 直接是 alloca 指令
        auto* sfm = parent_bb->getParent()->getStackFrameManager();
        int frame_id = sfm->getAllocaStackSlotId(alloca_inst);

        if (frame_id == -1) {
            throw std::runtime_error(
                "Cannot find frame index for alloca instruction in load");
        }

        // 生成frameaddr指令来获取栈地址
        auto frame_addr_reg = codeGen_->allocateReg();
        auto load_frame_addr_inst =
            std::make_unique<Instruction>(Opcode::FRAMEADDR, parent_bb);
        load_frame_addr_inst->addOperand(std::make_unique<RegisterOperand>(
            frame_addr_reg->getRegNum(), frame_addr_reg->isVirtual()));  // rd
        load_frame_addr_inst->addOperand(
            std::make_unique<FrameIndexOperand>(frame_id));  // FI
        parent_bb->addInstruction(std::move(load_frame_addr_inst));

        // 加载到新的寄存器
        auto new_reg = codeGen_->allocateReg();
        auto load_inst_ptr =
            std::make_unique<Instruction>(Opcode::LW, parent_bb);
        load_inst_ptr->addOperand(std::make_unique<RegisterOperand>(
            new_reg->getRegNum(), new_reg->isVirtual()));  // rd
        load_inst_ptr->addOperand(std::make_unique<MemoryOperand>(
            std::move(frame_addr_reg),
            std::make_unique<ImmediateOperand>(0)));  // memory address
        parent_bb->addInstruction(std::move(load_inst_ptr));

        // 建立load指令结果值到寄存器的映射
        codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                new_reg->isVirtual());

        return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                 new_reg->isVirtual());

    } else if (auto* gep_inst = midend::dyn_cast<midend::GetElementPtrInst>(
                   pointer_operand)) {
        // 是 GEP 指令的结果
        // 访问 GEP 指令，它会返回计算出的地址寄存器
        auto address_operand = visit(gep_inst, parent_bb);

        // address_operand 应该是一个寄存器操作数，包含计算出的地址
        auto* address_reg =
            dynamic_cast<RegisterOperand*>(address_operand.get());
        if (address_reg == nullptr) {
            throw std::runtime_error(
                "GEP instruction should return a register operand");
        }

        // 加载到新的寄存器
        auto new_reg = codeGen_->allocateReg();
        auto load_inst_ptr =
            std::make_unique<Instruction>(Opcode::LW, parent_bb);
        load_inst_ptr->addOperand(std::make_unique<RegisterOperand>(
            new_reg->getRegNum(), new_reg->isVirtual()));  // rd
        load_inst_ptr->addOperand(std::make_unique<MemoryOperand>(
            std::make_unique<RegisterOperand>(address_reg->getRegNum(),
                                              address_reg->isVirtual()),
            std::make_unique<ImmediateOperand>(0)));  // memory address
        parent_bb->addInstruction(std::move(load_inst_ptr));

        // 建立load指令结果值到寄存器的映射
        codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                new_reg->isVirtual());

        return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                 new_reg->isVirtual());

    } else {
        // 其他类型的指针操作数
        throw std::runtime_error(
            "Unsupported pointer operand type in load instruction: " +
            pointer_operand->toString());
    }
}

// 处理二元运算指令
// Handles binary operation instructions by generating the appropriate
// RISC-V instructions for the given midend instruction, allocating
// registers as needed, and returning the result operand. Supports constant
// folding for immediate operands and maps the result to a register.
std::unique_ptr<MachineOperand> Visitor::visitBinaryOp(
    const midend::Instruction* inst, BasicBlock* parent_bb) {
    if (!inst->isBinaryOp() && !inst->isComparison()) {
        throw std::runtime_error("Not a binary operation instruction: " +
                                 inst->toString());
    }
    if (inst->getNumOperands() != 2) {
        throw std::runtime_error(
            "Binary operation must have two operands, got " +
            std::to_string(inst->getNumOperands()));
    }

    std::unique_ptr<MachineOperand> lhs;
    {
        const auto foundReg = findRegForValue(inst->getOperand(0));
        if (foundReg.has_value()) {
            lhs = std::make_unique<RegisterOperand>(
                foundReg.value()->getRegNum(), foundReg.value()->isVirtual());
        } else {
            lhs = visit(inst->getOperand(0), parent_bb);
        }
    }
    std::unique_ptr<MachineOperand> rhs;
    {
        const auto foundReg = findRegForValue(inst->getOperand(1));
        if (foundReg.has_value()) {
            rhs = std::make_unique<RegisterOperand>(
                foundReg.value()->getRegNum(), foundReg.value()->isVirtual());
        } else {
            rhs = visit(inst->getOperand(1), parent_bb);
        }
    }

    // Only allocate a new register if needed (not for immediate result)
    std::unique_ptr<RegisterOperand> new_reg;

    // TODO(rikka): 关于 0 和 1 的判断优化等，后期写一个 Pass 来优化
    switch (inst->getOpcode()) {
        case midend::Opcode::Add: {
            // 先判断是否有立即数
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(lhs_imm->getValue() +
                                                          rhs_imm->getValue());
            }

            if ((lhs->getType() == OperandType::Immediate) !=
                (rhs->getType() == OperandType::Immediate)) {
                // 使用 addi 指令
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::ADDI, parent_bb);
                auto* imm_operand = dynamic_cast<ImmediateOperand*>(
                    lhs->getType() == OperandType::Immediate ? lhs.get()
                                                             : rhs.get());
                auto* reg_operand = dynamic_cast<RegisterOperand*>(
                    lhs->getType() == OperandType::Register ? lhs.get()
                                                            : rhs.get());
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    reg_operand->getRegNum(),
                    new_reg->isVirtual()));  // rs1
                instruction->addOperand(std::make_unique<ImmediateOperand>(
                    imm_operand->getValue()));  // imm

                parent_bb->addInstruction(std::move(instruction));
            } else {
                // 使用 add 指令
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::ADD, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::move(lhs));           // rs1
                instruction->addOperand(std::move(rhs));           // rs2
                parent_bb->addInstruction(std::move(instruction));
            }
            break;
        }

        case midend::Opcode::Sub: {
            // 先判断是否是两个立即数
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(lhs_imm->getValue() -
                                                          rhs_imm->getValue());
            }

            auto lhs_reg = immToReg(std::move(lhs), parent_bb);
            auto rhs_reg = immToReg(std::move(rhs), parent_bb);

            new_reg = codeGen_->allocateReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::SUB, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual()));  // rd
            instruction->addOperand(std::move(lhs_reg));       // rs1
            instruction->addOperand(std::move(rhs_reg));       // rs2

            parent_bb->addInstruction(std::move(instruction));
            break;
        }

        case midend::Opcode::Mul: {
            // 先判断是否是两个立即数
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(lhs_imm->getValue() *
                                                          rhs_imm->getValue());
            }

            auto lhs_reg = immToReg(std::move(lhs), parent_bb);
            auto rhs_reg = immToReg(std::move(rhs), parent_bb);

            new_reg = codeGen_->allocateReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::MUL, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual()));  // rd
            instruction->addOperand(std::move(lhs_reg));       // rs1
            instruction->addOperand(std::move(rhs_reg));       // rs2

            parent_bb->addInstruction(std::move(instruction));
            break;
        }

        case midend::Opcode::ICmpSGT: {
            // 处理有符号大于比较
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    lhs_imm->getValue() > rhs_imm->getValue() ? 1 : 0);
            }

            auto lhs_reg = immToReg(std::move(lhs), parent_bb);
            auto rhs_reg = immToReg(std::move(rhs), parent_bb);

            // 使用 sgt 指令
            new_reg = codeGen_->allocateReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::SGT, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual()));  // rd
            instruction->addOperand(std::move(lhs_reg));       // rs1
            instruction->addOperand(std::move(rhs_reg));       // rs2

            parent_bb->addInstruction(std::move(instruction));
            break;
        }

        case midend::Opcode::Rem: {
            // 处理取模操作
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                if (rhs_imm->getValue() == 0) {
                    throw std::runtime_error(
                        "Division by zero in modulo operation");
                }
                return std::make_unique<ImmediateOperand>(lhs_imm->getValue() %
                                                          rhs_imm->getValue());
            }

            auto lhs_reg = immToReg(std::move(lhs), parent_bb);
            auto rhs_reg = immToReg(std::move(rhs), parent_bb);

            new_reg = codeGen_->allocateReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::REM, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual()));  // rd
            instruction->addOperand(std::move(lhs_reg));       // rs1
            instruction->addOperand(std::move(rhs_reg));       // rs2

            parent_bb->addInstruction(std::move(instruction));
            break;
        }

        case midend::Opcode::ICmpEQ: {
            // 处理相等比较
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    lhs_imm->getValue() == rhs_imm->getValue() ? 1 : 0);
            }

            auto lhs_reg = immToReg(std::move(lhs), parent_bb);
            auto rhs_reg = immToReg(std::move(rhs), parent_bb);

            // 使用 xor 指令计算差值，然后用 seqz 指令检查是否为0
            auto xor_reg = codeGen_->allocateReg();
            auto xor_inst =
                std::make_unique<Instruction>(Opcode::XOR, parent_bb);
            xor_inst->addOperand(std::make_unique<RegisterOperand>(
                xor_reg->getRegNum(), xor_reg->isVirtual()));  // rd
            xor_inst->addOperand(std::move(lhs_reg));          // rs1
            xor_inst->addOperand(std::move(rhs_reg));          // rs2
            parent_bb->addInstruction(std::move(xor_inst));

            new_reg = codeGen_->allocateReg();
            auto seqz_inst =
                std::make_unique<Instruction>(Opcode::SEQZ, parent_bb);
            seqz_inst->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual()));  // rd
            seqz_inst->addOperand(std::move(xor_reg));         // rs1
            parent_bb->addInstruction(std::move(seqz_inst));
            break;
        }

        case midend::Opcode::ICmpNE: {
            // 处理不等比较
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    lhs_imm->getValue() != rhs_imm->getValue() ? 1 : 0);
            }

            auto lhs_reg = immToReg(std::move(lhs), parent_bb);
            auto rhs_reg = immToReg(std::move(rhs), parent_bb);

            // 使用 xor 指令计算差值，然后用 snez 指令检查是否非0
            auto xor_reg = codeGen_->allocateReg();
            auto xor_inst =
                std::make_unique<Instruction>(Opcode::XOR, parent_bb);
            xor_inst->addOperand(std::make_unique<RegisterOperand>(
                xor_reg->getRegNum(), xor_reg->isVirtual()));  // rd
            xor_inst->addOperand(std::move(lhs_reg));          // rs1
            xor_inst->addOperand(std::move(rhs_reg));          // rs2
            parent_bb->addInstruction(std::move(xor_inst));

            new_reg = codeGen_->allocateReg();
            auto snez_inst =
                std::make_unique<Instruction>(Opcode::SNEZ, parent_bb);
            snez_inst->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual()));  // rd
            snez_inst->addOperand(std::move(xor_reg));         // rs1
            parent_bb->addInstruction(std::move(snez_inst));
            break;
        }

        case midend::Opcode::ICmpSLT: {
            // 处理有符号小于比较
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    lhs_imm->getValue() < rhs_imm->getValue() ? 1 : 0);
            }

            new_reg = codeGen_->allocateReg();
            if (rhs->getType() == OperandType::Immediate) {
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                auto slti_inst =
                    std::make_unique<Instruction>(Opcode::SLTI, parent_bb);
                slti_inst->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                slti_inst->addOperand(std::move(lhs));             // rs1
                slti_inst->addOperand(std::make_unique<ImmediateOperand>(
                    rhs_imm->getValue()));  // imm

                parent_bb->addInstruction(std::move(slti_inst));
            } else {
                auto lhs_reg = immToReg(std::move(lhs), parent_bb);
                auto rhs_reg = immToReg(std::move(rhs), parent_bb);

                // 使用 slt 指令
                auto instruction =
                    std::make_unique<Instruction>(Opcode::SLT, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::move(lhs_reg));       // rs1
                instruction->addOperand(std::move(rhs_reg));       // rs2

                parent_bb->addInstruction(std::move(instruction));
            }
            break;
        }

        // 其他二元运算...
        default:
            throw std::runtime_error("Unsupported binary operation: " +
                                     inst->toString());
    }

    // 返回新分配的寄存器操作数（如果有）
    if (new_reg) {
        codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                new_reg->isVirtual());
        return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                 new_reg->isVirtual());
    }  // Should only happen if we returned early (immediate case)
    throw std::runtime_error("No register allocated for binary op result");
}

// 将值存储到寄存器中，生成 mv 或者 li 指令
void Visitor::storeOperandToReg(
    std::unique_ptr<MachineOperand> source_operand,
    std::unique_ptr<MachineOperand> dest_reg, BasicBlock* parent_bb,
    std::list<std::unique_ptr<Instruction>>::const_iterator insert_pos) {
    if (insert_pos ==
        std::list<std::unique_ptr<Instruction>>::const_iterator{}) {
        insert_pos = parent_bb->end();
    }  // 默认值

    if (dest_reg->getType() == OperandType::Register) {
        switch (source_operand->getType()) {
            case OperandType::Immediate: {
                auto inst =
                    std::make_unique<Instruction>(Opcode::LI, parent_bb);
                auto* const source_imm =
                    dynamic_cast<ImmediateOperand*>(source_operand.get());

                inst->addOperand(std::move(dest_reg));  // rd
                inst->addOperand(std::make_unique<ImmediateOperand>(
                    source_imm->getValue()));  // imm
                parent_bb->insert(insert_pos, std::move(inst));
                break;
            }
            case OperandType::Register: {
                auto inst =
                    std::make_unique<Instruction>(Opcode::MV, parent_bb);
                auto* reg_source =
                    dynamic_cast<RegisterOperand*>(source_operand.get());

                inst->addOperand(std::move(dest_reg));  // rd
                inst->addOperand(std::make_unique<RegisterOperand>(
                    reg_source->getRegNum(),
                    reg_source->isVirtual()));  // rs
                parent_bb->insert(insert_pos, std::move(inst));
                break;
            }

            default:
                // TODO(rikka): 其他类型的返回值处理
                throw std::runtime_error("Unsupported return value type: " +
                                         std::to_string(static_cast<int>(
                                             source_operand->getType())));
        }
        return;
    }
}

void Visitor::visitRetInstruction(const midend::Instruction* ret_inst,
                                  BasicBlock* parent_bb) {
    if (ret_inst->getOpcode() != midend::Opcode::Ret) {
        throw std::runtime_error("Unsupported return instruction: " +
                                 ret_inst->toString());
    }
    if (ret_inst->getNumOperands() != 1) {
        throw std::runtime_error(
            "Return instruction must have one operand, got " +
            std::to_string(ret_inst->getNumOperands()));
    }

    // 处理返回值
    auto ret_operand = visit(ret_inst->getOperand(0), parent_bb);
    storeOperandToReg(std::move(ret_operand),
                      std::make_unique<RegisterOperand>("a0"), parent_bb);
    // 添加返回指令
    auto riscv_ret_inst = std::make_unique<Instruction>(Opcode::RET, parent_bb);
    parent_bb->addInstruction(std::move(riscv_ret_inst));
}

// 封装 getRegForValue
std::optional<RegisterOperand*> Visitor::findRegForValue(
    const midend::Value* value) {
    try {
        return codeGen_->getRegForValue(value);
    } catch (const std::runtime_error& e) {
        // 如果没有找到寄存器，返回 std::nullopt
        return std::nullopt;
    }
}

std::unique_ptr<MachineOperand> Visitor::funcArgToReg(
    const midend::Argument* argument) {
    // 获取函数参数对应的物理寄存器或者栈帧
    if (argument->getArgNo() < 8) {
        // 如果参数编号小于的寄存器数量，直接返回对应的寄存器
        return std::make_unique<RegisterOperand>(
            "a" + std::to_string(argument->getArgNo()));
    }
    // 否则，从栈帧中取出来
    // TODO(rikka): 处理栈帧中的参数
    throw std::runtime_error("Stack frame arguments not implemented yet");
}

std::unique_ptr<MachineOperand> Visitor::visit(const midend::Value* value,
                                               BasicBlock* parent_bb) {
    // 处理值的访问
    // 检查是否已经处理过该值
    const auto foundReg = findRegForValue(value);
    if (foundReg.has_value()) {
        // 对于alloca指令，即使已经处理过，如果它被用作指针，也应该返回FrameIndex
        if (auto* alloca_inst = midend::dyn_cast<midend::AllocaInst>(value)) {
            auto* sfm = parent_bb->getParent()->getStackFrameManager();
            int frame_id = sfm->getAllocaStackSlotId(alloca_inst);
            if (frame_id != -1) {
                return std::make_unique<FrameIndexOperand>(frame_id);
            }
        }
        // 直接使用找到的寄存器操作数
        return std::make_unique<RegisterOperand>(foundReg.value()->getRegNum(),
                                                 foundReg.value()->isVirtual());
    }

    // 检查是否是常量
    if (midend::isa<midend::ConstantInt>(value)) {
        // 判断范围，是否在 [-2048, 2047] 之间
        auto value_int =
            midend::cast<midend::ConstantInt>(value)->getSignedValue();
        constexpr int64_t IMM_MIN = -2048;
        constexpr int64_t IMM_MAX = 2047;
        auto signed_value = static_cast<int64_t>(value_int);
        if (signed_value >= IMM_MIN && signed_value <= IMM_MAX) {
            return std::make_unique<ImmediateOperand>(value_int);
        }
        // 如果不在范围内，分配一个新的寄存器
        auto new_reg = codeGen_->allocateReg();
        codeGen_->mapValueToReg(value, new_reg->getRegNum(),
                                new_reg->isVirtual());
        auto inst = std::make_unique<Instruction>(Opcode::LI, parent_bb);
        inst->addOperand(
            std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                              new_reg->isVirtual()));  // rd
        inst->addOperand(std::make_unique<ImmediateOperand>(value_int));
        parent_bb->addInstruction(std::move(inst));
        // 返回新分配的寄存器操作数
        return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                 new_reg->isVirtual());
    }

    // 检查是否是指令，如果是则递归处理（作为值使用）
    if (midend::isa<midend::Instruction>(value)) {
        return visit(midend::cast<midend::Instruction>(value), parent_bb);
    }

    // 如果是函数参数，先看是否已经有对应的虚拟寄存器（开头处已经完成），如果没有则需要分配虚拟寄存器（在这一步完成）
    if (value->getValueKind() == midend::ValueKind::Argument) {
        const auto* argument = midend::cast<midend::Argument>(value);
        auto new_reg = codeGen_->allocateReg();
        codeGen_->mapValueToReg(value, new_reg->getRegNum(),
                                new_reg->isVirtual());
        auto source_reg = funcArgToReg(argument);
        storeOperandToReg(std::move(source_reg),
                          std::make_unique<RegisterOperand>(
                              new_reg->getRegNum(), new_reg->isVirtual()),
                          parent_bb, parent_bb->begin());

        return new_reg;
    }

    // 检查是否是alloca指令，如果是则应该返回对应的FrameIndex
    if (auto* alloca_inst = midend::dyn_cast<midend::AllocaInst>(value)) {
        auto* sfm = parent_bb->getParent()->getStackFrameManager();
        int frame_id = sfm->getAllocaStackSlotId(alloca_inst);
        if (frame_id == -1) {
            // 如果还没有为这个alloca分配FI，现在分配
            auto frame_operand = visitAllocaInst(alloca_inst, parent_bb);
            auto* fi_operand =
                dynamic_cast<FrameIndexOperand*>(frame_operand.get());
            if (fi_operand) {
                frame_id = fi_operand->getIndex();
            }
        }
        if (frame_id != -1) {
            return std::make_unique<FrameIndexOperand>(frame_id);
        }
    }

    // 检查是否是指针类型
    if (value->getType()->isPointerType()) {
        // 如果是指针类型，可能是一个alloca指令的结果
        if (const auto* alloca_inst =
                midend::dyn_cast<midend::AllocaInst>(value)) {
            return visitAllocaInst(alloca_inst, parent_bb);
        }
        // 其他指针类型的处理（如全局变量等）
        throw std::runtime_error("Pointer type not handled: " +
                                 value->toString());
    }

    throw std::runtime_error(
        "Unsupported value type: " + value->getName() + " (type: " +
        std::to_string(static_cast<int>(value->getValueKind())) + ")");
}

// 访问常量
void Visitor::visit(const midend::Constant* /*constant*/) {}

// 访问 global variable
void Visitor::visit(const midend::GlobalVariable* /*var*/) {
    // 其他操作...
}

}  // namespace riscv64