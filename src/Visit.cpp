#include "Visit.h"

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>

#include "ABI.h"
#include "CodeGen.h"
#include "IR/IRPrinter.h"
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
        visit(global, &riscv_module);
    }
    for (auto* const func : *module) {
        if (func->isDefinition()) {
            visit(func, &riscv_module);
        }
    }

    return riscv_module;
}

// 访问函数
void Visitor::visit(const midend::Function* func, Module* parent_module) {
    // 为新函数清理函数级别的映射
    codeGen_->clearFunctionLevelMappings();

    // 其他操作...
    auto riscv_func = std::make_unique<Function>(func->getName());
    auto* func_ptr = riscv_func.get();
    parent_module->addFunction(std::move(riscv_func));

    for (const auto& bb : *func) {
        // codeGen_->mapBBToLabel(bb, bb->getName());
        auto new_riscv_bb =
            std::make_unique<BasicBlock>(func_ptr, bb->getName());
        auto* bb_ptr = new_riscv_bb.get();
        func_ptr->addBasicBlock(std::move(new_riscv_bb));
        func_ptr->mapBasicBlock(bb, bb_ptr);
    }

    // 第二阶段：在第一个基本块开头处理所有函数参数
    auto first_bb_iter = func->begin();
    if (first_bb_iter != func->end()) {
        auto* first_riscv_bb = func_ptr->getBasicBlock(*first_bb_iter);
        if (first_riscv_bb != nullptr) {
            // 预先为所有参数分配虚拟寄存器并生成转移指令
            for (auto arg_it = func->arg_begin(); arg_it != func->arg_end();
                 arg_it++) {
                // 为参数分配虚拟寄存器
                auto new_reg = codeGen_->allocateReg();
                codeGen_->mapValueToReg(arg_it->get(), new_reg->getRegNum(),
                                        new_reg->isVirtual());

                // 获取参数的源寄存器或栈位置
                auto source_reg = funcArgToReg(arg_it->get(), first_riscv_bb);

                // 生成参数转移指令（插入到基本块开头）
                storeOperandToReg(
                    std::move(source_reg),
                    std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                      new_reg->isVirtual()),
                    first_riscv_bb,
                    first_riscv_bb->begin()  // 插入到开头
                );
            }
        }
    }

    for (const auto& bb : *func) {
        visit(bb, func_ptr);
        // func_ptr->mapBasicBlock(bb, new_riscv_bb);
    }

    // 为新函数清理函数级别的映射
    codeGen_->clearFunctionLevelMappings();

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
                    if (inst->getOperand(0)->isImm()) {
                        auto* immCondition = dynamic_cast<ImmediateOperand*>(
                            inst->getOperand(0));
                        if (immCondition->getValue() != 0) {
                            // 如果条件是立即数且不为0，直接跳转到目标基本块
                            auto* target = dynamic_cast<LabelOperand*>(
                                inst->getOperand(1));
                            if (target == nullptr) {
                                throw std::runtime_error(
                                    "Invalid target for unconditional jump");
                            }
                            successor =
                                getBBForLabel(target->getLabelName(), func);
                            bb->addSuccessor(successor);
                            successor->addPredecessor(bb.get());
                            return;
                        }
                    }

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
BasicBlock* Visitor::visit(const midend::BasicBlock* bb,
                           Function* parent_func) {
    // 其他操作...
    auto* bb_ptr = parent_func->getBasicBlock(bb);
    // parent_func->addBasicBlock(std::move(riscv_bb));
    // parent_func->mapBasicBlock(bb, bb_ptr);

    for (const auto& inst : *bb) {
        visit(inst, bb_ptr);
    }
    return bb_ptr;  // 返回新创建的基本块指针
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
        case midend::Opcode::ICmpSLE:
        case midend::Opcode::ICmpSGE:
            // 处理算术指令，此处直接生成
            // TODO(rikka): 关于 0, 1, 2^n(左移) 的判断优化等，后期写一个 Pass
            // 来优化
            return visitBinaryOp(inst, parent_bb);
            break;
        case midend::Opcode::UAdd:
        case midend::Opcode::USub:
        case midend::Opcode::Not:
            // 处理一元操作指令
            return visitUnaryOp(inst, parent_bb);
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
        case midend::Opcode::Cast:
            return visitCastInst(inst, parent_bb);
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

    // 处理 FrameIndex 操作数
    if (operand->getType() == OperandType::FrameIndex) {
        auto* frame_operand = dynamic_cast<FrameIndexOperand*>(operand.get());
        if (frame_operand == nullptr) {
            throw std::runtime_error("Invalid frame index operand type: " +
                                     operand->toString());
        }

        // 生成一个新的寄存器，并使用 FRAMEADDR 指令获取帧地址
        auto new_reg = codeGen_->allocateReg();
        auto instruction =
            std::make_unique<Instruction>(Opcode::FRAMEADDR, parent_bb);
        instruction->addOperand(std::make_unique<RegisterOperand>(
            new_reg->getRegNum(), new_reg->isVirtual()));  // rd
        instruction->addOperand(std::make_unique<FrameIndexOperand>(
            frame_operand->getIndex()));  // FI
        parent_bb->addInstruction(std::move(instruction));

        return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                 new_reg->isVirtual());
    }

    auto* imm_operand = dynamic_cast<ImmediateOperand*>(operand.get());
    if (imm_operand == nullptr) {
        throw std::runtime_error("Invalid immediate operand type: " +
                                 operand->toString());
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

std::unique_ptr<MachineOperand> Visitor::visitCastInst(
    const midend::Instruction* inst, BasicBlock* parent_bb) {
    auto* cast_inst = midend::dyn_cast<midend::CastInst>(inst);
    if (cast_inst == nullptr) {
        throw std::runtime_error("Not a cast instruction: " + inst->toString());
    }

    switch (cast_inst->getCastOpcode()) {
        case midend::CastInst::Trunc:
        case midend::CastInst::SIToFP: {
            auto* dest_type = cast_inst->getDestType();
            auto* src_type = cast_inst->getSrcType();
            if (dest_type == nullptr || src_type == nullptr) {
                throw std::runtime_error("Invalid operands for trunc cast");
            }
            if (dest_type->isIntegerType()) {
                // i32 -> i1
                if (dest_type->getBitWidth() == 1 &&
                    src_type->getBitWidth() > 1) {
                    // use sltiu rd, rs1, imm
                    auto new_reg = codeGen_->allocateReg();
                    auto* new_reg_ptr = new_reg.get();
                    auto src_operand =
                        visit(cast_inst->getOperand(0), parent_bb);
                    auto rs1 = immToReg(std::move(src_operand), parent_bb);
                    auto instruction =
                        std::make_unique<Instruction>(Opcode::SLTIU, parent_bb);
                    instruction->addOperand(std::move(new_reg));  // rd
                    instruction->addOperand(std::move(rs1));      // rs1
                    instruction->addOperand(
                        std::make_unique<ImmediateOperand>(1));  // imm

                    parent_bb->addInstruction(std::move(instruction));
                    return std::make_unique<RegisterOperand>(
                        new_reg_ptr->getRegNum(), new_reg_ptr->isVirtual());
                }

                if (dest_type->getBitWidth() == 32 &&
                    src_type->getBitWidth() == 1) {
                    return immToReg(visit(cast_inst->getOperand(0), parent_bb),
                                    parent_bb);
                }
            }
            throw std::runtime_error("Unsupported trunc cast type: " +
                                     dest_type->toString());
        } break;

        default:
            throw std::runtime_error("Unsupported cast type: " +
                                     cast_inst->toString());
    }
}

// 修复 visitGEPInst 方法，支持全局变量作为基地址
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

    std::unique_ptr<RegisterOperand> base_addr_reg;

    if (base_addr->getType() == OperandType::FrameIndex) {
        // 处理基于栈帧的地址
        auto get_base_addr_inst =
            std::make_unique<Instruction>(Opcode::FRAMEADDR, parent_bb);
        base_addr_reg = codeGen_->allocateReg();
        get_base_addr_inst->addOperand(std::make_unique<RegisterOperand>(
            base_addr_reg->getRegNum(), base_addr_reg->isVirtual()));  // rd
        get_base_addr_inst->addOperand(std::move(base_addr));          // FI
        parent_bb->addInstruction(std::move(get_base_addr_inst));
    } else if (base_addr->getType() == OperandType::Register) {
        // 基地址已经在寄存器中（如全局变量地址）
        auto* reg_operand = dynamic_cast<RegisterOperand*>(base_addr.get());
        base_addr_reg = std::make_unique<RegisterOperand>(
            reg_operand->getRegNum(), reg_operand->isVirtual());
    } else {
        throw std::runtime_error(
            "Base address must be a frame index or register operand, got: " +
            base_addr->toString());
    }

    // 获取 strides 和索引
    auto strides = gep_inst->getStrides();
    if (strides.size() != gep_inst->getNumIndices()) {
        throw std::runtime_error("Strides size mismatch with indices count");
    }

    // 初始化总偏移量为0
    auto total_offset_reg = codeGen_->allocateReg();
    auto li_zero_inst = std::make_unique<Instruction>(Opcode::LI, parent_bb);
    li_zero_inst->addOperand(std::make_unique<RegisterOperand>(
        total_offset_reg->getRegNum(), total_offset_reg->isVirtual()));
    li_zero_inst->addOperand(std::make_unique<ImmediateOperand>(0));
    parent_bb->addInstruction(std::move(li_zero_inst));

    // 遍历所有索引，计算 index[i] * stride[i] 并累加
    for (unsigned i = 0; i < gep_inst->getNumIndices(); ++i) {
        auto* index_value = gep_inst->getIndex(i);
        auto index_operand = visit(index_value, parent_bb);
        auto stride = strides[i];

        std::cout << "Processing index " << i
                  << ": value = " << index_value->toString()
                  << ", stride = " << stride << std::endl;

        // 检查索引是否为常量0，如果是则跳过
        if (auto* const_int =
                midend::dyn_cast<midend::ConstantInt>(index_value)) {
            if (const_int->getSignedValue() == 0) {
                continue;  // 跳过索引为0的情况，不会产生偏移
            }
        }

        // 将索引转换为寄存器
        auto index_reg = immToReg(std::move(index_operand), parent_bb);

        // 计算 index * stride
        std::unique_ptr<RegisterOperand> offset_reg;

        if (stride == 0) {
            // stride为0，跳过
            continue;
        } else if (stride == 1) {
            // stride为1，直接使用索引
            offset_reg = std::move(index_reg);
        } else if ((stride & (stride - 1)) == 0) {
            // stride是2的幂，使用左移优化
            int shift_amount = 0;
            auto temp = static_cast<unsigned int>(stride);
            while (temp > 1) {
                temp >>= 1;
                shift_amount++;
            }

            offset_reg = codeGen_->allocateReg();
            auto slli_inst =
                std::make_unique<Instruction>(Opcode::SLLI, parent_bb);
            slli_inst->addOperand(std::make_unique<RegisterOperand>(
                offset_reg->getRegNum(), offset_reg->isVirtual()));
            slli_inst->addOperand(std::move(index_reg));
            slli_inst->addOperand(
                std::make_unique<ImmediateOperand>(shift_amount));
            parent_bb->addInstruction(std::move(slli_inst));
        } else {
            // 一般情况，使用乘法
            auto stride_reg = codeGen_->allocateReg();
            auto li_stride_inst =
                std::make_unique<Instruction>(Opcode::LI, parent_bb);
            li_stride_inst->addOperand(std::make_unique<RegisterOperand>(
                stride_reg->getRegNum(), stride_reg->isVirtual()));
            li_stride_inst->addOperand(
                std::make_unique<ImmediateOperand>(stride));
            parent_bb->addInstruction(std::move(li_stride_inst));

            offset_reg = codeGen_->allocateReg();
            auto mul_inst =
                std::make_unique<Instruction>(Opcode::MUL, parent_bb);
            mul_inst->addOperand(std::make_unique<RegisterOperand>(
                offset_reg->getRegNum(), offset_reg->isVirtual()));
            mul_inst->addOperand(std::move(index_reg));
            mul_inst->addOperand(std::move(stride_reg));
            parent_bb->addInstruction(std::move(mul_inst));
        }

        // 累加到总偏移量：total_offset += index * stride
        auto new_total_offset_reg = codeGen_->allocateReg();
        auto add_inst = std::make_unique<Instruction>(Opcode::ADD, parent_bb);
        add_inst->addOperand(std::make_unique<RegisterOperand>(
            new_total_offset_reg->getRegNum(),
            new_total_offset_reg->isVirtual()));
        add_inst->addOperand(std::make_unique<RegisterOperand>(
            total_offset_reg->getRegNum(), total_offset_reg->isVirtual()));
        add_inst->addOperand(std::move(offset_reg));
        parent_bb->addInstruction(std::move(add_inst));

        total_offset_reg = std::move(new_total_offset_reg);
    }

    // 计算最终地址：基地址 + 总偏移量
    auto final_addr_reg = codeGen_->allocateReg();
    auto final_add_inst = std::make_unique<Instruction>(Opcode::ADD, parent_bb);
    final_add_inst->addOperand(std::make_unique<RegisterOperand>(
        final_addr_reg->getRegNum(), final_addr_reg->isVirtual()));
    final_add_inst->addOperand(std::make_unique<RegisterOperand>(
        base_addr_reg->getRegNum(), base_addr_reg->isVirtual()));
    final_add_inst->addOperand(std::make_unique<RegisterOperand>(
        total_offset_reg->getRegNum(), total_offset_reg->isVirtual()));
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

    // 处理参数传递
    size_t num_args = called_func->getNumArgs();

    // 前8个参数通过寄存器传递
    for (size_t arg_i = 0; arg_i < std::min(num_args, size_t(8)); ++arg_i) {
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
        auto dest_arg_operand = funcArgToReg(dest_arg, parent_bb);

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

    // 超过8个的参数通过栈传递 - 调用者负责将参数存储到栈上
    if (num_args > 8) {
        for (size_t arg_i = 8; arg_i < num_args; ++arg_i) {
            auto* source_operand = call_inst->getArgOperand(arg_i);
            if (source_operand == nullptr) {
                throw std::runtime_error(
                    "Source operand for argument " + std::to_string(arg_i) +
                    " is null in call instruction: " + inst->toString());
            }

            // 获取参数值
            auto source_value = visit(source_operand, parent_bb);

            // 处理不同类型的操作数
            std::unique_ptr<RegisterOperand> source_reg;

            if (source_value->getType() == OperandType::FrameIndex) {
                // 如果是 FrameIndex，需要先获取其地址
                source_reg = immToReg(std::move(source_value), parent_bb);
            } else {
                // 对于其他类型（立即数、寄存器），使用原有逻辑
                source_reg = immToReg(std::move(source_value), parent_bb);
            }

            // 计算栈上的偏移量：第9个参数(index=8)放在0(sp)，第10个参数放在4(sp)，以此类推
            int stack_offset =
                static_cast<int>((arg_i - 8) * 4);  // 假设每个参数4字节

            // 使用新的辅助函数生成存储指令
            generateMemoryInstruction(Opcode::SW, std::move(source_reg),
                                      std::make_unique<RegisterOperand>("sp"),
                                      stack_offset, parent_bb);
        }
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

    // 分配一个公共虚拟寄存器用于PHI结果
    auto phi_reg = codeGen_->allocateReg();
    auto* parent_func = parent_bb->getParent();

    // 记录PHI的映射
    codeGen_->mapValueToReg(inst, phi_reg->getRegNum(), phi_reg->isVirtual());

    // 对每个前驱块，在其跳转指令前插入赋值
    for (unsigned i = 0; i < phi_inst->getNumIncomingValues(); ++i) {
        auto* incoming_value = phi_inst->getIncomingValue(i);
        auto* incoming_bb_midend = phi_inst->getIncomingBlock(i);
        auto* incoming_bb = parent_func->getBasicBlock(incoming_bb_midend);

        if (!incoming_bb) {
            throw std::runtime_error("Incoming block not found for PHI");
        }

        // 计算插入点：跳转指令前
        auto insert_pos = incoming_bb->end();
        if (insert_pos != incoming_bb->begin()) {
            --insert_pos;
            // 跳过末尾的PHI相关赋值（如果有），确保在跳转指令前
            // 这里假设最后一条是跳转指令
        }

        // 访问输入值（在前驱块上下文）
        auto value_operand = visit(incoming_value, incoming_bb);

        // 赋值到公共寄存器
        auto dest_reg = std::make_unique<RegisterOperand>(phi_reg->getRegNum(),
                                                          phi_reg->isVirtual());
        storeOperandToReg(std::move(value_operand), std::move(dest_reg),
                          incoming_bb, insert_pos);
    }

    // PHI块本身不生成任何指令，只返回寄存器
    return std::make_unique<RegisterOperand>(phi_reg->getRegNum(),
                                             phi_reg->isVirtual());
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

        if (condition->isImm()) {
            // 如果条件是立即数，直接跳转到真分支或假分支
            auto* imm_cond = dynamic_cast<ImmediateOperand*>(condition.get());
            if (imm_cond->getValue() != 0) {
                // 条件为真，跳转到真分支
                auto instruction =
                    std::make_unique<Instruction>(Opcode::J, parent_bb);
                instruction->addOperand(
                    std::make_unique<LabelOperand>(true_bb));
                parent_bb->addInstruction(std::move(instruction));
            } else {
                // 条件为假，跳转到假分支
                auto instruction =
                    std::make_unique<Instruction>(Opcode::J, parent_bb);
                instruction->addOperand(
                    std::make_unique<LabelOperand>(false_bb));
                parent_bb->addInstruction(std::move(instruction));
            }
            return;
        }

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

    // 第一阶段：只为alloca创建抽象Frame Index，不计算具体偏移
    auto* allocated_type = alloca_inst->getAllocatedType();

    // 计算类型大小
    std::function<size_t(const midend::Type*)> calculateTypeSize =
        [&](const midend::Type* type) -> size_t {
        if (type->isPointerType()) {
            return 8;  // 64位指针
        } else if (type->isIntegerType()) {
            return (type->getBitWidth() + 7) / 8;  // 向上取整到字节
        } else if (type->isArrayType()) {
            const auto* arrayType = midend::dyn_cast<midend::ArrayType>(type);
            size_t elementSize = calculateTypeSize(arrayType->getElementType());
            return elementSize * arrayType->getNumElements();
        } else {
            return 4;  // 默认大小
        }
    };

    size_t typeSize = calculateTypeSize(allocated_type);

    // 创建抽象的栈对象（第一阶段不分配具体偏移）
    int fi_id = sfm->createAllocaObject(inst, typeSize);

    return std::make_unique<FrameIndexOperand>(fi_id);
}

// 注释：这些函数已经在 Visit.cpp 中实现，FrameIndexElimination
// 中复用了相同的逻辑 未来可以考虑将这些函数提取到一个共同的工具类中，比如
// RISCVUtils.h/cpp

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

    // 处理指针操作数 - 可能是 alloca 指令、GEP 指令或全局变量
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

        // 生成frameaddr指令来获取栈地址（每次都使用新的寄存器）
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
            std::make_unique<RegisterOperand>(frame_addr_reg->getRegNum(),
                                              frame_addr_reg->isVirtual()),
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

    } else if (auto* global_var =
                   midend::dyn_cast<midend::GlobalVariable>(pointer_operand)) {
        // 是全局变量
        // 生成全局变量地址加载指令
        auto global_addr_reg = codeGen_->allocateReg();
        auto global_addr_inst =
            std::make_unique<Instruction>(Opcode::LA, parent_bb);
        global_addr_inst->addOperand(std::make_unique<RegisterOperand>(
            global_addr_reg->getRegNum(), global_addr_reg->isVirtual()));  // rd
        global_addr_inst->addOperand(std::make_unique<LabelOperand>(
            global_var->getName()));  // global symbol
        parent_bb->addInstruction(std::move(global_addr_inst));

        // 生成存储指令
        auto sw_inst = std::make_unique<Instruction>(Opcode::SW, parent_bb);
        sw_inst->addOperand(std::move(value_operand));  // source register
        sw_inst->addOperand(std::make_unique<MemoryOperand>(
            std::make_unique<RegisterOperand>(global_addr_reg->getRegNum(),
                                              global_addr_reg->isVirtual()),
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

    // 处理指针操作数 - 可能是 alloca 指令、GEP 指令或全局变量
    if (auto* alloca_inst =
            midend::dyn_cast<midend::AllocaInst>(pointer_operand)) {
        // 直接是 alloca 指令
        auto* sfm = parent_bb->getParent()->getStackFrameManager();
        int frame_id = sfm->getAllocaStackSlotId(alloca_inst);

        if (frame_id == -1) {
            throw std::runtime_error(
                "Cannot find frame index for alloca instruction in load" +
                midend::IRPrinter::toString(alloca_inst));
        }

        // 生成frameaddr指令来获取栈地址（每次都使用新的寄存器）
        auto frame_addr_reg = codeGen_->allocateReg();
        auto load_frame_addr_inst =
            std::make_unique<Instruction>(Opcode::FRAMEADDR, parent_bb);
        load_frame_addr_inst->addOperand(std::make_unique<RegisterOperand>(
            frame_addr_reg->getRegNum(), frame_addr_reg->isVirtual()));  // rd
        load_frame_addr_inst->addOperand(
            std::make_unique<FrameIndexOperand>(frame_id));  // FI
        parent_bb->addInstruction(std::move(load_frame_addr_inst));

        // 加载到新的寄存器（也使用新的寄存器）
        auto new_reg = codeGen_->allocateReg();

        // 使用新的辅助函数生成内存指令
        generateMemoryInstruction(
            Opcode::LW,
            std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                              new_reg->isVirtual()),
            std::make_unique<RegisterOperand>(frame_addr_reg->getRegNum(),
                                              frame_addr_reg->isVirtual()),
            0, parent_bb);

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

        // 使用新的辅助函数生成内存指令
        generateMemoryInstruction(
            Opcode::LW,
            std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                              new_reg->isVirtual()),
            std::make_unique<RegisterOperand>(address_reg->getRegNum(),
                                              address_reg->isVirtual()),
            0, parent_bb);

        // 建立load指令结果值到寄存器的映射
        codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                new_reg->isVirtual());

        return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                 new_reg->isVirtual());

    } else if (auto* global_var =
                   midend::dyn_cast<midend::GlobalVariable>(pointer_operand)) {
        // 是全局变量
        // 生成全局变量地址加载指令
        auto global_addr_reg = codeGen_->allocateReg();
        auto global_addr_inst =
            std::make_unique<Instruction>(Opcode::LA, parent_bb);
        global_addr_inst->addOperand(std::make_unique<RegisterOperand>(
            global_addr_reg->getRegNum(), global_addr_reg->isVirtual()));  // rd
        global_addr_inst->addOperand(std::make_unique<LabelOperand>(
            global_var->getName()));  // global symbol
        parent_bb->addInstruction(std::move(global_addr_inst));

        // 从全局变量地址加载值
        auto new_reg = codeGen_->allocateReg();

        // 使用新的辅助函数生成内存指令
        generateMemoryInstruction(
            Opcode::LW,
            std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                              new_reg->isVirtual()),
            std::make_unique<RegisterOperand>(global_addr_reg->getRegNum(),
                                              global_addr_reg->isVirtual()),
            0, parent_bb);

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

std::unique_ptr<MachineOperand> Visitor::visitUnaryOp(
    const midend::Instruction* inst, BasicBlock* parent_bb) {
    if (!inst->isUnaryOp()) {
        throw std::runtime_error("Not a unary operation instruction: " +
                                 inst->toString());
    }

    if (inst->getNumOperands() != 1) {
        throw std::runtime_error("Unary operation must have one operand, got " +
                                 std::to_string(inst->getNumOperands()));
    }

    // Handle UAdd (unary plus): +operand
    // This is essentially a no-op, just return the operand
    if (inst->getOpcode() == midend::Opcode::UAdd) {
        auto operand = visit(inst->getOperand(0), parent_bb);

        // If it's already a register, return it directly
        if (operand->getType() == OperandType::Register) {
            auto* reg_operand = dynamic_cast<RegisterOperand*>(operand.get());
            codeGen_->mapValueToReg(inst, reg_operand->getRegNum(),
                                    reg_operand->isVirtual());
            return std::make_unique<RegisterOperand>(reg_operand->getRegNum(),
                                                     reg_operand->isVirtual());
        }

        // If it's an immediate, we can return it directly or load to register
        if (operand->getType() == OperandType::Immediate) {
            auto* imm_operand = dynamic_cast<ImmediateOperand*>(operand.get());
            // For unary plus, the value remains the same
            return std::make_unique<ImmediateOperand>(imm_operand->getValue());
        }

        throw std::runtime_error("Unsupported operand type for UAdd");
    }

    // Handle USub (unary minus): -operand
    if (inst->getOpcode() == midend::Opcode::USub) {
        auto operand = visit(inst->getOperand(0), parent_bb);

        // If both are immediates, do constant folding
        if (operand->getType() == OperandType::Immediate) {
            auto* imm_operand = dynamic_cast<ImmediateOperand*>(operand.get());
            return std::make_unique<ImmediateOperand>(-imm_operand->getValue());
        }

        // Convert operand to register if needed
        auto operand_reg = immToReg(std::move(operand), parent_bb);

        // Generate sub instruction: 0 - operand
        auto new_reg = codeGen_->allocateReg();
        auto instruction =
            std::make_unique<Instruction>(Opcode::SUB, parent_bb);
        instruction->addOperand(std::make_unique<RegisterOperand>(
            new_reg->getRegNum(), new_reg->isVirtual()));  // rd
        instruction->addOperand(
            std::make_unique<RegisterOperand>("zero"));   // rs1 (zero register)
        instruction->addOperand(std::move(operand_reg));  // rs2
        parent_bb->addInstruction(std::move(instruction));

        codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                new_reg->isVirtual());
        return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                 new_reg->isVirtual());
    }

    // Handle Not: !operand
    if (inst->getOpcode() == midend::Opcode::Not) {
        auto operand = visit(inst->getOperand(0), parent_bb);

        // If it's an immediate, do constant folding
        if (operand->getType() == OperandType::Immediate) {
            auto* imm_operand = dynamic_cast<ImmediateOperand*>(operand.get());
            // Bitwise NOT
            return std::make_unique<ImmediateOperand>(
                (imm_operand->getValue()) == 0);
        }

        // Convert operand to register if needed
        auto operand_reg = immToReg(std::move(operand), parent_bb);

        // Generate sltiu instruction: operand sltiu 1
        auto new_reg = codeGen_->allocateReg();
        auto instruction =
            std::make_unique<Instruction>(Opcode::SLTIU, parent_bb);
        instruction->addOperand(std::make_unique<RegisterOperand>(
            new_reg->getRegNum(), new_reg->isVirtual()));                // rd
        instruction->addOperand(std::move(operand_reg));                 // rs1
        instruction->addOperand(std::make_unique<ImmediateOperand>(1));  // imm
        parent_bb->addInstruction(std::move(instruction));

        codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                new_reg->isVirtual());
        return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                 new_reg->isVirtual());
    }

    throw std::runtime_error("Unsupported unary operation: " +
                             inst->toString());
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

            // 处理立即数操作数，利用交换律
            if (lhs->getType() == OperandType::Immediate ||
                rhs->getType() == OperandType::Immediate) {
                // 使用 addi 指令
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::ADDI, parent_bb);

                auto* imm_operand = dynamic_cast<ImmediateOperand*>(
                    lhs->getType() == OperandType::Immediate ? lhs.get()
                                                             : rhs.get());
                auto* reg_operand = dynamic_cast<RegisterOperand*>(
                    (lhs->getType() == OperandType::Register ? lhs.get()
                                                             : rhs.get()));

                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    reg_operand->getRegNum(),
                    reg_operand->isVirtual()));  // rs1
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

            // 处理右侧立即数：a - imm => addi a, -imm
            if (rhs->getType() == OperandType::Immediate) {
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::ADDI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::move(lhs));           // rs1
                instruction->addOperand(std::make_unique<ImmediateOperand>(
                    -rhs_imm->getValue()));  // -imm
                parent_bb->addInstruction(std::move(instruction));
            } else {
                // 其他情况使用寄存器-寄存器的 sub
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
            }
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

        case midend::Opcode::Div: {
            // 处理有符号除法
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                if (rhs_imm->getValue() == 0) {
                    throw std::runtime_error("Division by zero");
                }
                return std::make_unique<ImmediateOperand>(lhs_imm->getValue() /
                                                          rhs_imm->getValue());
            }

            auto lhs_reg = immToReg(std::move(lhs), parent_bb);
            auto rhs_reg = immToReg(std::move(rhs), parent_bb);

            new_reg = codeGen_->allocateReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::DIV, parent_bb);
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

        case midend::Opcode::And: {
            // 处理按位与运算
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(lhs_imm->getValue() &
                                                          rhs_imm->getValue());
            }

            // 处理立即数操作数，利用交换律使用 andi
            if (lhs->getType() == OperandType::Immediate ||
                rhs->getType() == OperandType::Immediate) {
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::ANDI, parent_bb);

                auto* imm_operand = dynamic_cast<ImmediateOperand*>(
                    lhs->getType() == OperandType::Immediate ? lhs.get()
                                                             : rhs.get());
                auto* reg_operand = dynamic_cast<RegisterOperand*>(
                    (lhs->getType() == OperandType::Register ? lhs.get()
                                                             : rhs.get()));

                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    reg_operand->getRegNum(),
                    reg_operand->isVirtual()));  // rs1
                instruction->addOperand(std::make_unique<ImmediateOperand>(
                    imm_operand->getValue()));  // imm

                parent_bb->addInstruction(std::move(instruction));
            } else {
                auto lhs_reg = immToReg(std::move(lhs), parent_bb);
                auto rhs_reg = immToReg(std::move(rhs), parent_bb);

                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::AND, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::move(lhs_reg));       // rs1
                instruction->addOperand(std::move(rhs_reg));       // rs2

                parent_bb->addInstruction(std::move(instruction));
            }
            break;
        }

        case midend::Opcode::Or: {
            // 处理按位或运算
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(lhs_imm->getValue() |
                                                          rhs_imm->getValue());
            }

            // 处理立即数操作数，利用交换律使用 ori
            if (lhs->getType() == OperandType::Immediate ||
                rhs->getType() == OperandType::Immediate) {
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::ORI, parent_bb);

                auto* imm_operand = dynamic_cast<ImmediateOperand*>(
                    lhs->getType() == OperandType::Immediate ? lhs.get()
                                                             : rhs.get());
                auto* reg_operand = dynamic_cast<RegisterOperand*>(
                    (lhs->getType() == OperandType::Register ? lhs.get()
                                                             : rhs.get()));

                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    reg_operand->getRegNum(),
                    reg_operand->isVirtual()));  // rs1
                instruction->addOperand(std::make_unique<ImmediateOperand>(
                    imm_operand->getValue()));  // imm

                parent_bb->addInstruction(std::move(instruction));
            } else {
                auto lhs_reg = immToReg(std::move(lhs), parent_bb);
                auto rhs_reg = immToReg(std::move(rhs), parent_bb);

                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::OR, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::move(lhs_reg));       // rs1
                instruction->addOperand(std::move(rhs_reg));       // rs2

                parent_bb->addInstruction(std::move(instruction));
            }
            break;
        }

        case midend::Opcode::Xor: {
            // 处理按位异或运算
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(lhs_imm->getValue() ^
                                                          rhs_imm->getValue());
            }

            // 处理立即数操作数，利用交换律使用 xori
            if (lhs->getType() == OperandType::Immediate ||
                rhs->getType() == OperandType::Immediate) {
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::XORI, parent_bb);

                auto* imm_operand = dynamic_cast<ImmediateOperand*>(
                    lhs->getType() == OperandType::Immediate ? lhs.get()
                                                             : rhs.get());
                auto* reg_operand = dynamic_cast<RegisterOperand*>(
                    (lhs->getType() == OperandType::Register ? lhs.get()
                                                             : rhs.get()));

                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    reg_operand->getRegNum(),
                    reg_operand->isVirtual()));  // rs1
                instruction->addOperand(std::make_unique<ImmediateOperand>(
                    imm_operand->getValue()));  // imm

                parent_bb->addInstruction(std::move(instruction));
            } else {
                auto lhs_reg = immToReg(std::move(lhs), parent_bb);
                auto rhs_reg = immToReg(std::move(rhs), parent_bb);

                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::XOR, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::move(lhs_reg));       // rs1
                instruction->addOperand(std::move(rhs_reg));       // rs2

                parent_bb->addInstruction(std::move(instruction));
            }
            break;
        }

        case midend::Opcode::Shl: {
            // 处理左移运算
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    lhs_imm->getValue() << rhs_imm->getValue());
            }

            // 处理右侧立即数：使用 slli
            if (rhs->getType() == OperandType::Immediate) {
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::SLLI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::move(lhs));           // rs1
                instruction->addOperand(std::make_unique<ImmediateOperand>(
                    rhs_imm->getValue()));  // shamt
                parent_bb->addInstruction(std::move(instruction));
            } else {
                auto lhs_reg = immToReg(std::move(lhs), parent_bb);
                auto rhs_reg = immToReg(std::move(rhs), parent_bb);

                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::SLL, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::move(lhs_reg));       // rs1
                instruction->addOperand(std::move(rhs_reg));       // rs2

                parent_bb->addInstruction(std::move(instruction));
            }
            break;
        }

        case midend::Opcode::Shr: {
            // 处理右移运算（算术右移）
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                // 对于有符号整数，使用算术右移
                return std::make_unique<ImmediateOperand>(lhs_imm->getValue() >>
                                                          rhs_imm->getValue());
            }

            // 处理右侧立即数：使用 srai
            if (rhs->getType() == OperandType::Immediate) {
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::SRAI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::move(lhs));           // rs1
                instruction->addOperand(std::make_unique<ImmediateOperand>(
                    rhs_imm->getValue()));  // shamt
                parent_bb->addInstruction(std::move(instruction));
            } else {
                auto lhs_reg = immToReg(std::move(lhs), parent_bb);
                auto rhs_reg = immToReg(std::move(rhs), parent_bb);

                new_reg = codeGen_->allocateReg();
                // 使用算术右移（保持符号位）
                auto instruction =
                    std::make_unique<Instruction>(Opcode::SRA, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::move(lhs_reg));       // rs1
                instruction->addOperand(std::move(rhs_reg));       // rs2

                parent_bb->addInstruction(std::move(instruction));
            }
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

            // 优化：a > imm 可以转换为 a >= (imm+1)，然后用 !(a < (imm+1))
            if (rhs->getType() == OperandType::Immediate) {
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                // a > imm 等价于 !(a <= imm) 等价于 !(a < (imm+1))
                new_reg = codeGen_->allocateReg();
                auto slti_inst =
                    std::make_unique<Instruction>(Opcode::SLTI, parent_bb);
                slti_inst->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                slti_inst->addOperand(std::move(lhs));             // rs1
                slti_inst->addOperand(std::make_unique<ImmediateOperand>(
                    rhs_imm->getValue() + 1));  // imm+1
                parent_bb->addInstruction(std::move(slti_inst));

                // 对结果取反
                auto result_reg = codeGen_->allocateReg();
                auto xori_inst =
                    std::make_unique<Instruction>(Opcode::XORI, parent_bb);
                xori_inst->addOperand(std::make_unique<RegisterOperand>(
                    result_reg->getRegNum(),
                    result_reg->isVirtual()));  // rd
                xori_inst->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rs1
                xori_inst->addOperand(
                    std::make_unique<ImmediateOperand>(1));  // 1
                parent_bb->addInstruction(std::move(xori_inst));

                new_reg = std::move(result_reg);
            } else {
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
            }
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

            // 优化：a == imm 可以用 addi + seqz
            if (lhs->getType() == OperandType::Immediate ||
                rhs->getType() == OperandType::Immediate) {
                auto* imm_operand = dynamic_cast<ImmediateOperand*>(
                    lhs->getType() == OperandType::Immediate ? lhs.get()
                                                             : rhs.get());
                auto* reg_operand = dynamic_cast<RegisterOperand*>(
                    (lhs->getType() == OperandType::Register ? lhs.get()
                                                             : rhs.get()));

                // a == imm 等价于 (a - imm) == 0
                auto sub_reg = codeGen_->allocateReg();
                auto addi_inst =
                    std::make_unique<Instruction>(Opcode::ADDI, parent_bb);
                addi_inst->addOperand(std::make_unique<RegisterOperand>(
                    sub_reg->getRegNum(), sub_reg->isVirtual()));  // rd
                addi_inst->addOperand(std::make_unique<RegisterOperand>(
                    reg_operand->getRegNum(),
                    reg_operand->isVirtual()));  // rs1
                addi_inst->addOperand(std::make_unique<ImmediateOperand>(
                    -imm_operand->getValue()));  // -imm
                parent_bb->addInstruction(std::move(addi_inst));

                new_reg = codeGen_->allocateReg();
                auto seqz_inst =
                    std::make_unique<Instruction>(Opcode::SEQZ, parent_bb);
                seqz_inst->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                seqz_inst->addOperand(std::make_unique<RegisterOperand>(
                    sub_reg->getRegNum(), sub_reg->isVirtual()));  // rs1
                parent_bb->addInstruction(std::move(seqz_inst));
            } else {
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
            }
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

            // 优化：a != imm 可以用 addi + snez
            if (lhs->getType() == OperandType::Immediate ||
                rhs->getType() == OperandType::Immediate) {
                auto* imm_operand = dynamic_cast<ImmediateOperand*>(
                    lhs->getType() == OperandType::Immediate ? lhs.get()
                                                             : rhs.get());
                auto* reg_operand = dynamic_cast<RegisterOperand*>(
                    lhs->getType() == OperandType::Register ? lhs.get()
                                                            : rhs.get());

                // a != imm 等价于 (a - imm) != 0
                auto sub_reg = codeGen_->allocateReg();
                auto addi_inst =
                    std::make_unique<Instruction>(Opcode::ADDI, parent_bb);
                addi_inst->addOperand(std::make_unique<RegisterOperand>(
                    sub_reg->getRegNum(), sub_reg->isVirtual()));  // rd
                addi_inst->addOperand(std::make_unique<RegisterOperand>(
                    reg_operand->getRegNum(),
                    reg_operand->isVirtual()));  // rs1
                addi_inst->addOperand(std::make_unique<ImmediateOperand>(
                    -imm_operand->getValue()));  // -imm
                parent_bb->addInstruction(std::move(addi_inst));

                new_reg = codeGen_->allocateReg();
                auto snez_inst =
                    std::make_unique<Instruction>(Opcode::SNEZ, parent_bb);
                snez_inst->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                snez_inst->addOperand(std::make_unique<RegisterOperand>(
                    sub_reg->getRegNum(), sub_reg->isVirtual()));  // rs1
                parent_bb->addInstruction(std::move(snez_inst));
            } else {
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
            }
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

        case midend::Opcode::ICmpSLE: {
            // 处理有符号小于等于比较 (a <= b 等价于 !(a > b))
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    lhs_imm->getValue() <= rhs_imm->getValue() ? 1 : 0);
            }

            auto lhs_reg = immToReg(std::move(lhs), parent_bb);
            auto rhs_reg = immToReg(std::move(rhs), parent_bb);

            // 使用 sgt 指令然后取反: !(lhs > rhs) = (lhs <= rhs)
            auto sgt_reg = codeGen_->allocateReg();
            auto sgt_inst =
                std::make_unique<Instruction>(Opcode::SGT, parent_bb);
            sgt_inst->addOperand(std::make_unique<RegisterOperand>(
                sgt_reg->getRegNum(), sgt_reg->isVirtual()));  // rd
            sgt_inst->addOperand(std::move(lhs_reg));          // rs1
            sgt_inst->addOperand(std::move(rhs_reg));          // rs2
            parent_bb->addInstruction(std::move(sgt_inst));

            new_reg = codeGen_->allocateReg();
            auto seqz_inst =
                std::make_unique<Instruction>(Opcode::SEQZ, parent_bb);
            seqz_inst->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual()));  // rd
            seqz_inst->addOperand(std::move(sgt_reg));         // rs1
            parent_bb->addInstruction(std::move(seqz_inst));
            break;
        }

        case midend::Opcode::ICmpSGE: {
            // 处理有符号大于等于比较 (a >= b 等价于 !(a < b))
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    lhs_imm->getValue() >= rhs_imm->getValue() ? 1 : 0);
            }

            auto lhs_reg = immToReg(std::move(lhs), parent_bb);
            auto rhs_reg = immToReg(std::move(rhs), parent_bb);

            // 使用 slt 指令然后取反: !(lhs < rhs) = (lhs >= rhs)
            auto slt_reg = codeGen_->allocateReg();
            auto slt_inst =
                std::make_unique<Instruction>(Opcode::SLT, parent_bb);
            slt_inst->addOperand(std::make_unique<RegisterOperand>(
                slt_reg->getRegNum(), slt_reg->isVirtual()));  // rd
            slt_inst->addOperand(std::move(lhs_reg));          // rs1
            slt_inst->addOperand(std::move(rhs_reg));          // rs2
            parent_bb->addInstruction(std::move(slt_inst));

            new_reg = codeGen_->allocateReg();
            auto seqz_inst =
                std::make_unique<Instruction>(Opcode::SEQZ, parent_bb);
            seqz_inst->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual()));  // rd
            seqz_inst->addOperand(std::move(slt_reg));         // rs1
            parent_bb->addInstruction(std::move(seqz_inst));
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
            case OperandType::FrameIndex: {
                auto inst =
                    std::make_unique<Instruction>(Opcode::FRAMEADDR, parent_bb);
                auto* frame_source =
                    dynamic_cast<FrameIndexOperand*>(source_operand.get());

                inst->addOperand(std::move(dest_reg));  // rd
                inst->addOperand(std::make_unique<FrameIndexOperand>(
                    frame_source->getIndex()));  // FI
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
    if (ret_inst->getNumOperands() > 1) {
        throw std::runtime_error(
            "Return instruction must no more than one operand, got " +
            std::to_string(ret_inst->getNumOperands()));
    }

    if (ret_inst->getNumOperands() == 0) {
        // 无返回值，直接添加返回指令
        auto riscv_ret_inst =
            std::make_unique<Instruction>(Opcode::RET, parent_bb);
        parent_bb->addInstruction(std::move(riscv_ret_inst));
        return;
    }

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
    const midend::Argument* argument, BasicBlock* parent_bb) {
    // 获取函数参数对应的物理寄存器或者栈帧
    if (argument->getArgNo() < 8) {
        // 如果参数编号小于的寄存器数量，直接返回对应的寄存器
        return std::make_unique<RegisterOperand>(
            "a" + std::to_string(argument->getArgNo()));
    }

    // 对于超过8个的参数，需要从栈上读取
    // 被调用者需要考虑自己的函数序言对sp的修改
    if (parent_bb == nullptr) {
        throw std::runtime_error(
            "Cannot generate load instruction for stack argument without "
            "BasicBlock context");
    }

    // 计算正确的偏移量
    int arg_offset = (argument->getArgNo() - 8) * 4;

    auto arg_reg = codeGen_->allocateReg();

    // 使用新的辅助函数生成加载指令
    generateMemoryInstruction(
        Opcode::LW,
        std::make_unique<RegisterOperand>(arg_reg->getRegNum(),
                                          arg_reg->isVirtual()),
        std::make_unique<RegisterOperand>("s0"),  // 使用帧指针
        arg_offset, parent_bb);

    return std::make_unique<RegisterOperand>(arg_reg->getRegNum(),
                                             arg_reg->isVirtual());
}

// 检查偏移量是否在有效的立即数范围内（-2048 到 +2047）
bool Visitor::isValidImmediateOffset(int64_t offset) {
    return offset >= -2048 && offset <= 2047;
}

// 处理大偏移量：生成临时寄存器并计算地址
std::unique_ptr<RegisterOperand> Visitor::handleLargeOffset(
    std::unique_ptr<RegisterOperand> base_reg, int64_t offset,
    BasicBlock* parent_bb) {
    if (isValidImmediateOffset(offset)) {
        // 偏移量在有效范围内，直接返回原始寄存器
        return base_reg;
    }

    // 偏移量超出范围，需要使用临时寄存器计算地址
    auto temp_reg = codeGen_->allocateReg();

    // 将偏移量加载到临时寄存器
    auto li_inst = std::make_unique<Instruction>(Opcode::LI, parent_bb);
    li_inst->addOperand(std::make_unique<RegisterOperand>(
        temp_reg->getRegNum(), temp_reg->isVirtual()));
    li_inst->addOperand(std::make_unique<ImmediateOperand>(offset));
    parent_bb->addInstruction(std::move(li_inst));

    // 计算最终地址：base + offset
    auto addr_reg = codeGen_->allocateReg();
    auto add_inst = std::make_unique<Instruction>(Opcode::ADD, parent_bb);
    add_inst->addOperand(std::make_unique<RegisterOperand>(
        addr_reg->getRegNum(), addr_reg->isVirtual()));
    add_inst->addOperand(std::move(base_reg));
    add_inst->addOperand(std::move(temp_reg));
    parent_bb->addInstruction(std::move(add_inst));

    return addr_reg;
}

// 生成内存指令，自动处理大偏移量
void Visitor::generateMemoryInstruction(
    Opcode opcode, std::unique_ptr<RegisterOperand> target_reg,
    std::unique_ptr<RegisterOperand> base_reg, int64_t offset,
    BasicBlock* parent_bb) {
    if (isValidImmediateOffset(offset)) {
        // 偏移量在有效范围内，直接生成指令
        auto inst = std::make_unique<Instruction>(opcode, parent_bb);
        inst->addOperand(std::move(target_reg));
        inst->addOperand(std::make_unique<MemoryOperand>(
            std::move(base_reg), std::make_unique<ImmediateOperand>(offset)));
        parent_bb->addInstruction(std::move(inst));
    } else {
        // 偏移量超出范围，先计算地址
        auto addr_reg =
            handleLargeOffset(std::move(base_reg), offset, parent_bb);

        // 使用计算出的地址和 0 偏移量生成指令
        auto inst = std::make_unique<Instruction>(opcode, parent_bb);
        inst->addOperand(std::move(target_reg));
        inst->addOperand(std::make_unique<MemoryOperand>(
            std::move(addr_reg), std::make_unique<ImmediateOperand>(0)));
        parent_bb->addInstruction(std::move(inst));
    }
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

    // 检查是否是全局变量
    if (auto* global_var = midend::dyn_cast<midend::GlobalVariable>(value)) {
        // 对于全局变量，生成LA指令来获取其地址
        auto global_addr_reg = codeGen_->allocateReg();
        auto global_addr_inst =
            std::make_unique<Instruction>(Opcode::LA, parent_bb);
        global_addr_inst->addOperand(std::make_unique<RegisterOperand>(
            global_addr_reg->getRegNum(), global_addr_reg->isVirtual()));  // rd
        global_addr_inst->addOperand(std::make_unique<LabelOperand>(
            global_var->getName()));  // global symbol
        parent_bb->addInstruction(std::move(global_addr_inst));

        // 建立全局变量到寄存器的映射
        codeGen_->mapValueToReg(value, global_addr_reg->getRegNum(),
                                global_addr_reg->isVirtual());

        return std::make_unique<RegisterOperand>(global_addr_reg->getRegNum(),
                                                 global_addr_reg->isVirtual());
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
        // 参数应该已经在函数开头被转移到虚拟寄存器了
        const auto foundReg = findRegForValue(value);
        if (foundReg.has_value()) {
            return std::make_unique<RegisterOperand>(
                foundReg.value()->getRegNum(), foundReg.value()->isVirtual());
        }
        throw std::runtime_error(
            "Function argument not found in register mapping: " +
            value->toString());
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
        // 全局变量的指针类型处理已在上面处理了
        // 其他指针类型的处理
        throw std::runtime_error("Pointer type not handled: " +
                                 value->toString());
    }

    throw std::runtime_error(
        "Unsupported value type: " + value->getName() + " (type: " +
        std::to_string(static_cast<int>(value->getValueKind())) + ")");
}

// 访问常量
void Visitor::visit(const midend::Constant* /*constant*/) {}

// 辅助函数：转换LLVM类型到CompilerType
// 修复类型转换函数，支持多维数组
CompilerType Visitor::convertLLVMTypeToCompilerType(
    const midend::Type* llvm_type) {
    if (llvm_type->isIntegerType()) {
        return CompilerType(BaseType::INT32);
    } else if (llvm_type->isFloatType()) {
        return CompilerType(BaseType::FLOAT32);
    } else if (llvm_type->isArrayType()) {
        auto* array_type = static_cast<const midend::ArrayType*>(llvm_type);
        auto* element_type = array_type->getElementType();

        // 处理多维数组
        std::vector<size_t> dimensions;
        const midend::Type* current_type = llvm_type;

        // 收集所有维度
        while (current_type->isArrayType()) {
            auto* arr_type =
                static_cast<const midend::ArrayType*>(current_type);
            dimensions.push_back(arr_type->getNumElements());
            current_type = arr_type->getElementType();
        }

        // 确定基础类型
        BaseType base_type;
        if (current_type->isIntegerType()) {
            base_type = BaseType::INT32;
        } else if (current_type->isFloatType()) {
            base_type = BaseType::FLOAT32;
        } else {
            throw std::runtime_error("Unsupported array element type");
        }

        // 创建多维数组类型
        if (dimensions.size() == 1) {
            return CompilerType(base_type, dimensions[0]);
        } else {
            // 对于多维数组，我们可以将其视为一维数组，总大小为所有维度的乘积
            size_t total_size = 1;
            for (size_t dim : dimensions) {
                total_size *= dim;
            }
            return CompilerType(base_type, total_size);
        }
    }

    throw std::runtime_error("Unsupported global variable type: " +
                             llvm_type->toString());
}

// 辅助函数：转换LLVM初始化器到ConstantInitializer
ConstantInitializer Visitor::convertLLVMInitializerToConstantInitializer(
    const midend::Value* init, const midend::Type* type) {
    std::cout << "Converting initializer: " << init->toString()
              << " for type: " << type->toString() << std::endl;

    // 处理单个整数常量
    if (init->getType()->isIntegerType()) {
        const auto* const_int = midend::dyn_cast<midend::ConstantInt>(init);
        if (!const_int) {
            throw std::runtime_error("Expected ConstantInt for integer type");
        }
        int32_t value = static_cast<int32_t>(const_int->getSignedValue());
        std::cout << "Found ConstantInt: " << value << std::endl;
        return value;
    }

    // 处理单个浮点常量
    if (init->getType()->isFloatType()) {
        const auto* const_float = midend::dyn_cast<midend::ConstantFP>(init);
        if (!const_float) {
            throw std::runtime_error("Expected ConstantFP for float type");
        }
        float value = const_float->getValue();
        std::cout << "Found ConstantFP: " << value << std::endl;
        return value;
    }

    // 处理数组常量
    if (init->getType()->isArrayType()) {
        const auto* const_array = midend::dyn_cast<midend::ConstantArray>(init);
        if (!const_array) {
            throw std::runtime_error("Expected ConstantArray for array type");
        }

        std::cout << "Processing ConstantArray with "
                  << const_array->getNumElements() << " elements" << std::endl;

        const auto* array_type =
            static_cast<const midend::ArrayType*>(init->getType());
        auto* element_type = array_type->getElementType();
        std::cout << "Array element type: " << element_type->toString()
                  << std::endl;

        // 递归处理嵌套数组或基本类型元素
        if (element_type->isArrayType()) {
            // 多维数组：需要展平处理
            std::vector<int32_t> flattened_values;

            // Get the expected size of each sub-array
            const auto* sub_array_type =
                static_cast<const midend::ArrayType*>(element_type);
            size_t sub_array_size = sub_array_type->getNumElements();

            // Get the expected number of sub-arrays
            const auto* outer_array_type =
                static_cast<const midend::ArrayType*>(type);
            size_t num_sub_arrays = outer_array_type->getNumElements();

            for (unsigned i = 0; i < num_sub_arrays; ++i) {
                if (i < const_array->getNumElements()) {
                    // Process explicitly initialized sub-array
                    auto* element = const_array->getElement(i);
                    std::cout << "Processing nested array element " << i << ": "
                              << element->toString() << std::endl;

                    auto nested_init =
                        convertLLVMInitializerToConstantInitializer(
                            element, element_type);

                    // Track how many elements we've added for this sub-array
                    size_t sub_array_start = flattened_values.size();

                    // 将嵌套数组的值添加到展平数组中
                    std::visit(
                        [&flattened_values](const auto& value) {
                            using T = std::decay_t<decltype(value)>;
                            if constexpr (std::is_same_v<
                                              T, std::vector<int32_t>>) {
                                flattened_values.insert(flattened_values.end(),
                                                        value.begin(),
                                                        value.end());
                            } else if constexpr (std::is_same_v<T, int32_t>) {
                                flattened_values.push_back(value);
                            }
                            // 对于其他类型，暂时忽略
                        },
                        nested_init);

                    // Pad with zeros if the sub-array is not fully initialized
                    size_t elements_added =
                        flattened_values.size() - sub_array_start;
                    if (elements_added < sub_array_size) {
                        flattened_values.insert(flattened_values.end(),
                                                sub_array_size - elements_added,
                                                0);
                    }
                } else {
                    // No initializer for this sub-array, fill with zeros
                    flattened_values.insert(flattened_values.end(),
                                            sub_array_size, 0);
                }
            }

            std::cout << "Flattened array size: " << flattened_values.size()
                      << std::endl;
            return flattened_values;

        } else if (element_type->isIntegerType()) {
            // 一维整数数组
            std::vector<int32_t> values;

            // Get the expected array size from the type
            const auto* array_type =
                static_cast<const midend::ArrayType*>(type);
            size_t expected_size = array_type->getNumElements();
            values.reserve(expected_size);

            for (unsigned i = 0; i < const_array->getNumElements(); ++i) {
                auto* element = const_array->getElement(i);
                std::cout << "Processing int array element " << i << ": "
                          << element->toString() << std::endl;

                if (const auto* const_int =
                        midend::dyn_cast<midend::ConstantInt>(element)) {
                    int32_t value =
                        static_cast<int32_t>(const_int->getSignedValue());
                    values.push_back(value);
                    std::cout << "  -> value: " << value << std::endl;
                } else {
                    // 对于非常量元素，默认为0
                    std::cout << "  -> default value: 0" << std::endl;
                    values.push_back(0);
                }
            }

            // Pad with zeros if the initializer is smaller than the array
            if (values.size() < expected_size) {
                values.insert(values.end(), expected_size - values.size(), 0);
            }

            std::cout << "Created int array with " << values.size()
                      << " elements" << std::endl;
            return values;

        } else if (element_type->isFloatType()) {
            // 一维浮点数组
            std::vector<float> values;

            // Get the expected array size from the type
            const auto* array_type =
                static_cast<const midend::ArrayType*>(type);
            size_t expected_size = array_type->getNumElements();
            values.reserve(expected_size);

            for (unsigned i = 0; i < const_array->getNumElements(); ++i) {
                auto* element = const_array->getElement(i);
                std::cout << "Processing float array element " << i << ": "
                          << element->toString() << std::endl;

                if (const auto* const_float =
                        midend::dyn_cast<midend::ConstantFP>(element)) {
                    float value = const_float->getValue();
                    values.push_back(value);
                    std::cout << "  -> value: " << value << std::endl;
                } else {
                    // 对于非常量元素，默认为0.0
                    std::cout << "  -> default value: 0.0" << std::endl;
                    values.push_back(0.0F);
                }
            }

            // Pad with zeros if the initializer is smaller than the array
            if (values.size() < expected_size) {
                values.insert(values.end(), expected_size - values.size(),
                              0.0F);
            }

            std::cout << "Created float array with " << values.size()
                      << " elements" << std::endl;
            return values;
        }
    }

    // 检查是否为零初始化数组（通过类型判断）
    if (type->isArrayType()) {
        const auto* array_type = static_cast<const midend::ArrayType*>(type);
        auto* element_type = array_type->getElementType();

        // 计算总元素数量（支持多维数组）
        size_t total_elements = 1;
        const midend::Type* current_type = type;
        while (current_type->isArrayType()) {
            auto* arr_type =
                static_cast<const midend::ArrayType*>(current_type);
            total_elements *= arr_type->getNumElements();
            current_type = arr_type->getElementType();
        }

        std::cout << "Creating zero-initialized array with " << total_elements
                  << " elements" << std::endl;

        if (current_type->isIntegerType()) {
            std::vector<int32_t> zero_values(total_elements, 0);
            return zero_values;
        } else if (current_type->isFloatType()) {
            std::vector<float> zero_values(total_elements, 0.0f);
            return zero_values;
        }
    }

    // 对于其他情况，返回零初始化
    std::cout << "Returning ZeroInitializer for unhandled case" << std::endl;
    return ZeroInitializer{};
}

bool checkIfZeroInitializer(const ConstantInitializer& init) {
    return std::visit(
        [](const auto& value) -> bool {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, int32_t>) {
                return value == 0;
            } else if constexpr (std::is_same_v<T, float>) {
                return value == 0.0f;
            } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
                return std::all_of(value.begin(), value.end(),
                                   [](int32_t v) { return v == 0; });
            } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                return std::all_of(value.begin(), value.end(),
                                   [](float v) { return v == 0.0f; });
            } else if constexpr (std::is_same_v<T, ZeroInitializer>) {
                return true;
            }
            return false;
        },
        init);
}

// 访问 global variable
void Visitor::visit(const midend::GlobalVariable* global_var,
                    Module* parent_module) {
    std::string name = global_var->getName();
    bool is_constant = global_var->isConstant();

    std::cout << "Processing global variable: " << name
              << ", is_constant: " << is_constant
              << ", has_initializer: " << global_var->hasInitializer()
              << std::endl;

    // 转换类型信息
    auto* llvm_type = global_var->getValueType();
    CompilerType compiler_type = convertLLVMTypeToCompilerType(llvm_type);

    std::cout << "Converted type - base: "
              << (compiler_type.base == BaseType::INT32 ? "INT32" : "FLOAT32")
              << ", array_size: " << compiler_type.array_size << std::endl;

    // 处理初始化器
    std::optional<ConstantInitializer> initializer;

    if (global_var->hasInitializer()) {
        auto* init =
            const_cast<midend::GlobalVariable*>(global_var)->getInitializer();
        std::cout << "Found initializer: " << init->toString() << std::endl;

        try {
            initializer =
                convertLLVMInitializerToConstantInitializer(init, llvm_type);
            std::cout << "Initializer processed successfully for " << name
                      << std::endl;

            // 检查是否为零初始化
            bool is_zero_init = checkIfZeroInitializer(initializer.value());
            std::cout << "Is zero initializer: " << is_zero_init << std::endl;

            if (is_zero_init) {
                // 零初始化应该放到 BSS 段
                std::cout << "Converting to ZeroInitializer for BSS section"
                          << std::endl;
                initializer = ZeroInitializer{};
            } else {
                // 打印非零初始化的详细信息
                std::visit(
                    [&name](const auto& value) {
                        using T = std::decay_t<decltype(value)>;
                        if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
                            std::cout << "Non-zero int array initializer for "
                                      << name << " with " << value.size()
                                      << " elements: ";
                            for (size_t i = 0;
                                 i < std::min(value.size(), size_t(10)); ++i) {
                                std::cout << value[i] << " ";
                            }
                            if (value.size() > 10) std::cout << "...";
                            std::cout << std::endl;
                        } else if constexpr (std::is_same_v<
                                                 T, std::vector<float>>) {
                            std::cout << "Non-zero float array initializer for "
                                      << name << " with " << value.size()
                                      << " elements" << std::endl;
                        }
                    },
                    initializer.value());
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing initializer for " << name << ": "
                      << e.what() << std::endl;
            // 发生错误时，使用零初始化
            initializer = ZeroInitializer{};
        }
    } else {
        std::cout << "No initializer found for " << name
                  << ", using ZeroInitializer" << std::endl;
        // 没有初始化器也放到 BSS 段
        initializer = ZeroInitializer{};
    }

    // 创建 GlobalVariable 对象
    GlobalVariable global_variable(name, compiler_type, is_constant,
                                   initializer);

    // 添加到模块中
    if (parent_module != nullptr) {
        parent_module->addGlobal(std::move(global_variable));
        std::cout << "Global variable " << name << " added to module"
                  << std::endl;
    }
}

}  // namespace riscv64