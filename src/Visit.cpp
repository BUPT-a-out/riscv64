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
                const auto* argument = arg_it->get();
                bool is_float_arg = argument->getType()->isFloatType();

                // 根据参数类型分配正确的寄存器类型
                std::unique_ptr<RegisterOperand> new_reg;
                if (is_float_arg) {
                    new_reg = codeGen_->allocateFloatReg();
                } else {
                    new_reg = codeGen_->allocateReg();
                }

                codeGen_->mapValueToReg(argument, new_reg->getRegNum(),
                                        new_reg->isVirtual());

                // 获取参数的源寄存器或栈位置
                auto source_reg = funcArgToReg(argument, first_riscv_bb);

                // 生成参数转移指令
                auto dest_reg = std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    is_float_arg ? RegisterType::Float : RegisterType::Integer);

                if (is_float_arg) {
                    // 浮点参数的处理
                    auto* source_reg_operand =
                        dynamic_cast<RegisterOperand*>(source_reg.get());
                    if (source_reg_operand) {
                        // 如果源是浮点寄存器，使用浮点移动指令
                        if (source_reg_operand->isFloatRegister()) {
                            auto fmov_inst = std::make_unique<Instruction>(
                                Opcode::FMOV_S, first_riscv_bb);
                            fmov_inst->addOperand(std::move(dest_reg));
                            fmov_inst->addOperand(std::move(source_reg));
                            first_riscv_bb->addInstruction(
                                std::move(fmov_inst));
                        } else {
                            // 如果源是整数寄存器（从栈加载），需要特殊处理
                            storeOperandToReg(std::move(source_reg),
                                              std::move(dest_reg),
                                              first_riscv_bb);
                        }
                    } else {
                        // 其他情况（立即数等）
                        storeOperandToReg(std::move(source_reg),
                                          std::move(dest_reg), first_riscv_bb);
                    }
                } else {
                    // 整数/指针参数，使用原有逻辑
                    storeOperandToReg(std::move(source_reg),
                                      std::move(dest_reg), first_riscv_bb);
                }
            }
        }
    }

    for (const auto& bb : *func) {
        visit(bb, func_ptr);
        // func_ptr->mapBasicBlock(bb, new_riscv_bb);
    }

    // 后处理：处理所有PHI节点
    for (const auto& bb : *func) {
        for (const auto& inst : *bb) {
            if (inst->getOpcode() == midend::Opcode::PHI) {
                processDeferredPhiNode(inst, bb, func_ptr);
            }
        }
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
        // 跳过PHI节点，稍后处理
        if (inst->getOpcode() == midend::Opcode::PHI) {
            // 为PHI节点分配寄存器但不生成赋值指令
            auto phi_reg = codeGen_->allocateReg();
            codeGen_->mapValueToReg(inst, phi_reg->getRegNum(),
                                    phi_reg->isVirtual());
            continue;
        }
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
        // 根据指令类型保持正确的寄存器类型
        bool is_float_inst = inst->getType()->isFloatType();
        return std::make_unique<RegisterOperand>(
            foundReg.value()->getRegNum(), foundReg.value()->isVirtual(),
            is_float_inst ? RegisterType::Float : RegisterType::Integer);
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
        case midend::Opcode::LAnd:
        case midend::Opcode::LOr:
            // 处理逻辑与和逻辑或操作，需要正确实现短路求值
            return visitLogicalOp(inst, parent_bb);
            break;
        case midend::Opcode::FAdd:
        case midend::Opcode::FSub:
        case midend::Opcode::FMul:
        case midend::Opcode::FDiv:
        case midend::Opcode::FCmpOEQ:
        case midend::Opcode::FCmpONE:
        case midend::Opcode::FCmpOLT:
        case midend::Opcode::FCmpOLE:
        case midend::Opcode::FCmpOGT:
        case midend::Opcode::FCmpOGE:
            // 处理浮点算术和比较指令
            return visitFloatBinaryOp(inst, parent_bb);
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
        return std::make_unique<RegisterOperand>(
            register_operand->getRegNum(), register_operand->isVirtual(),
            register_operand->getRegisterType());
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

    // 处理立即数操作数
    if (operand->getType() == OperandType::Immediate) {
        auto* imm_operand = dynamic_cast<ImmediateOperand*>(operand.get());
        if (imm_operand == nullptr) {
            throw std::runtime_error("Invalid immediate operand type: " +
                                     operand->toString());
        }

        // 检查是否为浮点立即数
        if (imm_operand->isFloat()) {
            float float_value = imm_operand->getFloatValue();

            // 特殊处理浮点零值
            if (float_value == 0.0f) {
                // 分配浮点寄存器
                auto float_reg = codeGen_->allocateFloatReg();

                // 使用 fcvt.s.w 指令将整数零转换为浮点零
                auto fcvt_inst =
                    std::make_unique<Instruction>(Opcode::FCVT_S_W, parent_bb);
                fcvt_inst->addOperand(std::make_unique<RegisterOperand>(
                    float_reg->getRegNum(), float_reg->isVirtual(),
                    RegisterType::Float));  // rd (float)
                fcvt_inst->addOperand(std::make_unique<RegisterOperand>(
                    "zero"));  // rs1 (int zero)
                parent_bb->addInstruction(std::move(fcvt_inst));

                return std::make_unique<RegisterOperand>(float_reg->getRegNum(),
                                                         float_reg->isVirtual(),
                                                         RegisterType::Float);
            }

            // 分配浮点寄存器
            auto float_reg = codeGen_->allocateFloatReg();

            // 获取或创建浮点常量的标签
            auto* pool = codeGen_->getFloatConstantPool();
            std::string label = pool->getOrCreateFloatConstant(float_value);

            // 分配临时整数寄存器用于地址计算
            auto addr_reg = codeGen_->allocateReg();

            // 生成 lui 指令：加载高20位地址
            auto lui_inst =
                std::make_unique<Instruction>(Opcode::LUI, parent_bb);
            lui_inst->addOperand(std::make_unique<RegisterOperand>(
                addr_reg->getRegNum(), addr_reg->isVirtual()));
            lui_inst->addOperand(
                std::make_unique<LabelOperand>(label + "@hi"));  // %hi(label)
            parent_bb->addInstruction(std::move(lui_inst));

            // 生成 addi 指令：加载低12位地址
            auto addi_inst =
                std::make_unique<Instruction>(Opcode::ADDI, parent_bb);
            addi_inst->addOperand(std::make_unique<RegisterOperand>(
                addr_reg->getRegNum(), addr_reg->isVirtual()));
            addi_inst->addOperand(std::make_unique<RegisterOperand>(
                addr_reg->getRegNum(), addr_reg->isVirtual()));
            addi_inst->addOperand(
                std::make_unique<LabelOperand>(label + "@lo"));  // %lo(label)
            parent_bb->addInstruction(std::move(addi_inst));

            // 生成 flw 指令：从内存加载浮点数
            auto flw_inst =
                std::make_unique<Instruction>(Opcode::FLW, parent_bb);
            flw_inst->addOperand(std::make_unique<RegisterOperand>(
                float_reg->getRegNum(), float_reg->isVirtual(),
                RegisterType::Float));
            flw_inst->addOperand(std::make_unique<MemoryOperand>(
                std::make_unique<RegisterOperand>(addr_reg->getRegNum(),
                                                  addr_reg->isVirtual()),
                std::make_unique<ImmediateOperand>(0)));
            parent_bb->addInstruction(std::move(flw_inst));

            return std::make_unique<RegisterOperand>(float_reg->getRegNum(),
                                                     float_reg->isVirtual(),
                                                     RegisterType::Float);
        }

        // 处理整数立即数
        if (!imm_operand->isFloat() && imm_operand->getValue() == 0) {
            // 检查当前指令是否需要浮点零（例如浮点比较指令）
            // 注意：这是一个近似的判断，实际上应该根据指令类型更精确地判断

            // 如果立即数是整数 0，直接返回 zero 寄存器
            return std::make_unique<RegisterOperand>("zero");
        }

        // 生成一个新的寄存器，并将立即数加载到该寄存器中
        auto instruction = std::make_unique<Instruction>(Opcode::LI, parent_bb);
        auto new_reg = codeGen_->allocateReg();  // 分配一个新的寄存器
        auto reg_num = new_reg->getRegNum();
        auto is_virtual = new_reg->isVirtual();
        instruction->addOperand(std::move(new_reg));  // rd
        instruction->addOperand(std::make_unique<ImmediateOperand>(
            imm_operand->getValue()));  // imm
        parent_bb->addInstruction(std::move(instruction));

        return std::make_unique<RegisterOperand>(reg_num, is_virtual);
    }

    throw std::runtime_error("Unsupported operand type in immToReg: " +
                             operand->toString());
}

// 确保操作数在浮点寄存器中，特殊处理零值
std::unique_ptr<RegisterOperand> Visitor::ensureFloatReg(
    std::unique_ptr<MachineOperand> operand, BasicBlock* parent_bb) {
    // 如果已经是寄存器，检查是否已经是浮点寄存器类型
    if (operand->isReg()) {
        auto* reg_op = dynamic_cast<RegisterOperand*>(operand.get());
        // 如果已经是浮点寄存器，直接返回
        if (reg_op->isFloatRegister()) {
            return std::make_unique<RegisterOperand>(
                reg_op->getRegNum(), reg_op->isVirtual(), RegisterType::Float);
        }

        // 如果是整数寄存器，需要通过FMV_W_X转换到浮点寄存器
        // 分配新的浮点寄存器
        auto float_reg = codeGen_->allocateFloatReg();

        // 生成 FMV_W_X 指令：将整数寄存器的位模式移动到浮点寄存器
        auto fmv_inst =
            std::make_unique<Instruction>(Opcode::FMV_W_X, parent_bb);
        fmv_inst->addOperand(std::make_unique<RegisterOperand>(
            float_reg->getRegNum(), float_reg->isVirtual(),
            RegisterType::Float));  // rd (浮点目标寄存器)
        fmv_inst->addOperand(std::make_unique<RegisterOperand>(
            reg_op->getRegNum(), reg_op->isVirtual(),
            RegisterType::Integer));  // rs1 (整数源寄存器)
        parent_bb->addInstruction(std::move(fmv_inst));

        return std::make_unique<RegisterOperand>(float_reg->getRegNum(),
                                                 float_reg->isVirtual(),
                                                 RegisterType::Float);
    }

    // 如果是立即数，检查是否为零值
    if (operand->getType() == OperandType::Immediate) {
        auto* imm_operand = dynamic_cast<ImmediateOperand*>(operand.get());

        // 检查是否为零值（整数零或浮点零）
        bool is_zero = false;
        if (imm_operand->isFloat()) {
            is_zero = (imm_operand->getFloatValue() == 0.0f);
        } else {
            is_zero = (imm_operand->getValue() == 0);
        }

        if (is_zero) {
            // 分配浮点寄存器
            auto float_reg = codeGen_->allocateFloatReg();

            // 使用 fcvt.s.w 指令将整数零转换为浮点零
            auto fcvt_inst =
                std::make_unique<Instruction>(Opcode::FCVT_S_W, parent_bb);
            fcvt_inst->addOperand(std::make_unique<RegisterOperand>(
                float_reg->getRegNum(), float_reg->isVirtual(),
                RegisterType::Float));  // rd (float)
            fcvt_inst->addOperand(
                std::make_unique<RegisterOperand>("zero"));  // rs1 (int zero)
            parent_bb->addInstruction(std::move(fcvt_inst));

            return std::make_unique<RegisterOperand>(float_reg->getRegNum(),
                                                     float_reg->isVirtual(),
                                                     RegisterType::Float);
        }
    }

    // 对于其他情况，使用原有的 immToReg 逻辑但确保返回浮点寄存器类型
    auto reg = immToReg(std::move(operand), parent_bb);
    return std::make_unique<RegisterOperand>(reg->getRegNum(), reg->isVirtual(),
                                             RegisterType::Float);
}

std::unique_ptr<MachineOperand> Visitor::visitCastInst(
    const midend::Instruction* inst, BasicBlock* parent_bb) {
    auto* cast_inst = midend::dyn_cast<midend::CastInst>(inst);
    if (cast_inst == nullptr) {
        throw std::runtime_error("Not a cast instruction: " + inst->toString());
    }

    switch (cast_inst->getCastOpcode()) {
        case midend::CastInst::Trunc:
        case midend::CastInst::SIToFP:
        case midend::CastInst::FPToSI: {
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
                    // i1 -> i32
                    return immToReg(visit(cast_inst->getOperand(0), parent_bb),
                                    parent_bb);
                }

                if (src_type->isFloatType()) {
                    // f32 -> int (truncate towards zero)
                    auto new_reg = codeGen_->allocateReg();
                    auto* new_reg_ptr = new_reg.get();
                    auto src_operand =
                        visit(cast_inst->getOperand(0), parent_bb);

                    auto instruction = std::make_unique<Instruction>(
                        Opcode::FCVT_W_S, parent_bb);
                    instruction->addOperand(std::make_unique<RegisterOperand>(
                        new_reg_ptr->getRegNum(), new_reg_ptr->isVirtual(),
                        RegisterType::Integer));  // rd (integer)
                    instruction->addOperand(
                        std::move(src_operand));  // rs1 (float)
                    instruction->addOperand(std::make_unique<LabelOperand>(
                        "rtz"));  // rtz, 截断到零
                    parent_bb->addInstruction(std::move(instruction));
                    return std::make_unique<RegisterOperand>(
                        new_reg_ptr->getRegNum(), new_reg_ptr->isVirtual(),
                        RegisterType::Integer);
                }
            }

            if (dest_type->isFloatType()) {
                // int -> f32
                if (src_type->isIntegerType()) {
                    auto new_reg = codeGen_->allocateFloatReg();
                    auto* new_reg_ptr = new_reg.get();
                    auto src_operand =
                        visit(cast_inst->getOperand(0), parent_bb);

                    // 确保源操作数是整数寄存器类型
                    auto src_reg = immToReg(std::move(src_operand), parent_bb);

                    auto instruction = std::make_unique<Instruction>(
                        Opcode::FCVT_S_W, parent_bb);
                    instruction->addOperand(std::make_unique<RegisterOperand>(
                        new_reg_ptr->getRegNum(), new_reg_ptr->isVirtual(),
                        RegisterType::Float));  // rd (float)
                    instruction->addOperand(std::make_unique<RegisterOperand>(
                        src_reg->getRegNum(), src_reg->isVirtual(),
                        RegisterType::Integer));  // rs1 (integer)
                    parent_bb->addInstruction(std::move(instruction));
                    return std::make_unique<RegisterOperand>(
                        new_reg_ptr->getRegNum(), new_reg_ptr->isVirtual(),
                        RegisterType::Float);
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

    // 检查是否所有索引都为常量0，如果是则直接返回基地址
    bool all_indices_zero = true;
    for (unsigned i = 0; i < gep_inst->getNumIndices(); ++i) {
        auto* index_value = gep_inst->getIndex(i);
        if (auto* const_int =
                midend::dyn_cast<midend::ConstantInt>(index_value)) {
            if (const_int->getSignedValue() != 0) {
                all_indices_zero = false;
                break;
            }
        } else {
            all_indices_zero = false;
            break;
        }
    }

    if (all_indices_zero) {
        // 所有索引都为0，直接返回基地址
        codeGen_->mapValueToReg(inst, base_addr_reg->getRegNum(),
                                base_addr_reg->isVirtual());
        return std::make_unique<RegisterOperand>(base_addr_reg->getRegNum(),
                                                 base_addr_reg->isVirtual());
    }

    // 计算总偏移量，采用更直接的策略
    std::unique_ptr<RegisterOperand> total_offset_reg = nullptr;

    // 遍历所有索引，计算 index[i] * stride[i] 并累加
    for (unsigned i = 0; i < gep_inst->getNumIndices(); ++i) {
        auto* index_value = gep_inst->getIndex(i);
        auto stride = strides[i];

        // 检查索引是否为常量0，如果是则跳过
        if (auto* const_int =
                midend::dyn_cast<midend::ConstantInt>(index_value)) {
            if (const_int->getSignedValue() == 0) {
                continue;  // 跳过索引为0的情况，不会产生偏移
            }
        }

        if (stride == 0) {
            // stride为0，跳过
            continue;
        }

        auto index_operand = visit(index_value, parent_bb);
        auto index_reg = immToReg(std::move(index_operand), parent_bb);

        // 计算 index * stride
        std::unique_ptr<RegisterOperand> offset_reg;

        if (stride == 1) {
            // stride为1，直接使用索引作为偏移量
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
            li_stride_inst->addOperand(std::make_unique<ImmediateOperand>(
                static_cast<std::int64_t>(stride)));
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

        // 累加到总偏移量
        if (total_offset_reg == nullptr) {
            // 第一个非零偏移量，直接使用
            total_offset_reg = std::move(offset_reg);
        } else {
            // 累加：total_offset += offset
            auto new_total_offset_reg = codeGen_->allocateReg();
            auto add_inst =
                std::make_unique<Instruction>(Opcode::ADD, parent_bb);
            add_inst->addOperand(std::make_unique<RegisterOperand>(
                new_total_offset_reg->getRegNum(),
                new_total_offset_reg->isVirtual()));
            add_inst->addOperand(std::make_unique<RegisterOperand>(
                total_offset_reg->getRegNum(), total_offset_reg->isVirtual()));
            add_inst->addOperand(std::move(offset_reg));
            parent_bb->addInstruction(std::move(add_inst));
            total_offset_reg = std::move(new_total_offset_reg);
        }
    }

    // 如果没有任何偏移量，直接返回基地址
    if (total_offset_reg == nullptr) {
        codeGen_->mapValueToReg(inst, base_addr_reg->getRegNum(),
                                base_addr_reg->isVirtual());
        return std::make_unique<RegisterOperand>(base_addr_reg->getRegNum(),
                                                 base_addr_reg->isVirtual());
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

    // 处理参数传递 - 根据RISC-V ABI，整数和浮点参数使用独立的寄存器组
    size_t num_args = called_func->getNumArgs();

    // 独立维护整数和浮点参数的寄存器计数
    int int_reg_count = 0;
    int float_reg_count = 0;
    std::vector<std::pair<size_t, bool>>
        stack_args;  // 记录需要通过栈传递的参数 (索引, 是否浮点)

    // 第一遍：处理寄存器参数，记录栈参数
    for (size_t arg_i = 0; arg_i < num_args; ++arg_i) {
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

        bool is_float_arg = dest_arg->getType()->isFloatType();
        bool use_register = false;

        if (is_float_arg) {
            if (float_reg_count < 8) {
                use_register = true;
                float_reg_count++;
            }
        } else if (dest_arg->getType()->isIntegerType() ||
                   dest_arg->getType()->isPointerType()) {
            if (int_reg_count < 8) {
                use_register = true;
                int_reg_count++;
            }
        }

        if (use_register) {
            // 通过寄存器传递参数
            auto dest_arg_operand = funcArgToReg(dest_arg, parent_bb);

            auto* reg_operand =
                dynamic_cast<RegisterOperand*>(dest_arg_operand.get());
            if (reg_operand == nullptr) {
                throw std::runtime_error(
                    "Function argument must be a register operand");
            }

            auto dest_reg = std::make_unique<RegisterOperand>(
                reg_operand->getRegNum(), reg_operand->isVirtual(),
                is_float_arg ? RegisterType::Float : RegisterType::Integer);

            // 获取源操作数并移动到目标寄存器
            auto source_value = visit(source_operand, parent_bb);
            storeOperandToReg(std::move(source_value), std::move(dest_reg),
                              parent_bb);

        } else {
            // 记录需要通过栈传递的参数
            stack_args.emplace_back(arg_i, is_float_arg);
        }
    }

    // 第二遍：处理栈参数
    if (!stack_args.empty()) {
        int stack_offset = 0;
        for (const auto& [arg_i, is_float] : stack_args) {
            auto* source_operand = call_inst->getArgOperand(arg_i);
            auto source_value = visit(source_operand, parent_bb);

            std::unique_ptr<RegisterOperand> source_reg;
            if (source_value->getType() == OperandType::FrameIndex) {
                source_reg = immToReg(std::move(source_value), parent_bb);
            } else {
                source_reg = immToReg(std::move(source_value), parent_bb);
            }

            // 获取参数的真实类型来决定指令和大小
            auto* dest_arg = called_func->getArg(arg_i);
            bool is_pointer = dest_arg->getType()->isPointerType();

            // 根据参数类型选择存储指令和大小
            Opcode store_opcode;
            int arg_size;

            if (is_float) {
                store_opcode = Opcode::FSW;
                arg_size = 4;  // 单精度浮点，32位
            } else if (is_pointer) {
                store_opcode = Opcode::SD;
                arg_size = 8;  // 指针，64位
            } else {
                store_opcode = Opcode::SW;
                arg_size = 4;  // 整数，32位
            }

            generateMemoryInstruction(store_opcode, std::move(source_reg),
                                      std::make_unique<RegisterOperand>("sp"),
                                      stack_offset, parent_bb);
            stack_offset += arg_size;
        }
    }

    // 生成调用指令
    auto riscv_call_inst =
        std::make_unique<Instruction>(Opcode::CALL, parent_bb);
    riscv_call_inst->addOperand(
        std::make_unique<LabelOperand>(called_func->getName()));  // 函数名
    parent_bb->addInstruction(std::move(riscv_call_inst));

    // 如果被调用函数有返回值，则从相应的返回寄存器获取值并保存到新寄存器
    if (!called_func->getReturnType()->isVoidType()) {
        // 根据返回值类型选择正确的返回寄存器和目标寄存器类型
        bool is_float_return = called_func->getReturnType()->isFloatType();
        std::string return_reg = is_float_return ? "fa0" : "a0";

        auto new_reg = is_float_return ? codeGen_->allocateFloatReg()
                                       : codeGen_->allocateReg();
        auto dest_reg = std::make_unique<RegisterOperand>(
            new_reg->getRegNum(), new_reg->isVirtual(),
            is_float_return ? RegisterType::Float : RegisterType::Integer);

        // 创建具有正确类型的源寄存器操作数
        auto source_reg = std::make_unique<RegisterOperand>(
            ABI::getRegNumFromABIName(return_reg),
            false,  // fa0/a0 are physical registers
            is_float_return ? RegisterType::Float : RegisterType::Integer);

        storeOperandToReg(std::move(source_reg), std::move(dest_reg),
                          parent_bb);
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

    // 延迟处理PHI节点，存储必要信息供后续处理
    // 这里我们只返回寄存器，实际的赋值指令将在后续阶段插入
    for (unsigned i = 0; i < phi_inst->getNumIncomingValues(); ++i) {
        auto* incoming_value = phi_inst->getIncomingValue(i);
        auto* incoming_bb_midend = phi_inst->getIncomingBlock(i);
        auto* incoming_bb = parent_func->getBasicBlock(incoming_bb_midend);

        if (!incoming_bb) {
            throw std::runtime_error("Incoming block not found for PHI");
        }

        // 立即处理，但使用更可靠的插入策略
        // 访问输入值（在前驱块上下文）
        auto value_operand = visit(incoming_value, incoming_bb);

        // 创建赋值指令
        auto dest_reg = std::make_unique<RegisterOperand>(phi_reg->getRegNum(),
                                                          phi_reg->isVirtual());

        // 改进的插入策略：查找真正的跳转指令并在其前插入
        auto insert_pos = incoming_bb->end();

        // 向后查找最后一条非PHI相关的指令（应该是跳转指令）
        bool found_jump = false;
        auto current_pos = incoming_bb->end();

        while (current_pos != incoming_bb->begin()) {
            --current_pos;
            auto* current_inst = current_pos->get();

            // 检查是否是跳转指令
            if (current_inst->isJumpInstr() || current_inst->isBranch() ||
                current_inst->getOpcode() == Opcode::BNEZ ||
                current_inst->getOpcode() == Opcode::BEQZ ||
                current_inst->getOpcode() == Opcode::BEQ ||
                current_inst->getOpcode() == Opcode::BNE ||
                current_inst->getOpcode() == Opcode::J) {
                insert_pos = current_pos;
                found_jump = true;
                break;
            }
        }

        // 如果没找到跳转指令，插入到末尾
        if (!found_jump) {
            insert_pos = incoming_bb->end();
        }

        storeOperandToReg(std::move(value_operand), std::move(dest_reg),
                          incoming_bb, insert_pos);
    }

    // PHI块本身不生成任何指令，只返回寄存器
    return std::make_unique<RegisterOperand>(phi_reg->getRegNum(),
                                             phi_reg->isVirtual());
}

// 延迟处理PHI节点的方法
void Visitor::processDeferredPhiNode(const midend::Instruction* inst,
                                     const midend::BasicBlock* bb_midend,
                                     Function* parent_func) {
    const auto* phi_inst = dynamic_cast<const midend::PHINode*>(inst);
    if (!phi_inst) {
        throw std::runtime_error("Not a PHI instruction");
    }

    // 获取已分配的PHI寄存器
    auto foundReg = findRegForValue(inst);
    if (!foundReg.has_value()) {
        throw std::runtime_error("PHI register not found");
    }
    auto phi_reg_num = foundReg.value()->getRegNum();
    bool phi_is_virtual = foundReg.value()->isVirtual();

    // 对每个前驱块，在其跳转指令前插入赋值
    for (unsigned i = 0; i < phi_inst->getNumIncomingValues(); ++i) {
        auto* incoming_value = phi_inst->getIncomingValue(i);
        auto* incoming_bb_midend = phi_inst->getIncomingBlock(i);
        auto* incoming_bb = parent_func->getBasicBlock(incoming_bb_midend);

        if (!incoming_bb) {
            throw std::runtime_error("Incoming block not found for PHI");
        }

        // 访问输入值
        auto value_operand = visit(incoming_value, incoming_bb);

        // 创建赋值指令 - 但要注意避免干扰条件判断
        auto dest_reg =
            std::make_unique<RegisterOperand>(phi_reg_num, phi_is_virtual);

        // 检查是否为常量值，如果是，我们需要特殊处理以避免干扰条件判断
        bool is_constant_phi = false;
        if (auto* imm_operand =
                dynamic_cast<ImmediateOperand*>(value_operand.get())) {
            is_constant_phi = true;
        }

        // 查找跳转指令并在其前插入
        auto insert_pos = incoming_bb->end();
        bool found_jump = false;

        // 更精确的跳转指令查找策略：
        // 1. 优先查找条件跳转指令
        // 2. 如果没有条件跳转，则查找无条件跳转
        auto conditional_jump_pos = incoming_bb->end();
        auto unconditional_jump_pos = incoming_bb->end();
        bool found_conditional = false;
        bool found_unconditional = false;

        for (auto it = incoming_bb->begin(); it != incoming_bb->end(); ++it) {
            auto* current_inst = it->get();

            // 检查条件跳转指令
            if (current_inst->getOpcode() == Opcode::BNEZ ||
                current_inst->getOpcode() == Opcode::BEQZ ||
                current_inst->getOpcode() == Opcode::BEQ ||
                current_inst->getOpcode() == Opcode::BNE ||
                current_inst->getOpcode() == Opcode::BLT ||
                current_inst->getOpcode() == Opcode::BGE ||
                current_inst->getOpcode() == Opcode::BLTU ||
                current_inst->getOpcode() == Opcode::BGEU) {
                conditional_jump_pos = it;
                found_conditional = true;
                break;  // 找到第一个条件跳转就停止
            }
            // 检查无条件跳转指令
            else if (current_inst->getOpcode() == Opcode::J ||
                     current_inst->getOpcode() == Opcode::JAL ||
                     current_inst->getOpcode() == Opcode::JALR ||
                     current_inst->isJumpInstr()) {
                if (!found_unconditional) {
                    unconditional_jump_pos = it;
                    found_unconditional = true;
                }
            }
        }

        // 选择插入位置：优先条件跳转，其次无条件跳转
        if (found_conditional) {
            insert_pos = conditional_jump_pos;
            found_jump = true;
        } else if (found_unconditional) {
            insert_pos = unconditional_jump_pos;
            found_jump = true;
        }

        // 如果没找到跳转指令，插入到末尾
        if (!found_jump) {
            insert_pos = incoming_bb->end();
        }

        // 临时修复：对于常量PHI值，检查是否会干扰条件判断
        bool should_skip_phi = false;
        if (is_constant_phi && found_conditional) {
            // 更强的干扰检测：检查条件跳转指令使用的条件寄存器
            auto* cond_inst = insert_pos->get();
            if (cond_inst && (cond_inst->getOpcode() == Opcode::BNEZ ||
                              cond_inst->getOpcode() == Opcode::BEQZ)) {
                if (!cond_inst->getOperands().empty()) {
                    auto* cond_reg_op = cond_inst->getOperands()[0].get();
                    if (auto* cond_reg =
                            dynamic_cast<RegisterOperand*>(cond_reg_op)) {
                        // 如果PHI目标寄存器与条件跳转使用的寄存器相同，跳过PHI赋值
                        if (cond_reg->getRegNum() == phi_reg_num) {
                            should_skip_phi = true;
                        }
                    }
                }
            }

            // 额外检查：检查条件计算指令
            if (!should_skip_phi) {
                auto check_pos = insert_pos;
                int check_count = 0;
                while (check_pos != incoming_bb->begin() && check_count < 5) {
                    --check_pos;
                    auto* inst = check_pos->get();
                    if (inst->getOpcode() == Opcode::SEQZ ||
                        inst->getOpcode() == Opcode::SNEZ ||
                        inst->getOpcode() == Opcode::SLT ||
                        inst->getOpcode() == Opcode::SGT ||
                        inst->getOpcode() == Opcode::XOR) {
                        if (!inst->getOperands().empty()) {
                            auto* target_op = inst->getOperands()[0].get();
                            if (auto* target_reg =
                                    dynamic_cast<RegisterOperand*>(target_op)) {
                                if (target_reg->getRegNum() == phi_reg_num) {
                                    should_skip_phi = true;
                                    break;
                                }
                            }
                        }
                    }
                    check_count++;
                }
            }
        }

        // 简单启发式规则：对于可能干扰条件判断的PHI，使用更安全的策略
        if (should_skip_phi || (is_constant_phi && found_conditional)) {
            // 如果检测到潜在干扰，暂时跳过这个PHI赋值
            // 这是一个临时解决方案，确保基本逻辑正确
        } else {
            storeOperandToReg(std::move(value_operand), std::move(dest_reg),
                              incoming_bb, insert_pos);
        }
    }
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

    // 获取存储的值，根据类型确保正确的寄存器类型
    auto raw_value_operand = visit(store_inst->getValueOperand(), parent_bb);
    auto value_operand =
        store_inst->getValueOperand()->getType()->isFloatType()
            ? ensureFloatReg(std::move(raw_value_operand), parent_bb)
            : immToReg(std::move(raw_value_operand), parent_bb);

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

        // 根据原始IR中存储值的类型选择存储指令
        bool is_float_store =
            store_inst->getValueOperand()->getType()->isFloatType();
        Opcode store_opcode = is_float_store ? Opcode::FSW : Opcode::SW;
        auto store_inst_new =
            std::make_unique<Instruction>(store_opcode, parent_bb);
        store_inst_new->addOperand(
            std::move(value_operand));  // source register
        store_inst_new->addOperand(std::make_unique<MemoryOperand>(
            std::make_unique<RegisterOperand>(frame_addr_reg->getRegNum(),
                                              frame_addr_reg->isVirtual()),
            std::make_unique<ImmediateOperand>(0)));  // memory address
        parent_bb->addInstruction(std::move(store_inst_new));

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

        // 根据原始IR中存储值的类型选择存储指令
        bool is_float_store =
            store_inst->getValueOperand()->getType()->isFloatType();

        Opcode store_opcode = is_float_store ? Opcode::FSW : Opcode::SW;
        auto store_inst_new =
            std::make_unique<Instruction>(store_opcode, parent_bb);
        store_inst_new->addOperand(
            std::move(value_operand));  // source register
        store_inst_new->addOperand(std::make_unique<MemoryOperand>(
            std::make_unique<RegisterOperand>(address_reg->getRegNum(),
                                              address_reg->isVirtual()),
            std::make_unique<ImmediateOperand>(0)));  // memory address
        parent_bb->addInstruction(std::move(store_inst_new));

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

        // 根据原始IR中存储值的类型选择存储指令
        bool is_float_store =
            store_inst->getValueOperand()->getType()->isFloatType();

        Opcode store_opcode = is_float_store ? Opcode::FSW : Opcode::SW;
        auto store_inst_new =
            std::make_unique<Instruction>(store_opcode, parent_bb);
        store_inst_new->addOperand(
            std::move(value_operand));  // source register
        store_inst_new->addOperand(std::make_unique<MemoryOperand>(
            std::make_unique<RegisterOperand>(global_addr_reg->getRegNum(),
                                              global_addr_reg->isVirtual()),
            std::make_unique<ImmediateOperand>(0)));  // memory address
        parent_bb->addInstruction(std::move(store_inst_new));

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

        // 根据load指令的返回类型选择寄存器和加载指令
        bool is_float_load = load_inst->getType()->isFloatType();
        auto new_reg = is_float_load ? codeGen_->allocateFloatReg()
                                     : codeGen_->allocateReg();
        Opcode load_opcode = is_float_load ? Opcode::FLW : Opcode::LW;

        // 使用正确的加载指令
        generateMemoryInstruction(
            load_opcode,
            std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                is_float_load ? RegisterType::Float : RegisterType::Integer),
            std::make_unique<RegisterOperand>(frame_addr_reg->getRegNum(),
                                              frame_addr_reg->isVirtual()),
            0, parent_bb);

        // 建立load指令结果值到寄存器的映射
        codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                new_reg->isVirtual());

        return std::make_unique<RegisterOperand>(
            new_reg->getRegNum(), new_reg->isVirtual(),
            is_float_load ? RegisterType::Float : RegisterType::Integer);

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

        // 根据load指令的返回类型选择寄存器和加载指令
        bool is_float_load = load_inst->getType()->isFloatType();
        auto new_reg = is_float_load ? codeGen_->allocateFloatReg()
                                     : codeGen_->allocateReg();
        Opcode load_opcode = is_float_load ? Opcode::FLW : Opcode::LW;

        // 使用正确的加载指令
        generateMemoryInstruction(
            load_opcode,
            std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                is_float_load ? RegisterType::Float : RegisterType::Integer),
            std::make_unique<RegisterOperand>(address_reg->getRegNum(),
                                              address_reg->isVirtual()),
            0, parent_bb);

        // 建立load指令结果值到寄存器的映射
        codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                new_reg->isVirtual());

        return std::make_unique<RegisterOperand>(
            new_reg->getRegNum(), new_reg->isVirtual(),
            is_float_load ? RegisterType::Float : RegisterType::Integer);

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

        // 根据load指令的返回类型选择寄存器和加载指令
        bool is_float_load = load_inst->getType()->isFloatType();
        auto new_reg = is_float_load ? codeGen_->allocateFloatReg()
                                     : codeGen_->allocateReg();
        Opcode load_opcode = is_float_load ? Opcode::FLW : Opcode::LW;

        // 使用正确的加载指令
        generateMemoryInstruction(
            load_opcode,
            std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                is_float_load ? RegisterType::Float : RegisterType::Integer),
            std::make_unique<RegisterOperand>(global_addr_reg->getRegNum(),
                                              global_addr_reg->isVirtual()),
            0, parent_bb);

        // 建立load指令结果值到寄存器的映射
        codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                new_reg->isVirtual());

        return std::make_unique<RegisterOperand>(
            new_reg->getRegNum(), new_reg->isVirtual(),
            is_float_load ? RegisterType::Float : RegisterType::Integer);

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

        // 检查是否为浮点操作
        bool is_float_op = inst->getType()->isFloatType();

        // If both are immediates, do constant folding
        if (operand->getType() == OperandType::Immediate) {
            auto* imm_operand = dynamic_cast<ImmediateOperand*>(operand.get());
            if (is_float_op) {
                return std::make_unique<ImmediateOperand>(
                    -imm_operand->getFloatValue());
            } else {
                return std::make_unique<ImmediateOperand>(
                    -imm_operand->getValue());
            }
        }

        // Convert operand to register if needed
        auto operand_reg = immToReg(std::move(operand), parent_bb);

        if (is_float_op) {
            // 浮点数取负：使用 fneg.s 指令
            auto new_reg = codeGen_->allocateFloatReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::FNEG_S, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Float));                        // rd
            instruction->addOperand(std::move(operand_reg));  // rs1
            parent_bb->addInstruction(std::move(instruction));

            codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                    new_reg->isVirtual());
            return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                     new_reg->isVirtual(),
                                                     RegisterType::Float);
        } else {
            // 整数取负：使用 sub 指令: 0 - operand
            auto new_reg = codeGen_->allocateReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::SUB, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual()));  // rd
            instruction->addOperand(std::make_unique<RegisterOperand>(
                "zero"));  // rs1 (zero register)
            instruction->addOperand(std::move(operand_reg));  // rs2
            parent_bb->addInstruction(std::move(instruction));

            codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                    new_reg->isVirtual());
            return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                     new_reg->isVirtual());
        }
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

    // 检查是否为浮点操作，以便正确创建操作数
    bool is_float_op = inst->getType()->isFloatType();

    std::unique_ptr<MachineOperand> lhs;
    {
        const auto foundReg = findRegForValue(inst->getOperand(0));
        if (foundReg.has_value()) {
            // 根据操作类型选择正确的寄存器类型
            lhs = std::make_unique<RegisterOperand>(
                foundReg.value()->getRegNum(), foundReg.value()->isVirtual(),
                is_float_op ? RegisterType::Float : RegisterType::Integer);
        } else {
            lhs = visit(inst->getOperand(0), parent_bb);
        }
    }
    std::unique_ptr<MachineOperand> rhs;
    {
        const auto foundReg = findRegForValue(inst->getOperand(1));
        if (foundReg.has_value()) {
            // 根据操作类型选择正确的寄存器类型
            rhs = std::make_unique<RegisterOperand>(
                foundReg.value()->getRegNum(), foundReg.value()->isVirtual(),
                is_float_op ? RegisterType::Float : RegisterType::Integer);
        } else {
            rhs = visit(inst->getOperand(1), parent_bb);
        }
    }

    // Only allocate a new register if needed (not for immediate result)
    std::unique_ptr<RegisterOperand> new_reg;

    // TODO(rikka): 关于 0 和 1 的判断优化等，后期写一个 Pass 来优化
    switch (inst->getOpcode()) {
        case midend::Opcode::Add: {
            // 检查是否为浮点操作
            // bool is_float_op = inst->getType()->isFloatType(); // 已移到上面

            // 先判断是否有立即数
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());

                if (is_float_op) {
                    return std::make_unique<ImmediateOperand>(
                        lhs_imm->getFloatValue() + rhs_imm->getFloatValue());
                } else {
                    return std::make_unique<ImmediateOperand>(
                        lhs_imm->getValue() + rhs_imm->getValue());
                }
            }

            if (is_float_op) {
                // 浮点加法：使用 fadd 指令
                new_reg = codeGen_->allocateFloatReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::FADD_S, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    RegisterType::Float));                // rd
                instruction->addOperand(std::move(lhs));  // rs1
                instruction->addOperand(std::move(rhs));  // rs2
                parent_bb->addInstruction(std::move(instruction));
            } else {
                // 整数加法：原有逻辑
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
            }
            break;
        }

        case midend::Opcode::Sub: {
            // 检查是否为浮点操作
            bool is_float_op = inst->getType()->isFloatType();

            // 先判断是否是两个立即数
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                if (is_float_op) {
                    return std::make_unique<ImmediateOperand>(
                        lhs_imm->getFloatValue() - rhs_imm->getFloatValue());
                } else {
                    return std::make_unique<ImmediateOperand>(
                        lhs_imm->getValue() - rhs_imm->getValue());
                }
            }

            if (is_float_op) {
                // 浮点减法：使用 fsub 指令
                new_reg = codeGen_->allocateFloatReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::FSUB_S, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    RegisterType::Float));                // rd
                instruction->addOperand(std::move(lhs));  // rs1
                instruction->addOperand(std::move(rhs));  // rs2
                parent_bb->addInstruction(std::move(instruction));
            } else {
                // 整数减法：原有逻辑
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
            }
            break;
        }

        case midend::Opcode::Mul: {
            // 检查是否为浮点乘法
            if (is_float_op) {
                // 重定向到浮点乘法函数
                return visitFloatBinaryOp(inst, parent_bb);
            }

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
            // 检查是否为浮点除法
            if (is_float_op) {
                // 重定向到浮点除法函数
                return visitFloatBinaryOp(inst, parent_bb);
            }

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
            // 检查是否为浮点比较
            bool is_float_cmp = inst->getOperand(0)->getType()->isFloatType();
            if (is_float_cmp) {
                // 浮点大于比较，重定向到浮点比较函数
                return visitFloatBinaryOp(inst, parent_bb);
            }

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
            // 检查是否为浮点比较
            bool is_float_cmp = inst->getOperand(0)->getType()->isFloatType();
            if (is_float_cmp) {
                // 浮点小于比较，重定向到浮点比较函数
                return visitFloatBinaryOp(inst, parent_bb);
            }

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
        // 根据指令类型返回正确的寄存器类型
        bool is_float_result = inst->getType()->isFloatType();

        // 确保值被正确映射以避免重复计算
        codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                new_reg->isVirtual());

        // 根据指令类型返回正确的寄存器类型
        return std::make_unique<RegisterOperand>(
            new_reg->getRegNum(), new_reg->isVirtual(),
            is_float_result ? RegisterType::Float : RegisterType::Integer);
    }  // Should only happen if we returned early (immediate case)
    throw std::runtime_error("No register allocated for binary op result");
}

std::unique_ptr<MachineOperand> Visitor::visitFloatBinaryOp(
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
                foundReg.value()->getRegNum(), foundReg.value()->isVirtual(),
                RegisterType::Float);
        } else {
            lhs = visit(inst->getOperand(0), parent_bb);
        }
    }
    std::unique_ptr<MachineOperand> rhs;
    {
        const auto foundReg = findRegForValue(inst->getOperand(1));
        if (foundReg.has_value()) {
            rhs = std::make_unique<RegisterOperand>(
                foundReg.value()->getRegNum(), foundReg.value()->isVirtual(),
                RegisterType::Float);
        } else {
            rhs = visit(inst->getOperand(1), parent_bb);
        }
    }

    // Only allocate a new register if needed (not for immediate result)
    std::unique_ptr<RegisterOperand> new_reg;

    // TODO(rikka): 关于 0 和 1 的判断优化等，后期写一个 Pass 来优化
    switch (inst->getOpcode()) {
        case midend::Opcode::FAdd: {
            // 先判断是否有立即数
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    lhs_imm->getFloatValue() + rhs_imm->getFloatValue());
            }

            // 使用 fadd 指令
            new_reg = codeGen_->allocateFloatReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::FADD_S, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Float));                // rd
            instruction->addOperand(std::move(lhs));  // rs1
            instruction->addOperand(std::move(rhs));  // rs2
            parent_bb->addInstruction(std::move(instruction));

            break;
        }

        case midend::Opcode::FSub: {
            // 先判断是否是两个立即数
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    lhs_imm->getFloatValue() - rhs_imm->getFloatValue());
            }

            // 使用 fsub 指令
            new_reg = codeGen_->allocateFloatReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::FSUB_S, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Float));                // rd
            instruction->addOperand(std::move(lhs));  // rs1
            instruction->addOperand(std::move(rhs));  // rs2
            parent_bb->addInstruction(std::move(instruction));

            break;
        }

        case midend::Opcode::FMul: {
            // 先判断是否是两个立即数
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    lhs_imm->getFloatValue() * rhs_imm->getFloatValue());
            }

            // 使用 fmul 指令
            new_reg = codeGen_->allocateFloatReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::FMUL_S, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Float));                // rd
            instruction->addOperand(std::move(lhs));  // rs1
            instruction->addOperand(std::move(rhs));  // rs2
            parent_bb->addInstruction(std::move(instruction));

            break;
        }

        case midend::Opcode::FDiv: {
            // 先判断是否是两个立即数
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                if (rhs_imm->getFloatValue() == 0.0f) {
                    throw std::runtime_error("Division by zero");
                }
                return std::make_unique<ImmediateOperand>(
                    lhs_imm->getFloatValue() / rhs_imm->getFloatValue());
            }

            // 使用 fdiv 指令
            new_reg = codeGen_->allocateFloatReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::FDIV_S, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Float));                // rd
            instruction->addOperand(std::move(lhs));  // rs1
            instruction->addOperand(std::move(rhs));  // rs2
            parent_bb->addInstruction(std::move(instruction));

            break;
        }

        case midend::Opcode::FCmpOEQ: {
            // 浮点相等比较 (ordered equal)
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    (lhs_imm->getFloatValue() == rhs_imm->getFloatValue()) ? 1
                                                                           : 0);
            }

            // 确保操作数都在浮点寄存器中
            auto lhs_reg = ensureFloatReg(std::move(lhs), parent_bb);
            auto rhs_reg = ensureFloatReg(std::move(rhs), parent_bb);

            // 使用 feq.s 指令
            new_reg = codeGen_->allocateReg();  // 比较结果是整数
            auto instruction =
                std::make_unique<Instruction>(Opcode::FEQ_S, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Integer));                  // rd
            instruction->addOperand(std::move(lhs_reg));  // rs1
            instruction->addOperand(std::move(rhs_reg));  // rs2
            parent_bb->addInstruction(std::move(instruction));

            break;
        }

        case midend::Opcode::FCmpONE: {
            // 浮点不等比较 (ordered not equal)
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    (lhs_imm->getFloatValue() != rhs_imm->getFloatValue()) ? 1
                                                                           : 0);
            }

            // 确保操作数都在浮点寄存器中
            auto lhs_reg = ensureFloatReg(std::move(lhs), parent_bb);
            auto rhs_reg = ensureFloatReg(std::move(rhs), parent_bb);

            // 先使用 feq.s 指令得到相等结果，然后取反
            new_reg = codeGen_->allocateReg();
            auto feq_inst =
                std::make_unique<Instruction>(Opcode::FEQ_S, parent_bb);
            feq_inst->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Integer));               // rd
            feq_inst->addOperand(std::move(lhs_reg));  // rs1
            feq_inst->addOperand(std::move(rhs_reg));  // rs2
            parent_bb->addInstruction(std::move(feq_inst));

            // 使用 seqz 指令取反（如果相等结果为0，则设置为1；否则设置为0）
            auto result_reg = codeGen_->allocateReg();
            auto seqz_inst =
                std::make_unique<Instruction>(Opcode::SEQZ, parent_bb);
            seqz_inst->addOperand(std::make_unique<RegisterOperand>(
                result_reg->getRegNum(), result_reg->isVirtual(),
                RegisterType::Integer));  // rd
            seqz_inst->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Integer));  // rs1
            parent_bb->addInstruction(std::move(seqz_inst));

            // 更新 new_reg 为最终结果寄存器
            new_reg = std::move(result_reg);
            break;
        }

        case midend::Opcode::FCmpOLT: {
            // 浮点小于比较 (ordered less than)
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    (lhs_imm->getFloatValue() < rhs_imm->getFloatValue()) ? 1
                                                                          : 0);
            }

            // 确保操作数都在浮点寄存器中
            auto lhs_reg = ensureFloatReg(std::move(lhs), parent_bb);
            auto rhs_reg = ensureFloatReg(std::move(rhs), parent_bb);

            // 使用 flt.s 指令
            new_reg = codeGen_->allocateReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::FLT_S, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Integer));                  // rd
            instruction->addOperand(std::move(lhs_reg));  // rs1
            instruction->addOperand(std::move(rhs_reg));  // rs2
            parent_bb->addInstruction(std::move(instruction));

            break;
        }

        case midend::Opcode::FCmpOLE: {
            // 浮点小于等于比较 (ordered less than or equal)
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    (lhs_imm->getFloatValue() <= rhs_imm->getFloatValue()) ? 1
                                                                           : 0);
            }

            // 确保操作数都在浮点寄存器中
            auto lhs_reg = ensureFloatReg(std::move(lhs), parent_bb);
            auto rhs_reg = ensureFloatReg(std::move(rhs), parent_bb);

            // 使用 fle.s 指令
            new_reg = codeGen_->allocateReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::FLE_S, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Integer));                  // rd
            instruction->addOperand(std::move(lhs_reg));  // rs1
            instruction->addOperand(std::move(rhs_reg));  // rs2
            parent_bb->addInstruction(std::move(instruction));

            break;
        }

        case midend::Opcode::FCmpOGT: {
            // 浮点大于比较 (ordered greater than)
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    (lhs_imm->getFloatValue() > rhs_imm->getFloatValue()) ? 1
                                                                          : 0);
            }

            // 确保操作数都在浮点寄存器中
            auto lhs_reg = ensureFloatReg(std::move(lhs), parent_bb);
            auto rhs_reg = ensureFloatReg(std::move(rhs), parent_bb);

            // RISC-V 没有直接的 fgt.s 指令，使用 flt.s 但交换操作数
            new_reg = codeGen_->allocateReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::FLT_S, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Integer));  // rd
            instruction->addOperand(
                std::move(rhs_reg));  // rs1 (交换：rhs < lhs)
            instruction->addOperand(std::move(lhs_reg));  // rs2
            parent_bb->addInstruction(std::move(instruction));

            break;
        }

        case midend::Opcode::FCmpOGE: {
            // 浮点大于等于比较 (ordered greater than or equal)
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    (lhs_imm->getFloatValue() >= rhs_imm->getFloatValue()) ? 1
                                                                           : 0);
            }

            // 确保操作数都在浮点寄存器中
            auto lhs_reg = ensureFloatReg(std::move(lhs), parent_bb);
            auto rhs_reg = ensureFloatReg(std::move(rhs), parent_bb);

            // RISC-V 没有直接的 fge.s 指令，使用 fle.s 但交换操作数
            new_reg = codeGen_->allocateReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::FLE_S, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Integer));  // rd
            instruction->addOperand(
                std::move(rhs_reg));  // rs1 (交换：rhs <= lhs)
            instruction->addOperand(std::move(lhs_reg));  // rs2
            parent_bb->addInstruction(std::move(instruction));

            break;
        }

        case midend::Opcode::ICmpSGT: {
            // 处理来自整数比较重定向的浮点大于比较
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    (lhs_imm->getFloatValue() > rhs_imm->getFloatValue()) ? 1
                                                                          : 0);
            }

            // 确保操作数都在浮点寄存器中
            auto lhs_reg = immToReg(std::move(lhs), parent_bb);
            auto rhs_reg = immToReg(std::move(rhs), parent_bb);

            // 使用 flt.s 但交换操作数来实现大于比较
            new_reg = codeGen_->allocateReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::FLT_S, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Integer));  // rd
            instruction->addOperand(
                std::move(rhs_reg));  // rs1 (交换：rhs < lhs)
            instruction->addOperand(std::move(lhs_reg));  // rs2
            parent_bb->addInstruction(std::move(instruction));

            break;
        }

        case midend::Opcode::ICmpSLT: {
            // 处理来自整数比较重定向的浮点小于比较
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                return std::make_unique<ImmediateOperand>(
                    (lhs_imm->getFloatValue() < rhs_imm->getFloatValue()) ? 1
                                                                          : 0);
            }

            // 确保操作数都在浮点寄存器中
            auto lhs_reg = ensureFloatReg(std::move(lhs), parent_bb);
            auto rhs_reg = ensureFloatReg(std::move(rhs), parent_bb);

            // 使用 flt.s 指令实现小于比较
            new_reg = codeGen_->allocateReg();
            auto instruction =
                std::make_unique<Instruction>(Opcode::FLT_S, parent_bb);
            instruction->addOperand(std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                RegisterType::Integer));                  // rd
            instruction->addOperand(std::move(lhs_reg));  // rs1
            instruction->addOperand(std::move(rhs_reg));  // rs2
            parent_bb->addInstruction(std::move(instruction));

            break;
        }

        // 兼容性处理：将一般的 Mul 操作重定向到 FMul
        case midend::Opcode::Mul: {
            // 如果是浮点类型，处理为 FMul
            if (inst->getType()->isFloatType()) {
                // 先判断是否是两个立即数
                if ((lhs->getType() == OperandType::Immediate) &&
                    (rhs->getType() == OperandType::Immediate)) {
                    auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                    auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                    return std::make_unique<ImmediateOperand>(
                        lhs_imm->getFloatValue() * rhs_imm->getFloatValue());
                }

                // 确保操作数都在浮点寄存器中
                auto lhs_reg = immToReg(std::move(lhs), parent_bb);
                auto rhs_reg = immToReg(std::move(rhs), parent_bb);

                // 使用 fmul 指令
                new_reg = codeGen_->allocateFloatReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::FMUL_S, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    RegisterType::Float));                    // rd
                instruction->addOperand(std::move(lhs_reg));  // rs1
                instruction->addOperand(std::move(rhs_reg));  // rs2
                parent_bb->addInstruction(std::move(instruction));

                break;
            }
            // 如果不是浮点类型，抛出错误（不应该到这里）
            throw std::runtime_error(
                "Integer Mul operation in float context: " + inst->toString());
        }

        // 兼容性处理：将一般的 Div 操作重定向到 FDiv
        case midend::Opcode::Div: {
            // 如果是浮点类型，处理为 FDiv
            if (inst->getType()->isFloatType()) {
                // 先判断是否是两个立即数
                if ((lhs->getType() == OperandType::Immediate) &&
                    (rhs->getType() == OperandType::Immediate)) {
                    auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                    auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                    if (rhs_imm->getFloatValue() == 0.0f) {
                        throw std::runtime_error("Division by zero");
                    }
                    return std::make_unique<ImmediateOperand>(
                        lhs_imm->getFloatValue() / rhs_imm->getFloatValue());
                }

                // 确保操作数都在浮点寄存器中
                auto lhs_reg = immToReg(std::move(lhs), parent_bb);
                auto rhs_reg = immToReg(std::move(rhs), parent_bb);

                // 使用 fdiv 指令
                new_reg = codeGen_->allocateFloatReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::FDIV_S, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    RegisterType::Float));                    // rd
                instruction->addOperand(std::move(lhs_reg));  // rs1
                instruction->addOperand(std::move(rhs_reg));  // rs2
                parent_bb->addInstruction(std::move(instruction));

                break;
            }
            // 如果不是浮点类型，抛出错误（不应该到这里）
            throw std::runtime_error(
                "Integer Div operation in float context: " + inst->toString());
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
        // 根据指令类型返回正确的寄存器类型
        bool is_float_result = inst->getType()->isFloatType();
        return std::make_unique<RegisterOperand>(
            new_reg->getRegNum(), new_reg->isVirtual(),
            is_float_result ? RegisterType::Float : RegisterType::Integer);
    }  // Should only happen if we returned early (immediate case)
    throw std::runtime_error("No register allocated for binary op result");
}

// 处理逻辑操作（LAnd和LOr），实现正确的短路求值
std::unique_ptr<MachineOperand> Visitor::visitLogicalOp(
    const midend::Instruction* inst, BasicBlock* parent_bb) {
    if (inst->getOpcode() != midend::Opcode::LAnd &&
        inst->getOpcode() != midend::Opcode::LOr) {
        throw std::runtime_error("Not a logical operation instruction: " +
                                 inst->toString());
    }

    if (inst->getNumOperands() != 2) {
        throw std::runtime_error(
            "Logical operation must have two operands, got " +
            std::to_string(inst->getNumOperands()));
    }

    // 为最终结果分配寄存器
    auto result_reg = codeGen_->allocateReg();

    // 获取左操作数并转换为布尔值
    auto lhs = visit(inst->getOperand(0), parent_bb);
    auto lhs_reg = immToReg(std::move(lhs), parent_bb);

    // 将左操作数转换为布尔值（0或1）
    auto lhs_bool_reg = codeGen_->allocateReg();
    auto lhs_snez_inst = std::make_unique<Instruction>(Opcode::SNEZ, parent_bb);
    lhs_snez_inst->addOperand(std::make_unique<RegisterOperand>(
        lhs_bool_reg->getRegNum(), lhs_bool_reg->isVirtual()));
    lhs_snez_inst->addOperand(std::move(lhs_reg));
    parent_bb->addInstruction(std::move(lhs_snez_inst));

    // 获取右操作数并转换为布尔值
    auto rhs = visit(inst->getOperand(1), parent_bb);
    auto rhs_reg = immToReg(std::move(rhs), parent_bb);

    auto rhs_bool_reg = codeGen_->allocateReg();
    auto rhs_snez_inst = std::make_unique<Instruction>(Opcode::SNEZ, parent_bb);
    rhs_snez_inst->addOperand(std::make_unique<RegisterOperand>(
        rhs_bool_reg->getRegNum(), rhs_bool_reg->isVirtual()));
    rhs_snez_inst->addOperand(std::move(rhs_reg));
    parent_bb->addInstruction(std::move(rhs_snez_inst));

    if (inst->getOpcode() == midend::Opcode::LAnd) {
        // 逻辑与：result = lhs_bool && rhs_bool
        // 使用按位与实现：两个操作数都是0或1，所以按位与等于逻辑与
        auto and_inst = std::make_unique<Instruction>(Opcode::AND, parent_bb);
        and_inst->addOperand(std::make_unique<RegisterOperand>(
            result_reg->getRegNum(), result_reg->isVirtual()));
        and_inst->addOperand(std::make_unique<RegisterOperand>(
            lhs_bool_reg->getRegNum(), lhs_bool_reg->isVirtual()));
        and_inst->addOperand(std::make_unique<RegisterOperand>(
            rhs_bool_reg->getRegNum(), rhs_bool_reg->isVirtual()));
        parent_bb->addInstruction(std::move(and_inst));
    } else {  // LOr
        // 逻辑或：result = lhs_bool || rhs_bool
        // 实现：result = lhs_bool + rhs_bool, 然后如果result != 0则设为1
        auto add_reg = codeGen_->allocateReg();
        auto add_inst = std::make_unique<Instruction>(Opcode::ADD, parent_bb);
        add_inst->addOperand(std::make_unique<RegisterOperand>(
            add_reg->getRegNum(), add_reg->isVirtual()));
        add_inst->addOperand(std::make_unique<RegisterOperand>(
            lhs_bool_reg->getRegNum(), lhs_bool_reg->isVirtual()));
        add_inst->addOperand(std::make_unique<RegisterOperand>(
            rhs_bool_reg->getRegNum(), rhs_bool_reg->isVirtual()));
        parent_bb->addInstruction(std::move(add_inst));

        // 将和转换为布尔值（如果和非零则为1，否则为0）
        auto snez_inst = std::make_unique<Instruction>(Opcode::SNEZ, parent_bb);
        snez_inst->addOperand(std::make_unique<RegisterOperand>(
            result_reg->getRegNum(), result_reg->isVirtual()));
        snez_inst->addOperand(std::make_unique<RegisterOperand>(
            add_reg->getRegNum(), add_reg->isVirtual()));
        parent_bb->addInstruction(std::move(snez_inst));
    }

    // 建立指令结果值到寄存器的映射
    codeGen_->mapValueToReg(inst, result_reg->getRegNum(),
                            result_reg->isVirtual());

    return std::make_unique<RegisterOperand>(result_reg->getRegNum(),
                                             result_reg->isVirtual(),
                                             RegisterType::Integer);
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
                auto* reg_source =
                    dynamic_cast<RegisterOperand*>(source_operand.get());
                auto* reg_dest = dynamic_cast<RegisterOperand*>(dest_reg.get());

                // 根据源和目标寄存器类型选择正确的移动指令
                Opcode move_opcode;
                if (reg_source && reg_dest) {
                    if (reg_source->isFloatRegister() &&
                        reg_dest->isFloatRegister()) {
                        // 浮点到浮点：使用 FMOV_S
                        move_opcode = Opcode::FMOV_S;
                    } else if (!reg_source->isFloatRegister() &&
                               reg_dest->isFloatRegister()) {
                        // 整数到浮点：使用 FMV_W_X
                        move_opcode = Opcode::FMV_W_X;
                    } else if (reg_source->isFloatRegister() &&
                               !reg_dest->isFloatRegister()) {
                        // 浮点到整数：使用 FMV_X_W
                        move_opcode = Opcode::FMV_X_W;
                    } else {
                        // 整数到整数：使用 MV
                        move_opcode = Opcode::MV;
                    }
                } else {
                    // 默认使用 MV
                    move_opcode = Opcode::MV;
                }

                auto inst =
                    std::make_unique<Instruction>(move_opcode, parent_bb);

                inst->addOperand(std::move(dest_reg));  // rd
                inst->addOperand(std::make_unique<RegisterOperand>(
                    reg_source->getRegNum(), reg_source->isVirtual(),
                    reg_source->getRegisterType()));  // rs，保持原有类型
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

    // 根据返回值类型选择正确的返回寄存器
    bool is_float_return = ret_inst->getOperand(0)->getType()->isFloatType();
    std::string return_reg = is_float_return ? "fa0" : "a0";

    // 创建具有正确类型的目标寄存器操作数
    auto dest_reg = std::make_unique<RegisterOperand>(
        ABI::getRegNumFromABIName(return_reg),
        false,  // fa0/a0 are physical registers
        is_float_return ? RegisterType::Float : RegisterType::Integer);

    storeOperandToReg(std::move(ret_operand), std::move(dest_reg), parent_bb);
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
    // 需要根据参数类型和在同类型参数中的位置来确定寄存器

    // 遍历函数参数，分别计算整数和浮点参数的索引
    auto* function = argument->getParent();
    int int_arg_index = 0;
    int float_arg_index = 0;
    int current_arg_index = -1;
    bool is_current_float = false;

    for (auto arg_iter = function->arg_begin(); arg_iter != function->arg_end();
         ++arg_iter) {
        const auto* arg = arg_iter->get();
        if (arg == argument) {
            current_arg_index = int_arg_index + float_arg_index;
            is_current_float = arg->getType()->isFloatType();
            break;
        }

        if (arg->getType()->isFloatType()) {
            float_arg_index++;
        } else if (arg->getType()->isIntegerType() ||
                   arg->getType()->isPointerType()) {
            int_arg_index++;
        }
    }

    if (current_arg_index == -1) {
        throw std::runtime_error("Argument not found in function signature");
    }

    // 根据参数类型选择寄存器
    if (is_current_float) {
        // 浮点参数：fa0-fa7
        if (float_arg_index < 8) {
            std::string reg_name = "fa" + std::to_string(float_arg_index);
            unsigned reg_num = ABI::getRegNumFromABIName(reg_name);
            return std::make_unique<RegisterOperand>(reg_num, false,
                                                     RegisterType::Float);
        }
    } else {
        // 整数/指针参数：a0-a7
        if (int_arg_index < 8) {
            std::string reg_name = "a" + std::to_string(int_arg_index);
            unsigned reg_num = ABI::getRegNumFromABIName(reg_name);
            return std::make_unique<RegisterOperand>(reg_num, false,
                                                     RegisterType::Integer);
        }
    }

    // 对于超过8个的参数（无论整数还是浮点），需要从栈上读取
    if (parent_bb == nullptr) {
        throw std::runtime_error(
            "Cannot generate load instruction for stack argument without "
            "BasicBlock context");
    }

    // 获取当前参数的真实类型信息
    bool is_current_pointer = argument->getType()->isPointerType();

    // 计算栈上的偏移量：需要累计所有之前栈参数的实际大小
    int arg_offset = 0;

    // 遍历前面的所有栈参数（第9个参数开始）
    for (auto arg_iter = function->arg_begin(); arg_iter != function->arg_end();
         ++arg_iter) {
        const auto* arg = arg_iter->get();

        // 计算当前参数在总参数中的索引
        int total_index = 0;
        for (auto check_iter = function->arg_begin(); check_iter != arg_iter;
             ++check_iter) {
            total_index++;
        }

        if (arg == argument) {
            break;  // 找到当前参数，停止计算
        }

        // 只计算栈参数（第9个参数开始，索引≥8）的偏移
        if (total_index >= 8) {
            int this_arg_size;
            if (arg->getType()->isFloatType()) {
                this_arg_size = 8;  // 浮点参数，8字节
            } else if (arg->getType()->isPointerType()) {
                this_arg_size = 8;  // 指针参数，64位，8字节
            } else {
                this_arg_size = 4;  // 普通整数参数，32位，4字节
            }
            arg_offset += this_arg_size;
        }
    }

    auto arg_reg = codeGen_->allocateReg();

    // 根据参数类型选择加载指令
    Opcode load_opcode;
    if (is_current_float) {
        load_opcode = Opcode::FLD;
    } else if (is_current_pointer) {
        load_opcode = Opcode::LD;  // 指针使用64位加载
    } else {
        load_opcode = Opcode::LW;  // 普通整数使用32位加载
    }

    // 使用新的辅助函数生成加载指令
    generateMemoryInstruction(
        load_opcode,
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
        // 直接使用找到的寄存器操作数，保持正确的寄存器类型
        bool is_float_value = value->getType()->isFloatType();
        return std::make_unique<RegisterOperand>(
            foundReg.value()->getRegNum(), foundReg.value()->isVirtual(),
            is_float_value ? RegisterType::Float : RegisterType::Integer);
    }

    // 检查是否是全局变量
    if (auto* global_var = midend::dyn_cast<midend::GlobalVariable>(value)) {
        std::cout << "DEBUG: Found global variable reference: "
                  << global_var->getName()
                  << ", isConstant: " << global_var->isConstant()
                  << ", hasInitializer: " << global_var->hasInitializer()
                  << std::endl;

        // 1. 生成LA指令来获取地址
        auto global_addr_reg = codeGen_->allocateReg();
        auto global_addr_inst =
            std::make_unique<Instruction>(Opcode::LA, parent_bb);
        global_addr_inst->addOperand(std::make_unique<RegisterOperand>(
            global_addr_reg->getRegNum(), global_addr_reg->isVirtual()));  // rd
        global_addr_inst->addOperand(std::make_unique<LabelOperand>(
            global_var->getName()));  // global symbol
        parent_bb->addInstruction(std::move(global_addr_inst));

        // 2. 检查是否是数组类型 - 如果是数组，只返回地址，不加载值
        if (global_var->getValueType()->isArrayType()) {
            // 对于数组类型的全局变量，返回基地址用于后续的GEP计算
            std::cout << "DEBUG: Global array " << global_var->getName()
                      << " - returning base address for GEP" << std::endl;
            return std::make_unique<RegisterOperand>(
                global_addr_reg->getRegNum(), global_addr_reg->isVirtual());
        } else {
            // 对于标量类型的全局变量，生成加载指令来获取值
            bool is_float_value = global_var->getValueType()->isFloatType();
            auto value_reg = is_float_value ? codeGen_->allocateFloatReg()
                                            : codeGen_->allocateReg();
            Opcode load_opcode = is_float_value ? Opcode::FLW : Opcode::LW;

            auto load_inst =
                std::make_unique<Instruction>(load_opcode, parent_bb);
            load_inst->addOperand(std::make_unique<RegisterOperand>(
                value_reg->getRegNum(), value_reg->isVirtual(),
                is_float_value ? RegisterType::Float : RegisterType::Integer));
            load_inst->addOperand(std::make_unique<MemoryOperand>(
                std::make_unique<RegisterOperand>(global_addr_reg->getRegNum(),
                                                  global_addr_reg->isVirtual()),
                std::make_unique<ImmediateOperand>(0)));
            parent_bb->addInstruction(std::move(load_inst));

            return std::make_unique<RegisterOperand>(
                value_reg->getRegNum(), value_reg->isVirtual(),
                is_float_value ? RegisterType::Float : RegisterType::Integer);
        }
    }

    // 检查是否是常量
    if (midend::isa<midend::ConstantInt>(value)) {
        // 如果值的类型是浮点类型，即使它是ConstantInt，也应该作为浮点处理
        if (value->getType()->isFloatType()) {
            auto* int_const = midend::cast<midend::ConstantInt>(value);
            // 这是一个浮点常量，但被错误地创建为ConstantInt
            int32_t int_value = int_const->getSignedValue();

            // 将整数位模式重新解释为浮点数
            union {
                int32_t i;
                float f;
            } converter;
            converter.i = int_value;
            float float_value = converter.f;

            // 特殊处理浮点零值
            if (float_value == 0.0f) {
                auto float_reg = codeGen_->allocateFloatReg();
                codeGen_->mapValueToReg(value, float_reg->getRegNum(),
                                        float_reg->isVirtual());
                auto fcvt_inst =
                    std::make_unique<Instruction>(Opcode::FCVT_S_W, parent_bb);
                fcvt_inst->addOperand(std::make_unique<RegisterOperand>(
                    float_reg->getRegNum(), float_reg->isVirtual(),
                    RegisterType::Float));
                fcvt_inst->addOperand(
                    std::make_unique<RegisterOperand>("zero"));
                parent_bb->addInstruction(std::move(fcvt_inst));
                return std::make_unique<RegisterOperand>(float_reg->getRegNum(),
                                                         float_reg->isVirtual(),
                                                         RegisterType::Float);
            }

            // 对于非零浮点值，使用 LI + FMV_W_X 序列
            auto int_reg = codeGen_->allocateReg();
            auto li_inst = std::make_unique<Instruction>(Opcode::LI, parent_bb);
            li_inst->addOperand(std::make_unique<RegisterOperand>(
                int_reg->getRegNum(), int_reg->isVirtual()));
            li_inst->addOperand(std::make_unique<ImmediateOperand>(int_value));
            parent_bb->addInstruction(std::move(li_inst));

            auto float_reg = codeGen_->allocateFloatReg();
            codeGen_->mapValueToReg(value, float_reg->getRegNum(),
                                    float_reg->isVirtual());

            auto fmv_inst =
                std::make_unique<Instruction>(Opcode::FMV_W_X, parent_bb);
            fmv_inst->addOperand(std::make_unique<RegisterOperand>(
                float_reg->getRegNum(), float_reg->isVirtual(),
                RegisterType::Float));
            fmv_inst->addOperand(std::make_unique<RegisterOperand>(
                int_reg->getRegNum(), int_reg->isVirtual(),
                RegisterType::Integer));
            parent_bb->addInstruction(std::move(fmv_inst));

            return std::make_unique<RegisterOperand>(float_reg->getRegNum(),
                                                     float_reg->isVirtual(),
                                                     RegisterType::Float);
        }

        // 正常的整数常量处理
        // 判断范围，是否在 [-2048, 2047] 之间
        auto* int_const = midend::cast<midend::ConstantInt>(value);
        auto value_int = int_const->getSignedValue();
        std::cout << "DEBUG: Processing integer constant: " << value_int
                  << std::endl;
        constexpr int64_t IMM_MIN = -2048;
        constexpr int64_t IMM_MAX = 2047;
        auto signed_value = static_cast<int64_t>(value_int);
        if (signed_value >= IMM_MIN && signed_value <= IMM_MAX) {
            std::cout << "DEBUG: Returning immediate operand: " << value_int
                      << std::endl;
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

    // 检查是否是浮点常量
    if (midend::isa<midend::ConstantFP>(value)) {
        auto* float_const = midend::cast<midend::ConstantFP>(value);
        float float_value = float_const->getValue();

        // 特殊处理浮点零值
        if (float_value == 0.0f) {
            // 分配浮点寄存器
            auto float_reg = codeGen_->allocateFloatReg();
            // 不建立映射，避免寄存器复用问题
            // codeGen_->mapValueToReg(value, float_reg->getRegNum(),
            //                         float_reg->isVirtual());

            // 使用 fcvt.s.w 指令将整数零转换为浮点零
            auto fcvt_inst =
                std::make_unique<Instruction>(Opcode::FCVT_S_W, parent_bb);
            fcvt_inst->addOperand(std::make_unique<RegisterOperand>(
                float_reg->getRegNum(), float_reg->isVirtual(),
                RegisterType::Float));  // rd (float)
            fcvt_inst->addOperand(
                std::make_unique<RegisterOperand>("zero"));  // rs1 (int zero)
            parent_bb->addInstruction(std::move(fcvt_inst));

            // 返回浮点寄存器操作数
            return std::make_unique<RegisterOperand>(float_reg->getRegNum(),
                                                     float_reg->isVirtual(),
                                                     RegisterType::Float);
        }

        // 获取浮点数的32位二进制表示
        union {
            float f;
            int32_t i;
        } converter;
        converter.f = float_value;
        int32_t bit_pattern = converter.i;

        // 1. 分配整数寄存器用于加载位模式
        auto int_reg = codeGen_->allocateReg();
        auto li_inst = std::make_unique<Instruction>(Opcode::LI, parent_bb);
        li_inst->addOperand(std::make_unique<RegisterOperand>(
            int_reg->getRegNum(), int_reg->isVirtual()));  // rd
        li_inst->addOperand(std::make_unique<ImmediateOperand>(bit_pattern));
        parent_bb->addInstruction(std::move(li_inst));

        // 2. 分配浮点寄存器
        auto float_reg = codeGen_->allocateFloatReg();
        // 不建立映射，避免寄存器复用问题
        // codeGen_->mapValueToReg(value, float_reg->getRegNum(),
        //                         float_reg->isVirtual());

        // 3. 生成fmv.w.x指令：将位模式从整数寄存器移动到浮点寄存器
        auto fmv_inst =
            std::make_unique<Instruction>(Opcode::FMV_W_X, parent_bb);
        fmv_inst->addOperand(std::make_unique<RegisterOperand>(
            float_reg->getRegNum(), float_reg->isVirtual(),
            RegisterType::Float));  // rd (float)
        fmv_inst->addOperand(std::make_unique<RegisterOperand>(
            int_reg->getRegNum(), int_reg->isVirtual(),
            RegisterType::Integer));  // rs1 (int)
        parent_bb->addInstruction(std::move(fmv_inst));

        // 返回浮点寄存器操作数
        return std::make_unique<RegisterOperand>(float_reg->getRegNum(),
                                                 float_reg->isVirtual(),
                                                 RegisterType::Float);
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
            // 首先确定最终的基础类型
            const midend::Type* base_element_type = element_type;
            while (base_element_type->isArrayType()) {
                auto* arr_type =
                    static_cast<const midend::ArrayType*>(base_element_type);
                base_element_type = arr_type->getElementType();
            }

            // Get the expected size of each sub-array
            const auto* sub_array_type =
                static_cast<const midend::ArrayType*>(element_type);
            size_t sub_array_size = sub_array_type->getNumElements();

            // Get the expected number of sub-arrays
            const auto* outer_array_type =
                static_cast<const midend::ArrayType*>(type);
            size_t num_sub_arrays = outer_array_type->getNumElements();

            if (base_element_type->isIntegerType()) {
                // 整数类型的多维数组
                std::vector<int32_t> flattened_values;

                for (unsigned i = 0; i < num_sub_arrays; ++i) {
                    if (i < const_array->getNumElements()) {
                        // Process explicitly initialized sub-array
                        auto* element = const_array->getElement(i);
                        std::cout << "Processing nested int array element " << i
                                  << ": " << element->toString() << std::endl;

                        auto nested_init =
                            convertLLVMInitializerToConstantInitializer(
                                element, element_type);

                        // Track how many elements we've added for this
                        // sub-array
                        size_t sub_array_start = flattened_values.size();

                        // 将嵌套数组的值添加到展平数组中
                        std::visit(
                            [&flattened_values](const auto& value) {
                                using T = std::decay_t<decltype(value)>;
                                if constexpr (std::is_same_v<
                                                  T, std::vector<int32_t>>) {
                                    flattened_values.insert(
                                        flattened_values.end(), value.begin(),
                                        value.end());
                                } else if constexpr (std::is_same_v<T,
                                                                    int32_t>) {
                                    flattened_values.push_back(value);
                                }
                                // 对于其他类型，暂时忽略
                            },
                            nested_init);

                        // Pad with zeros if the sub-array is not fully
                        // initialized
                        size_t elements_added =
                            flattened_values.size() - sub_array_start;
                        if (elements_added < sub_array_size) {
                            flattened_values.insert(
                                flattened_values.end(),
                                sub_array_size - elements_added, 0);
                        }
                    } else {
                        // No initializer for this sub-array, fill with zeros
                        flattened_values.insert(flattened_values.end(),
                                                sub_array_size, 0);
                    }
                }

                std::cout << "Flattened int array size: "
                          << flattened_values.size() << std::endl;
                return flattened_values;

            } else if (base_element_type->isFloatType()) {
                // 浮点类型的多维数组
                std::vector<float> flattened_values;

                for (unsigned i = 0; i < num_sub_arrays; ++i) {
                    if (i < const_array->getNumElements()) {
                        // Process explicitly initialized sub-array
                        auto* element = const_array->getElement(i);
                        std::cout << "Processing nested float array element "
                                  << i << ": " << element->toString()
                                  << std::endl;

                        auto nested_init =
                            convertLLVMInitializerToConstantInitializer(
                                element, element_type);

                        // Track how many elements we've added for this
                        // sub-array
                        size_t sub_array_start = flattened_values.size();

                        // 将嵌套数组的值添加到展平数组中
                        std::visit(
                            [&flattened_values](const auto& value) {
                                using T = std::decay_t<decltype(value)>;
                                if constexpr (std::is_same_v<
                                                  T, std::vector<float>>) {
                                    flattened_values.insert(
                                        flattened_values.end(), value.begin(),
                                        value.end());
                                } else if constexpr (std::is_same_v<T, float>) {
                                    flattened_values.push_back(value);
                                }
                                // 对于其他类型，暂时忽略
                            },
                            nested_init);

                        // Pad with zeros if the sub-array is not fully
                        // initialized
                        size_t elements_added =
                            flattened_values.size() - sub_array_start;
                        if (elements_added < sub_array_size) {
                            flattened_values.insert(
                                flattened_values.end(),
                                sub_array_size - elements_added, 0.0f);
                        }
                    } else {
                        // No initializer for this sub-array, fill with zeros
                        flattened_values.insert(flattened_values.end(),
                                                sub_array_size, 0.0f);
                    }
                }

                std::cout << "Flattened float array size: "
                          << flattened_values.size() << std::endl;
                return flattened_values;
            } else {
                throw std::runtime_error(
                    "Unsupported multi-dimensional array element type");
            }

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