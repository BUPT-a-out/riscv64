#include "Visit.h"

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <vector>

#include "ABI.h"
#include "CodeGen.h"
#include "IR/IRPrinter.h"
#include "IR/Instructions.h"
#include "IR/Module.h"
#include "Instructions/All.h"
#include "MagicDivision.h"
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
    // create rv-function
    // TODO: extract function
    auto riscv_func = std::make_unique<Function>(func->getName());
    auto* func_ptr = riscv_func.get();
    parent_module->addFunction(std::move(riscv_func));

    // create BB
    // TODO: extract function
    for (const auto& bb : *func) {
        // codeGen_->mapBBToLabel(bb, bb->getName());
        auto new_riscv_bb =
            std::make_unique<BasicBlock>(func_ptr, bb->getName());
        auto* bb_ptr = new_riscv_bb.get();
        func_ptr->addBasicBlock(std::move(new_riscv_bb));
        func_ptr->mapBasicBlock(bb, bb_ptr);
    }

    // process args
    // TODO: extract function
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

                // TODO: 用continue减少嵌套层次
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
    processAllPhiNodes(func, func_ptr);

    // 为新函数清理函数级别的映射
    codeGen_->clearFunctionLevelMappings();

    // 此时 func_ptr 已经包含了所有基本块，开始维护 CFG
    createCFG(func_ptr);

    // TODO: extract function
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

            // TODO: 拆分成三个函数
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
            // 为PHI节点根据类型分配正确的寄存器
            bool is_float_phi = inst->getType()->isFloatType();
            // TODO: extract function
            auto phi_reg = is_float_phi ? codeGen_->allocateFloatReg()
                                        : codeGen_->allocateReg();
            codeGen_->mapValueToReg(
                inst, phi_reg->getRegNum(), phi_reg->isVirtual(),
                is_float_phi ? RegisterType::Float : RegisterType::Integer);
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
            return visitFloatBinaryOp(inst, parent_bb);
            break;
        case midend::Opcode::UAdd:
        case midend::Opcode::USub:
        case midend::Opcode::Not:
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
            // TODO: extract function
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

        // TODO: extract function
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

    // TODO: remove this switch
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

            throw std::runtime_error(
                "Unsupported trunc cast type: " + dest_type->toString() +
                " -> " + src_type->toString());
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

    // TODO: extract function
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
        // TODO: extract function: reg calculateOffset(BB parent, reg index_reg, uint stride)
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
    // 计算参数的实际大小（根据RISC-V64，i32和f32都是4字节）
    auto getArgSize = [](const midend::Type* type) -> size_t {
        if (type->isIntegerType()) {
            // i32类型
            return 4;
        } else if (type->isFloatType()) {
            // f32类型
            return 4;
        } else if (type->isPointerType()) {
            // 指针类型在RISC-V64上是8字节，但根据用户要求只支持i32和f32
            return 8;  // 保持现有行为，避免破坏指针处理
        }
        return 4;  // 默认4字节
    };
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

    // 处理参数传递 - 根据RISC-V ABI，前8个参数位置使用寄存器，其余用栈
    size_t num_args = called_func->getNumArgs();

    // 按照RISC-V ABI规范：前8个参数位置使用寄存器，其余参数使用栈
    std::vector<std::pair<size_t, bool>>
        stack_args;  // 记录需要通过栈传递的参数 (索引, 是否浮点)

    // 计算实际需要的栈空间大小，根据参数类型
    size_t stack_space = 0;
    for (size_t i = 8; i < num_args; ++i) {
        auto* arg = called_func->getArg(i);
        size_t arg_size = getArgSize(arg->getType());
        // RISC-V栈参数需要8字节对齐
        stack_space += (arg_size + 7) & ~7;  // 向上对齐到8字节边界
    }

    // 预计算栈参数的偏移
    std::vector<int64_t> stack_arg_offsets;
    int64_t current_offset = 0;
    for (size_t i = 8; i < num_args; ++i) {
        auto* arg = called_func->getArg(i);
        size_t arg_size = getArgSize(arg->getType());
        stack_arg_offsets.push_back(current_offset);
        // RISC-V栈参数需要8字节对齐
        current_offset += (arg_size + 7) & ~7;
    }

    // 一次性分配栈空间
    if (stack_space > 0) {
        if (isValidImmediateOffset(-stack_space)) {
            auto stack_alloc_inst =
                std::make_unique<Instruction>(Opcode::ADDI, parent_bb);
            stack_alloc_inst->addOperand(
                std::make_unique<RegisterOperand>("sp"));
            stack_alloc_inst->addOperand(
                std::make_unique<RegisterOperand>("sp"));
            stack_alloc_inst->addOperand(std::make_unique<ImmediateOperand>(
                -static_cast<int64_t>(stack_space)));
            parent_bb->addInstruction(std::move(stack_alloc_inst));
        } else {
            auto li_inst = std::make_unique<Instruction>(Opcode::LI, parent_bb);
            auto li_reg = codeGen_->allocateReg();
            li_inst->addOperand(std::make_unique<RegisterOperand>(
                li_reg->getRegNum(), li_reg->isVirtual()));
            li_inst->addOperand(std::make_unique<ImmediateOperand>(
                -static_cast<int64_t>(stack_space)));
            parent_bb->addInstruction(std::move(li_inst));

            auto add_inst =
                std::make_unique<Instruction>(Opcode::ADD, parent_bb);
            add_inst->addOperand(std::make_unique<RegisterOperand>("sp"));
            add_inst->addOperand(std::make_unique<RegisterOperand>("sp"));
            add_inst->addOperand(std::make_unique<RegisterOperand>(
                li_reg->getRegNum(), li_reg->isVirtual()));
            parent_bb->addInstruction(std::move(add_inst));
        }
    }

    // 处理所有参数
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

        if (arg_i < 8) {
            // 前8个参数使用寄存器
            std::string reg_name;
            if (is_float_arg) {
                reg_name = "fa" + std::to_string(arg_i);
            } else {
                reg_name = "a" + std::to_string(arg_i);
            }

            auto dest_reg = std::make_unique<RegisterOperand>(
                ABI::getRegNumFromABIName(reg_name), false,
                is_float_arg ? RegisterType::Float : RegisterType::Integer);

            // 获取源操作数并移动到目标寄存器
            auto source_value = visit(source_operand, parent_bb);
            storeOperandToReg(std::move(source_value), std::move(dest_reg),
                              parent_bb);
        } else {
            // 超过8个的参数使用栈传递，按正序放置
            auto source_value = visit(source_operand, parent_bb);

            std::unique_ptr<RegisterOperand> source_reg;
            if (source_value->getType() == OperandType::FrameIndex) {
                source_reg = immToReg(std::move(source_value), parent_bb);
            } else {
                source_reg = immToReg(std::move(source_value), parent_bb);
            }

            // 使用预计算的栈偏移
            int64_t stack_offset = stack_arg_offsets[arg_i - 8];

            // 根据参数类型选择存储指令
            Opcode store_opcode;
            if (is_float_arg) {
                store_opcode = Opcode::FSW;
            } else if (dest_arg->getType()->isPointerType()) {
                store_opcode = Opcode::SD;
            } else {
                store_opcode = Opcode::SW;
            }

            // 将参数存储在对应的栈位置
            generateMemoryInstruction(store_opcode, std::move(source_reg),
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

    // 调用后清理栈空间
    // 重要修复：确保栈参数在被调用函数执行期间保持有效
    if (stack_space > 0) {
        int64_t positive_space = static_cast<int64_t>(stack_space);
        if (isValidImmediateOffset(positive_space)) {
            // 栈空间在立即数范围内，直接使用 addi
            auto stack_restore_inst =
                std::make_unique<Instruction>(Opcode::ADDI, parent_bb);
            stack_restore_inst->addOperand(
                std::make_unique<RegisterOperand>("sp"));
            stack_restore_inst->addOperand(
                std::make_unique<RegisterOperand>("sp"));
            stack_restore_inst->addOperand(
                std::make_unique<ImmediateOperand>(positive_space));
            parent_bb->addInstruction(std::move(stack_restore_inst));
        } else {
            // 栈空间超出立即数范围，使用 li + add
            auto li_inst = std::make_unique<Instruction>(Opcode::LI, parent_bb);
            li_inst->addOperand(
                std::make_unique<RegisterOperand>(5, false));  // t0
            li_inst->addOperand(std::make_unique<ImmediateOperand>(
                static_cast<int64_t>(stack_space)));
            parent_bb->addInstruction(std::move(li_inst));

            auto add_inst =
                std::make_unique<Instruction>(Opcode::ADD, parent_bb);
            add_inst->addOperand(std::make_unique<RegisterOperand>("sp"));
            add_inst->addOperand(std::make_unique<RegisterOperand>("sp"));
            add_inst->addOperand(
                std::make_unique<RegisterOperand>(5, false));  // t0
            parent_bb->addInstruction(std::move(add_inst));
        }
    }

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

    // 只分配寄存器，不生成指令
    // 实际的PHI指令处理将在processDeferredPhiNode中完成
    bool is_float_phi = inst->getType()->isFloatType();
    auto phi_reg =
        is_float_phi ? codeGen_->allocateFloatReg() : codeGen_->allocateReg();

    // 记录PHI的映射
    codeGen_->mapValueToReg(
        inst, phi_reg->getRegNum(), phi_reg->isVirtual(),
        is_float_phi ? RegisterType::Float : RegisterType::Integer);

    // PHI块本身不生成任何指令，只返回寄存器
    return std::make_unique<RegisterOperand>(
        phi_reg->getRegNum(), phi_reg->isVirtual(),
        is_float_phi ? RegisterType::Float : RegisterType::Integer);
}

// 新的并行拷贝调度实现
void Visitor::processAllPhiNodes(const midend::Function* func,
                                 Function* parent_func) {
    // 为每个包含PHI节点的基本块收集所有PHI节点
    std::map<const midend::BasicBlock*, std::vector<const midend::Instruction*>>
        phi_map;

    for (const auto& bb : *func) {
        for (const auto& inst : *bb) {
            if (inst->getOpcode() == midend::Opcode::PHI) {
                phi_map[bb].push_back(inst);
            }
        }
    }

    // 为每个基本块处理其PHI节点
    for (const auto& [bb_midend, phi_nodes] : phi_map) {
        if (phi_nodes.empty()) continue;

        // 获取所有前驱边
        std::set<const midend::BasicBlock*> predecessors;
        for (const auto* phi_inst_ptr : phi_nodes) {
            const auto* phi_inst =
                dynamic_cast<const midend::PHINode*>(phi_inst_ptr);
            if (!phi_inst) continue;

            for (unsigned i = 0; i < phi_inst->getNumIncomingValues(); ++i) {
                predecessors.insert(phi_inst->getIncomingBlock(i));
            }
        }

        // 为每条前驱边生成并行拷贝
        for (const auto* pred_bb_midend : predecessors) {
            generateParallelCopyForEdge(phi_nodes, pred_bb_midend, parent_func);
        }
    }
}

// 为单条前驱边生成并行拷贝
void Visitor::generateParallelCopyForEdge(
    const std::vector<const midend::Instruction*>& phi_nodes,
    const midend::BasicBlock* pred_bb_midend, Function* parent_func) {
    auto* pred_bb = parent_func->getBasicBlock(pred_bb_midend);
    if (!pred_bb) {
        throw std::runtime_error("Predecessor block not found");
    }

    // 查找终结符指令位置
    auto insert_pos = pred_bb->end();
    for (auto it = pred_bb->begin(); it != pred_bb->end(); ++it) {
        auto* current_inst = it->get();
        if (current_inst->isTerminator()) {
            insert_pos = it;
            break;
        }
    }

    // 收集所有拷贝操作：dest_reg -> src_operand
    std::vector<CopyOperation> copy_ops;

    // 构建拷贝操作列表
    for (const auto* phi_inst_ptr : phi_nodes) {
        const auto* phi_inst =
            dynamic_cast<const midend::PHINode*>(phi_inst_ptr);
        if (!phi_inst) continue;

        // 找到对应前驱边的值
        const midend::Value* incoming_value = nullptr;
        for (unsigned i = 0; i < phi_inst->getNumIncomingValues(); ++i) {
            if (phi_inst->getIncomingBlock(i) == pred_bb_midend) {
                incoming_value = phi_inst->getIncomingValue(i);
                break;
            }
        }

        if (!incoming_value) continue;

        // 获取PHI节点的目标寄存器
        auto foundReg = findRegForValue(phi_inst_ptr);
        if (!foundReg.has_value()) {
            throw std::runtime_error("PHI register not found");
        }

        // 处理源操作数
        std::unique_ptr<MachineOperand> src_operand;
        bool is_constant = false;

        if (auto* const_int =
                midend::dyn_cast<midend::ConstantInt>(incoming_value)) {
            // 处理整数常量
            auto value_int = const_int->getSignedValue();
            src_operand = std::make_unique<ImmediateOperand>(value_int);
            is_constant = true;
        } else if (auto* const_float =
                       midend::dyn_cast<midend::ConstantFP>(incoming_value)) {
            // 处理浮点常量
            auto float_val = const_float->getValue();
            union {
                float f;
                int32_t i;
            } converter;
            converter.f = float_val;
            src_operand = std::make_unique<ImmediateOperand>(converter.i);
            is_constant = true;
        } else {
            // 处理寄存器操作数
            auto temp_bb = std::make_unique<BasicBlock>(pred_bb->getParent(),
                                                        "temp_phi_bb");
            src_operand = visit(incoming_value, temp_bb.get());

            // 将临时基本块中的指令移动到原基本块
            for (auto it = temp_bb->begin(); it != temp_bb->end();) {
                auto current_it = it++;
                auto inst_ptr = std::move(*current_it);
                inst_ptr->setParent(pred_bb);
                pred_bb->insert(insert_pos, std::move(inst_ptr));
            }
        }

        copy_ops.push_back(
            {foundReg.value(), std::move(src_operand), is_constant});
    }

    // 执行并行拷贝调度
    scheduleParallelCopy(copy_ops, pred_bb, insert_pos);
}

// 并行拷贝调度算法
void Visitor::scheduleParallelCopy(
    std::vector<CopyOperation>& copy_ops, BasicBlock* bb,
    std::list<std::unique_ptr<Instruction>>::const_iterator insert_pos) {
    if (copy_ops.empty()) return;

    // 分离常量拷贝和寄存器拷贝
    std::vector<CopyOperation> constant_copies;
    std::vector<CopyOperation> register_copies;

    for (auto& op : copy_ops) {
        if (op.is_constant) {
            constant_copies.push_back(std::move(op));
        } else {
            register_copies.push_back(std::move(op));
        }
    }

    // 1. 先处理寄存器到寄存器的拷贝（可能有依赖关系）
    scheduleRegisterCopies(register_copies, bb, insert_pos);

    // 2. 最后处理常量拷贝（没有依赖关系）
    for (auto& op : constant_copies) {
        generateCopyInstruction(op.dest_reg, std::move(op.src_operand), bb,
                                insert_pos);
    }
}

// 调度寄存器拷贝，处理依赖关系和环
void Visitor::scheduleRegisterCopies(
    std::vector<CopyOperation>& register_copies, BasicBlock* bb,
    std::list<std::unique_ptr<Instruction>>::const_iterator insert_pos) {
    if (register_copies.empty()) return;

    // 构建源寄存器和目标寄存器的集合
    std::set<int> source_regs;
    std::set<int> dest_regs;
    std::map<int, RegisterOperand*>
        dest_reg_operands;  // dest_reg_num -> RegisterOperand*
    std::map<int, RegisterOperand*>
        source_reg_operands;  // src_reg_num -> RegisterOperand*

    for (const auto& op : register_copies) {
        if (op.src_operand->getType() == OperandType::Register) {
            auto* src_reg =
                dynamic_cast<RegisterOperand*>(op.src_operand.get());
            if (src_reg) {
                int dest_num = op.dest_reg->getRegNum();
                int src_num = src_reg->getRegNum();

                // 跳过自拷贝
                if (dest_num != src_num) {
                    source_regs.insert(src_num);
                    dest_regs.insert(dest_num);
                    dest_reg_operands[dest_num] = op.dest_reg;
                    source_reg_operands[src_num] = src_reg;
                }
            }
        }
    }

    // 找到既是源又是目标的寄存器（需要临时寄存器保存）
    std::set<int> conflict_regs;
    std::map<int, int> temp_reg_map;  // original_reg -> temp_reg

    for (int reg : source_regs) {
        if (dest_regs.count(reg)) {
            conflict_regs.insert(reg);
        }
    }

    // 为冲突寄存器分配临时寄存器并保存值
    for (int reg : conflict_regs) {
        // 获取原始寄存器的类型
        auto* original_reg = source_reg_operands[reg];
        bool is_float = original_reg->isFloatRegister();

        // 根据寄存器类型分配临时寄存器
        std::unique_ptr<RegisterOperand> temp_reg;
        if (is_float) {
            temp_reg = codeGen_->allocateFloatReg();
        } else {
            temp_reg = codeGen_->allocateReg();
        }
        temp_reg_map[reg] = temp_reg->getRegNum();

        // 生成保存指令: temp_reg <- original_reg
        auto temp_operand = std::make_unique<RegisterOperand>(
            temp_reg->getRegNum(), temp_reg->isVirtual(),
            is_float ? RegisterType::Float : RegisterType::Integer);
        auto src_operand = std::make_unique<RegisterOperand>(
            reg, true, is_float ? RegisterType::Float : RegisterType::Integer);
        generateCopyInstruction(temp_operand.get(), std::move(src_operand), bb,
                                insert_pos);
    }

    // 执行所有拷贝操作，将冲突寄存器的源替换为临时寄存器
    for (auto& op : register_copies) {
        if (op.src_operand->getType() == OperandType::Register) {
            auto* src_reg =
                dynamic_cast<RegisterOperand*>(op.src_operand.get());
            if (src_reg) {
                int dest_num = op.dest_reg->getRegNum();
                int src_num = src_reg->getRegNum();

                // 跳过自拷贝
                if (dest_num == src_num) continue;

                // 如果源寄存器是冲突寄存器，使用临时寄存器
                if (temp_reg_map.count(src_num)) {
                    // 使用与源寄存器相同的类型
                    RegisterType reg_type = src_reg->getRegisterType();
                    auto temp_src = std::make_unique<RegisterOperand>(
                        temp_reg_map[src_num], true, reg_type);
                    generateCopyInstruction(op.dest_reg, std::move(temp_src),
                                            bb, insert_pos);
                } else {
                    generateCopyInstruction(
                        op.dest_reg, std::move(op.src_operand), bb, insert_pos);
                }
            }
        }
    }
}

// 生成单个拷贝指令
void Visitor::generateCopyInstruction(
    RegisterOperand* dest_reg, std::unique_ptr<MachineOperand> src_operand,
    BasicBlock* bb,
    std::list<std::unique_ptr<Instruction>>::const_iterator insert_pos) {
    if (src_operand->isImm()) {
        // 常量拷贝
        auto* imm_op = dynamic_cast<ImmediateOperand*>(src_operand.get());
        auto value = imm_op->getValue();

        if (dest_reg->isFloatRegister()) {
            // 浮点寄存器立即数加载：li temp_reg, imm; fmv.w.x dest_reg,
            // temp_reg
            auto temp_reg = codeGen_->allocateReg();  // 分配临时整数寄存器

            // 先加载立即数到临时整数寄存器
            auto li_inst = std::make_unique<Instruction>(Opcode::LI, bb);
            li_inst->addOperand(std::make_unique<RegisterOperand>(
                temp_reg->getRegNum(), temp_reg->isVirtual(),
                RegisterType::Integer));
            li_inst->addOperand(std::move(src_operand));
            bb->insert(insert_pos, std::move(li_inst));

            // 然后从整数寄存器移动到浮点寄存器
            auto fmv_inst = std::make_unique<Instruction>(Opcode::FMV_W_X, bb);
            fmv_inst->addOperand(std::make_unique<RegisterOperand>(
                dest_reg->getRegNum(), dest_reg->isVirtual(),
                RegisterType::Float));
            fmv_inst->addOperand(std::make_unique<RegisterOperand>(
                temp_reg->getRegNum(), temp_reg->isVirtual(),
                RegisterType::Integer));
            bb->insert(insert_pos, std::move(fmv_inst));
        } else {
            // 整数寄存器立即数加载
            constexpr int64_t IMM_MIN = -2048;
            constexpr int64_t IMM_MAX = 2047;

            if (value >= IMM_MIN && value <= IMM_MAX) {
                // 使用 addi rd, x0, imm
                auto addi_inst =
                    std::make_unique<Instruction>(Opcode::ADDI, bb);
                addi_inst->addOperand(std::make_unique<RegisterOperand>(
                    dest_reg->getRegNum(), dest_reg->isVirtual(),
                    dest_reg->getRegisterType()));
                addi_inst->addOperand(std::make_unique<RegisterOperand>(
                    0, false, RegisterType::Integer));  // x0
                addi_inst->addOperand(
                    std::make_unique<ImmediateOperand>(value));
                bb->insert(insert_pos, std::move(addi_inst));
            } else {
                // 使用 li 指令
                auto li_inst = std::make_unique<Instruction>(Opcode::LI, bb);
                li_inst->addOperand(std::make_unique<RegisterOperand>(
                    dest_reg->getRegNum(), dest_reg->isVirtual(),
                    dest_reg->getRegisterType()));
                li_inst->addOperand(std::make_unique<ImmediateOperand>(value));
                bb->insert(insert_pos, std::move(li_inst));
            }
        }
    } else {
        // 寄存器拷贝，根据寄存器类型选择指令
        auto* src_reg = dynamic_cast<RegisterOperand*>(src_operand.get());
        if (src_reg && dest_reg->isFloatRegister() &&
            src_reg->isFloatRegister()) {
            // 浮点寄存器之间的拷贝，使用 fsgnj.s 指令
            auto fsgnj_inst =
                std::make_unique<Instruction>(Opcode::FSGNJ_S, bb);
            fsgnj_inst->addOperand(std::make_unique<RegisterOperand>(
                dest_reg->getRegNum(), dest_reg->isVirtual(),
                dest_reg->getRegisterType()));
            fsgnj_inst->addOperand(std::make_unique<RegisterOperand>(
                src_reg->getRegNum(), src_reg->isVirtual(),
                src_reg->getRegisterType()));
            fsgnj_inst->addOperand(std::make_unique<RegisterOperand>(
                src_reg->getRegNum(), src_reg->isVirtual(),
                src_reg->getRegisterType()));  // FSGNJ.S 需要两个源操作数
            bb->insert(insert_pos, std::move(fsgnj_inst));
        } else {
            // 整数寄存器拷贝，使用 mv 指令
            auto mv_inst = std::make_unique<Instruction>(Opcode::MV, bb);
            mv_inst->addOperand(std::make_unique<RegisterOperand>(
                dest_reg->getRegNum(), dest_reg->isVirtual(),
                dest_reg->getRegisterType()));
            mv_inst->addOperand(std::move(src_operand));
            bb->insert(insert_pos, std::move(mv_inst));
        }
    }
}

// 延迟处理PHI节点的方法 - 简化版本（保留用于兼容性，但不再使用）
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

    // 对每个前驱块，在其终结符指令前插入赋值
    for (unsigned i = 0; i < phi_inst->getNumIncomingValues(); ++i) {
        auto* incoming_value = phi_inst->getIncomingValue(i);
        auto* incoming_bb_midend = phi_inst->getIncomingBlock(i);
        auto* incoming_bb = parent_func->getBasicBlock(incoming_bb_midend);

        if (!incoming_bb) {
            throw std::runtime_error("Incoming block not found for PHI");
        }

        // 查找终结符指令位置
        auto insert_pos = incoming_bb->end();
        for (auto it = incoming_bb->begin(); it != incoming_bb->end(); ++it) {
            auto* current_inst = it->get();
            if (current_inst->isTerminator()) {
                insert_pos = it;
                break;
            }
        }

        // 处理常量值
        std::unique_ptr<MachineOperand> value_operand;
        if (auto* const_int =
                midend::dyn_cast<midend::ConstantInt>(incoming_value)) {
            auto value_int = const_int->getSignedValue();
            constexpr int64_t IMM_MIN = -2048;
            constexpr int64_t IMM_MAX = 2047;

            if (value_int >= IMM_MIN && value_int <= IMM_MAX) {
                value_operand = std::make_unique<ImmediateOperand>(value_int);
            } else {
                // 对于大常量，生成li指令
                auto temp_reg = codeGen_->allocateReg();
                auto li_inst =
                    std::make_unique<Instruction>(Opcode::LI, incoming_bb);
                li_inst->addOperand(std::make_unique<RegisterOperand>(
                    temp_reg->getRegNum(), temp_reg->isVirtual(),
                    RegisterType::Integer));
                li_inst->addOperand(
                    std::make_unique<ImmediateOperand>(value_int));
                incoming_bb->insert(insert_pos, std::move(li_inst));
                value_operand = std::make_unique<RegisterOperand>(
                    temp_reg->getRegNum(), temp_reg->isVirtual(),
                    RegisterType::Integer);
            }
        } else if (auto* const_float =
                       midend::dyn_cast<midend::ConstantFP>(incoming_value)) {
            // 处理浮点常量
            auto temp_reg = codeGen_->allocateReg();
            auto float_val = const_float->getValue();

            // 生成浮点常量加载指令
            union {
                float f;
                int32_t i;
            } converter;
            converter.f = float_val;

            auto li_inst =
                std::make_unique<Instruction>(Opcode::LI, incoming_bb);
            li_inst->addOperand(std::make_unique<RegisterOperand>(
                temp_reg->getRegNum(), temp_reg->isVirtual(),
                RegisterType::Integer));
            li_inst->addOperand(
                std::make_unique<ImmediateOperand>(converter.i));
            incoming_bb->insert(insert_pos, std::move(li_inst));

            auto fmv_inst =
                std::make_unique<Instruction>(Opcode::FMV_W_X, incoming_bb);
            auto float_reg = codeGen_->allocateFloatReg();
            fmv_inst->addOperand(std::make_unique<RegisterOperand>(
                float_reg->getRegNum(), float_reg->isVirtual(),
                RegisterType::Float));
            fmv_inst->addOperand(std::make_unique<RegisterOperand>(
                temp_reg->getRegNum(), temp_reg->isVirtual(),
                RegisterType::Integer));
            incoming_bb->insert(insert_pos, std::move(fmv_inst));

            value_operand = std::make_unique<RegisterOperand>(
                float_reg->getRegNum(), float_reg->isVirtual(),
                RegisterType::Float);
        } else {
            // 对于非常量值，使用临时基本块生成指令
            auto temp_bb = std::make_unique<BasicBlock>(
                incoming_bb->getParent(), "temp_phi_bb");
            value_operand = visit(incoming_value, temp_bb.get());

            // 将临时基本块中的指令移动到原基本块
            for (auto it = temp_bb->begin(); it != temp_bb->end();) {
                auto current_it = it++;
                auto inst_ptr = std::move(*current_it);
                inst_ptr->setParent(incoming_bb);
                incoming_bb->insert(insert_pos, std::move(inst_ptr));
            }
        }

        // 创建赋值指令
        bool is_float_phi = inst->getType()->isFloatType();
        auto dest_reg = std::make_unique<RegisterOperand>(
            phi_reg_num, phi_is_virtual,
            is_float_phi ? RegisterType::Float : RegisterType::Integer);

        storeOperandToReg(std::move(value_operand), std::move(dest_reg),
                          incoming_bb, insert_pos);
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
        // 为了避免PHI节点处理时覆盖条件寄存器，我们保存条件值到临时寄存器
        auto condition_reg = immToReg(std::move(condition), parent_bb);
        auto temp_condition_reg = codeGen_->allocateReg();

        // 保存条件值到临时寄存器
        auto mv_inst = std::make_unique<Instruction>(Opcode::MV, parent_bb);
        mv_inst->addOperand(std::make_unique<RegisterOperand>(
            temp_condition_reg->getRegNum(), temp_condition_reg->isVirtual()));
        mv_inst->addOperand(std::move(condition_reg));
        parent_bb->addInstruction(std::move(mv_inst));

        // 使用临时寄存器进行条件跳转
        auto instruction =
            std::make_unique<Instruction>(Opcode::BNEZ, parent_bb);
        instruction->addOperand(std::make_unique<RegisterOperand>(
            temp_condition_reg->getRegNum(), temp_condition_reg->isVirtual()));
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

    // 获取指针操作数
    auto* pointer_operand = store_inst->getPointerOperand();

    // 先处理指针操作数，确保alloca指令被优先处理
    if (auto* alloca_inst =
            midend::dyn_cast<midend::AllocaInst>(pointer_operand)) {
        auto* sfm = parent_bb->getParent()->getStackFrameManager();
        int frame_id = sfm->getAllocaStackSlotId(alloca_inst);

        if (frame_id == -1) {
            // 如果还没有为这个alloca分配FI，现在分配
            visitAllocaInst(alloca_inst, parent_bb);
        }
    }

    // 获取存储的值，根据类型确保正确的寄存器类型
    auto raw_value_operand = visit(store_inst->getValueOperand(), parent_bb);

    // 检查目标指针类型
    auto* dest_type = store_inst->getPointerOperand()->getType();
    bool is_int_dest = false;
    if (auto* ptr_type = midend::dyn_cast<midend::PointerType>(dest_type)) {
        if (auto* element_type = ptr_type->getElementType()) {
            is_int_dest = element_type->isIntegerType();
        }
    }

    auto* value_to_store = store_inst->getValueOperand();
    bool should_use_float_reg =
        !is_int_dest && value_to_store->getType()->isFloatType();

    std::cout << "DEBUG: Store analysis - value type: "
              << value_to_store->getType()->toString()
              << ", dest is int: " << is_int_dest
              << ", will use float reg: " << should_use_float_reg << std::endl;

    // 根据值的实际类型选择合适的寄存器类型
    std::unique_ptr<MachineOperand> value_operand;
    if (should_use_float_reg) {
        value_operand = ensureFloatReg(std::move(raw_value_operand), parent_bb);
    } else {
        value_operand = immToReg(std::move(raw_value_operand), parent_bb);
    }

    // 根据value_operand的实际寄存器类型决定存储指令类型
    bool is_float_store = false;
    if (auto* reg_operand =
            dynamic_cast<RegisterOperand*>(value_operand.get())) {
        is_float_store = reg_operand->isFloatRegister();
    }
    std::cout << "DEBUG: Store analysis - value type: "
              << store_inst->getValueOperand()->getType()->toString()
              << ", actual register is float: " << is_float_store << std::endl;

    // 处理指针操作数 - 可能是 alloca 指令、GEP 指令或全局变量
    if (auto* alloca_inst =
            midend::dyn_cast<midend::AllocaInst>(pointer_operand)) {
        // 直接是 alloca 指令
        auto* sfm = parent_bb->getParent()->getStackFrameManager();
        int frame_id = sfm->getAllocaStackSlotId(alloca_inst);

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

        // 根据存储值的实际类型选择存储指令（与前面的类型检查保持一致）
        bool is_float_store = should_use_float_reg;
        std::cout << "DEBUG: Store instruction type check - is_float: "
                  << is_float_store << ", value type: "
                  << store_inst->getValueOperand()->getType()->toString()
                  << std::endl;
        Opcode store_opcode = is_float_store ? Opcode::FSW : Opcode::SW;
        std::cout << "DEBUG: Selected store opcode: "
                  << (store_opcode == Opcode::SW
                          ? "SW"
                          : (store_opcode == Opcode::FSW ? "FSW" : "OTHER"))
                  << std::endl;
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

        // 根据value_operand的实际寄存器类型选择存储指令
        Opcode store_opcode = is_float_store ? Opcode::FSW : Opcode::SW;
        std::cout << "DEBUG: Store to GEP - using float store: "
                  << is_float_store << ", value type: "
                  << store_inst->getValueOperand()->getType()->toString()
                  << std::endl;
        std::cout << "DEBUG: Selected store opcode for GEP: "
                  << (store_opcode == Opcode::SW
                          ? "SW"
                          : (store_opcode == Opcode::FSW ? "FSW" : "OTHER"))
                  << std::endl;
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

        // 根据value_operand的实际寄存器类型选择存储指令
        Opcode store_opcode = is_float_store ? Opcode::FSW : Opcode::SW;
        std::cout << "DEBUG: Store to Global - using float store: "
                  << is_float_store << ", value type: "
                  << store_inst->getValueOperand()->getType()->toString()
                  << std::endl;
        std::cout << "DEBUG: Selected store opcode for Global: "
                  << (store_opcode == Opcode::SW
                          ? "SW"
                          : (store_opcode == Opcode::FSW ? "FSW" : "OTHER"))
                  << std::endl;
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

        // If it's an immediate, load to register since it might be referenced
        if (operand->getType() == OperandType::Immediate) {
            auto* imm_operand = dynamic_cast<ImmediateOperand*>(operand.get());
            bool is_float_op = inst->getType()->isFloatType();
            std::unique_ptr<RegisterOperand> new_reg;

            if (is_float_op) {
                // 对于浮点一元加号，使用 ensureFloatReg 处理立即数
                auto temp_operand = std::make_unique<ImmediateOperand>(
                    imm_operand->getFloatValue());
                new_reg = ensureFloatReg(std::move(temp_operand), parent_bb);
            } else {
                // 对于整数一元加号，将立即数加载到整数寄存器
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::LI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    RegisterType::Integer));
                instruction->addOperand(std::make_unique<ImmediateOperand>(
                    imm_operand->getValue()));
                parent_bb->addInstruction(std::move(instruction));
            }

            // 建立映射
            codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                    new_reg->isVirtual());

            return std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                is_float_op ? RegisterType::Float : RegisterType::Integer);
        }

        throw std::runtime_error("Unsupported operand type for UAdd");
    }

    // Handle USub (unary minus): -operand
    if (inst->getOpcode() == midend::Opcode::USub) {
        auto operand = visit(inst->getOperand(0), parent_bb);

        // 检查是否为浮点操作
        bool is_float_op = inst->getType()->isFloatType();

        // If both are immediates, do constant folding but allocate to register
        if (operand->getType() == OperandType::Immediate) {
            auto* imm_operand = dynamic_cast<ImmediateOperand*>(operand.get());
            std::unique_ptr<RegisterOperand> new_reg;

            if (is_float_op) {
                // 常量折叠浮点取负，使用 ensureFloatReg 处理
                float result = -imm_operand->getFloatValue();
                auto result_operand =
                    std::make_unique<ImmediateOperand>(result);
                new_reg = ensureFloatReg(std::move(result_operand), parent_bb);
            } else {
                // 常量折叠整数取负，但结果分配到寄存器
                int32_t result = -imm_operand->getValue();
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::LI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    RegisterType::Integer));
                instruction->addOperand(
                    std::make_unique<ImmediateOperand>(result));
                parent_bb->addInstruction(std::move(instruction));
            }

            // 建立映射
            codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                    new_reg->isVirtual());

            return std::make_unique<RegisterOperand>(
                new_reg->getRegNum(), new_reg->isVirtual(),
                is_float_op ? RegisterType::Float : RegisterType::Integer);
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

            // 先判断是否有立即数 - 进行常量折叠但结果需要分配到寄存器
            if ((lhs->getType() == OperandType::Immediate) &&
                (rhs->getType() == OperandType::Immediate)) {
                auto* lhs_imm = dynamic_cast<ImmediateOperand*>(lhs.get());
                auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());

                if (is_float_op) {
                    // 常量折叠浮点加法，但结果分配到寄存器
                    float result =
                        lhs_imm->getFloatValue() + rhs_imm->getFloatValue();
                    auto result_operand =
                        std::make_unique<ImmediateOperand>(result);
                    new_reg =
                        ensureFloatReg(std::move(result_operand), parent_bb);
                } else {
                    // 常量折叠整数加法，但结果分配到寄存器
                    int32_t result = lhs_imm->getValue() + rhs_imm->getValue();
                    new_reg = codeGen_->allocateReg();
                    auto instruction =
                        std::make_unique<Instruction>(Opcode::LI, parent_bb);
                    instruction->addOperand(std::make_unique<RegisterOperand>(
                        new_reg->getRegNum(), new_reg->isVirtual(),
                        RegisterType::Integer));
                    instruction->addOperand(
                        std::make_unique<ImmediateOperand>(result));
                    parent_bb->addInstruction(std::move(instruction));
                }

                // 建立映射并返回
                codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                        new_reg->isVirtual());
                return std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    is_float_op ? RegisterType::Float : RegisterType::Integer);
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
                        std::make_unique<Instruction>(Opcode::ADDIW, parent_bb);

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
                        std::make_unique<Instruction>(Opcode::ADDW, parent_bb);
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
                    float result =
                        lhs_imm->getFloatValue() - rhs_imm->getFloatValue();
                    // 对于浮点常量，直接返回立即数（因为后续会通过FloatConstantPool处理）
                    return std::make_unique<ImmediateOperand>(result);
                } else {
                    int64_t result = lhs_imm->getValue() - rhs_imm->getValue();
                    // 分配寄存器并生成 li 指令
                    new_reg = codeGen_->allocateReg();
                    auto instruction =
                        std::make_unique<Instruction>(Opcode::LI, parent_bb);
                    instruction->addOperand(std::make_unique<RegisterOperand>(
                        new_reg->getRegNum(), new_reg->isVirtual()));
                    instruction->addOperand(
                        std::make_unique<ImmediateOperand>(result));
                    parent_bb->addInstruction(std::move(instruction));
                }
            } else if (is_float_op) {
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
                        std::make_unique<Instruction>(Opcode::ADDIW, parent_bb);
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
                        std::make_unique<Instruction>(Opcode::SUBW, parent_bb);
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

                // 计算常量值
                int64_t result = lhs_imm->getValue() * rhs_imm->getValue();

                // 分配寄存器并生成 li 指令
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::LI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));
                instruction->addOperand(
                    std::make_unique<ImmediateOperand>(result));
                parent_bb->addInstruction(std::move(instruction));
            } else {
                auto lhs_reg = immToReg(std::move(lhs), parent_bb);
                auto rhs_reg = immToReg(std::move(rhs), parent_bb);

                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::MULW, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::move(lhs_reg));       // rs1
                instruction->addOperand(std::move(rhs_reg));       // rs2

                parent_bb->addInstruction(std::move(instruction));
            }
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
                // 计算常量值
                int64_t result = lhs_imm->getValue() / rhs_imm->getValue();

                // 分配寄存器并生成 li 指令
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::LI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));
                instruction->addOperand(
                    std::make_unique<ImmediateOperand>(result));
                parent_bb->addInstruction(std::move(instruction));
            } else {
                // 检查右操作数是否为常量，如果是则尝试魔数除法优化
                if (rhs->getType() == OperandType::Immediate) {
                    auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                    auto divisor = static_cast<int32_t>(rhs_imm->getValue());

                    // 尝试使用魔数除法优化
                    if (MagicDivision::canOptimize(divisor)) {
                        auto lhs_reg = immToReg(std::move(lhs), parent_bb);
                        auto result_reg = MagicDivision::generateMagicDivision(
                            std::move(lhs_reg), divisor, parent_bb);

                        new_reg = std::move(result_reg);
                        break;
                    }
                }

                // 回退到标准除法指令
                auto lhs_reg = immToReg(std::move(lhs), parent_bb);
                auto rhs_reg = immToReg(std::move(rhs), parent_bb);

                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::DIVW, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::move(lhs_reg));       // rs1
                instruction->addOperand(std::move(rhs_reg));       // rs2

                parent_bb->addInstruction(std::move(instruction));
            }
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
                // 计算常量值
                int64_t result = lhs_imm->getValue() % rhs_imm->getValue();

                // 分配寄存器并生成 li 指令
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::LI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));
                instruction->addOperand(
                    std::make_unique<ImmediateOperand>(result));
                parent_bb->addInstruction(std::move(instruction));
            } else {
                // 检查右操作数是否为常量，如果是则尝试魔数取模优化
                if (rhs->getType() == OperandType::Immediate) {
                    auto* rhs_imm = dynamic_cast<ImmediateOperand*>(rhs.get());
                    auto divisor = static_cast<int32_t>(rhs_imm->getValue());

                    // 尝试使用魔数取模优化
                    if (MagicDivision::canOptimize(divisor)) {
                        auto lhs_reg = immToReg(std::move(lhs), parent_bb);
                        auto result_reg = MagicDivision::generateMagicModulo(
                            std::move(lhs_reg), divisor, parent_bb);

                        new_reg = std::move(result_reg);
                        break;
                    }
                }

                // 回退到标准取模指令
                auto lhs_reg = immToReg(std::move(lhs), parent_bb);
                auto rhs_reg = immToReg(std::move(rhs), parent_bb);

                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::REMW, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual()));  // rd
                instruction->addOperand(std::move(lhs_reg));       // rs1
                instruction->addOperand(std::move(rhs_reg));       // rs2

                parent_bb->addInstruction(std::move(instruction));
            }
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
                    std::make_unique<Instruction>(Opcode::SLLIW, parent_bb);
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
                    std::make_unique<Instruction>(Opcode::SLLW, parent_bb);
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
                    std::make_unique<Instruction>(Opcode::SRAIW, parent_bb);
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
                    std::make_unique<Instruction>(Opcode::SRAW, parent_bb);
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
                // 常量折叠比较，但结果分配到寄存器
                int32_t result =
                    (lhs_imm->getValue() > rhs_imm->getValue()) ? 1 : 0;
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::LI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    RegisterType::Integer));
                instruction->addOperand(
                    std::make_unique<ImmediateOperand>(result));
                parent_bb->addInstruction(std::move(instruction));

                // 建立映射并返回
                codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                        new_reg->isVirtual());
                return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                         new_reg->isVirtual(),
                                                         RegisterType::Integer);
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
                // 常量折叠比较，但结果分配到寄存器
                int32_t result =
                    (lhs_imm->getValue() == rhs_imm->getValue()) ? 1 : 0;
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::LI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    RegisterType::Integer));
                instruction->addOperand(
                    std::make_unique<ImmediateOperand>(result));
                parent_bb->addInstruction(std::move(instruction));

                // 建立映射并返回
                codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                        new_reg->isVirtual());
                return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                         new_reg->isVirtual(),
                                                         RegisterType::Integer);
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
                // 常量折叠比较，但结果分配到寄存器
                int32_t result =
                    (lhs_imm->getValue() != rhs_imm->getValue()) ? 1 : 0;
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::LI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    RegisterType::Integer));
                instruction->addOperand(
                    std::make_unique<ImmediateOperand>(result));
                parent_bb->addInstruction(std::move(instruction));

                // 建立映射并返回
                codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                        new_reg->isVirtual());
                return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                         new_reg->isVirtual(),
                                                         RegisterType::Integer);
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
                // 常量折叠比较，但结果分配到寄存器
                int32_t result =
                    (lhs_imm->getValue() < rhs_imm->getValue()) ? 1 : 0;
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::LI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    RegisterType::Integer));
                instruction->addOperand(
                    std::make_unique<ImmediateOperand>(result));
                parent_bb->addInstruction(std::move(instruction));

                // 建立映射并返回
                codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                        new_reg->isVirtual());
                return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                         new_reg->isVirtual(),
                                                         RegisterType::Integer);
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
                // 常量折叠比较，但结果分配到寄存器
                int32_t result =
                    (lhs_imm->getValue() <= rhs_imm->getValue()) ? 1 : 0;
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::LI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    RegisterType::Integer));
                instruction->addOperand(
                    std::make_unique<ImmediateOperand>(result));
                parent_bb->addInstruction(std::move(instruction));

                // 建立映射并返回
                codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                        new_reg->isVirtual());
                return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                         new_reg->isVirtual(),
                                                         RegisterType::Integer);
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
                // 常量折叠比较，但结果分配到寄存器
                int32_t result =
                    (lhs_imm->getValue() >= rhs_imm->getValue()) ? 1 : 0;
                new_reg = codeGen_->allocateReg();
                auto instruction =
                    std::make_unique<Instruction>(Opcode::LI, parent_bb);
                instruction->addOperand(std::make_unique<RegisterOperand>(
                    new_reg->getRegNum(), new_reg->isVirtual(),
                    RegisterType::Integer));
                instruction->addOperand(
                    std::make_unique<ImmediateOperand>(result));
                parent_bb->addInstruction(std::move(instruction));

                // 建立映射并返回
                codeGen_->mapValueToReg(inst, new_reg->getRegNum(),
                                        new_reg->isVirtual());
                return std::make_unique<RegisterOperand>(new_reg->getRegNum(),
                                                         new_reg->isVirtual(),
                                                         RegisterType::Integer);
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

    // 先计算左操作数并转换为布尔值
    auto lhs = visit(inst->getOperand(0), parent_bb);
    auto lhs_reg = immToReg(std::move(lhs), parent_bb);

    // 将左操作数转换为布尔值（0或1）
    auto lhs_bool_reg = codeGen_->allocateReg();
    auto lhs_snez_inst = std::make_unique<Instruction>(Opcode::SNEZ, parent_bb);
    lhs_snez_inst->addOperand(std::make_unique<RegisterOperand>(
        lhs_bool_reg->getRegNum(), lhs_bool_reg->isVirtual()));
    lhs_snez_inst->addOperand(std::move(lhs_reg));
    parent_bb->addInstruction(std::move(lhs_snez_inst));

    // 临时存储左操作数结果到result_reg
    auto mv_left_inst = std::make_unique<Instruction>(Opcode::MV, parent_bb);
    mv_left_inst->addOperand(std::make_unique<RegisterOperand>(
        result_reg->getRegNum(), result_reg->isVirtual()));
    mv_left_inst->addOperand(std::make_unique<RegisterOperand>(
        lhs_bool_reg->getRegNum(), lhs_bool_reg->isVirtual()));
    parent_bb->addInstruction(std::move(mv_left_inst));

    if (inst->getOpcode() == midend::Opcode::LAnd) {
        // 逻辑与的短路求值：只有左操作数为真时才计算右操作数
        // 如果左操作数为假，结果已经是0，跳过右操作数

        // 生成唯一标签
        // TODO: extract function
        auto skip_label =
            "logical_and_skip_" + std::to_string(codeGen_->getNextLabelNum());

        // 如果左操作数为假（0），跳过右操作数的计算
        auto beqz_inst = std::make_unique<Instruction>(Opcode::BEQZ, parent_bb);
        beqz_inst->addOperand(std::make_unique<RegisterOperand>(
            lhs_bool_reg->getRegNum(), lhs_bool_reg->isVirtual()));
        beqz_inst->addOperand(std::make_unique<LabelOperand>(skip_label));
        parent_bb->addInstruction(std::move(beqz_inst));

        // 左操作数为真，计算右操作数
        auto rhs = visit(inst->getOperand(1), parent_bb);
        auto rhs_reg = immToReg(std::move(rhs), parent_bb);

        auto rhs_bool_reg = codeGen_->allocateReg();
        auto rhs_snez_inst =
            std::make_unique<Instruction>(Opcode::SNEZ, parent_bb);
        rhs_snez_inst->addOperand(std::make_unique<RegisterOperand>(
            rhs_bool_reg->getRegNum(), rhs_bool_reg->isVirtual()));
        rhs_snez_inst->addOperand(std::move(rhs_reg));
        parent_bb->addInstruction(std::move(rhs_snez_inst));

        // 结果 = lhs_bool && rhs_bool（都是0或1，用按位与实现）
        auto and_inst = std::make_unique<Instruction>(Opcode::AND, parent_bb);
        and_inst->addOperand(std::make_unique<RegisterOperand>(
            result_reg->getRegNum(), result_reg->isVirtual()));
        and_inst->addOperand(std::make_unique<RegisterOperand>(
            lhs_bool_reg->getRegNum(), lhs_bool_reg->isVirtual()));
        and_inst->addOperand(std::make_unique<RegisterOperand>(
            rhs_bool_reg->getRegNum(), rhs_bool_reg->isVirtual()));
        parent_bb->addInstruction(std::move(and_inst));

        // skip_label位置使用NOP占位
        auto nop_inst = std::make_unique<Instruction>(Opcode::NOP, parent_bb);
        parent_bb->addInstruction(std::move(nop_inst));

    } else {  // LOr
        // 逻辑或的短路求值：只有左操作数为假时才计算右操作数
        // 如果左操作数为真，结果已经是1，跳过右操作数

        auto skip_label =
            "logical_or_skip_" + std::to_string(codeGen_->getNextLabelNum());

        // 如果左操作数为真（非0），跳过右操作数的计算
        auto bnez_inst = std::make_unique<Instruction>(Opcode::BNEZ, parent_bb);
        bnez_inst->addOperand(std::make_unique<RegisterOperand>(
            lhs_bool_reg->getRegNum(), lhs_bool_reg->isVirtual()));
        bnez_inst->addOperand(std::make_unique<LabelOperand>(skip_label));
        parent_bb->addInstruction(std::move(bnez_inst));

        // 左操作数为假，计算右操作数
        auto rhs = visit(inst->getOperand(1), parent_bb);
        auto rhs_reg = immToReg(std::move(rhs), parent_bb);

        auto rhs_bool_reg = codeGen_->allocateReg();
        auto rhs_snez_inst =
            std::make_unique<Instruction>(Opcode::SNEZ, parent_bb);
        rhs_snez_inst->addOperand(std::make_unique<RegisterOperand>(
            rhs_bool_reg->getRegNum(), rhs_bool_reg->isVirtual()));
        rhs_snez_inst->addOperand(std::move(rhs_reg));
        parent_bb->addInstruction(std::move(rhs_snez_inst));

        // 结果 = 右操作数的布尔值（因为左操作数为假）
        auto mv_rhs_inst = std::make_unique<Instruction>(Opcode::MV, parent_bb);
        mv_rhs_inst->addOperand(std::make_unique<RegisterOperand>(
            result_reg->getRegNum(), result_reg->isVirtual()));
        mv_rhs_inst->addOperand(std::make_unique<RegisterOperand>(
            rhs_bool_reg->getRegNum(), rhs_bool_reg->isVirtual()));
        parent_bb->addInstruction(std::move(mv_rhs_inst));

        // skip_label位置使用NOP占位
        auto nop_inst = std::make_unique<Instruction>(Opcode::NOP, parent_bb);
        parent_bb->addInstruction(std::move(nop_inst));
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
                auto* const source_imm =
                    dynamic_cast<ImmediateOperand*>(source_operand.get());
                auto* reg_dest = dynamic_cast<RegisterOperand*>(dest_reg.get());

                if (reg_dest && reg_dest->isFloatRegister()) {
                    // 浮点寄存器：需要先加载到整数寄存器，再移动到浮点寄存器
                    auto temp_reg = codeGen_->allocateReg();

                    // 1. li temp_reg, imm
                    auto li_inst =
                        std::make_unique<Instruction>(Opcode::LI, parent_bb);
                    li_inst->addOperand(std::make_unique<RegisterOperand>(
                        temp_reg->getRegNum(), temp_reg->isVirtual(),
                        RegisterType::Integer));
                    li_inst->addOperand(std::make_unique<ImmediateOperand>(
                        source_imm->getValue()));
                    parent_bb->insert(insert_pos, std::move(li_inst));

                    // 2. fmv.w.x dest_reg, temp_reg
                    auto fmv_inst = std::make_unique<Instruction>(
                        Opcode::FMV_W_X, parent_bb);
                    fmv_inst->addOperand(std::move(dest_reg));
                    fmv_inst->addOperand(std::make_unique<RegisterOperand>(
                        temp_reg->getRegNum(), temp_reg->isVirtual(),
                        RegisterType::Integer));
                    parent_bb->insert(insert_pos, std::move(fmv_inst));
                } else {
                    // 整数寄存器：直接使用 li 指令
                    auto li_inst =
                        std::make_unique<Instruction>(Opcode::LI, parent_bb);
                    li_inst->addOperand(std::move(dest_reg));  // rd
                    li_inst->addOperand(std::make_unique<ImmediateOperand>(
                        source_imm->getValue()));  // imm
                    parent_bb->insert(insert_pos, std::move(li_inst));
                }
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
// TODO: 修改getRegForValue, 直接返回optional
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
    // 计算参数的实际大小（根据RISC-V64，i32和f32都是4字节）
    auto getArgSize = [](const midend::Type* type) -> size_t {
        if (type->isIntegerType()) {
            // i32类型
            return 4;
        } else if (type->isFloatType()) {
            // f32类型
            return 4;
        } else if (type->isPointerType()) {
            // 指针类型在RISC-V64上是8字节，但根据用户要求只支持i32和f32
            return 8;  // 保持现有行为，避免破坏指针处理
        }
        return 4;  // 默认4字节
    };

    // 获取函数参数对应的物理寄存器或者栈帧
    // 根据RISC-V ABI：前8个参数位置使用寄存器，其余使用栈

    // 遍历函数参数，计算当前参数的位置索引
    auto* function = argument->getParent();
    int current_arg_position = -1;
    bool is_current_float = argument->getType()->isFloatType();

    for (auto arg_iter = function->arg_begin(); arg_iter != function->arg_end();
         ++arg_iter) {
        const auto* arg = arg_iter->get();
        current_arg_position++;

        if (arg == argument) {
            break;  // 找到当前参数
        }
    }

    if (current_arg_position == -1) {
        throw std::runtime_error("Argument not found in function signature");
    }

    // 前8个参数位置使用寄存器
    // TODO: 8个整数+8个浮点.
    if (current_arg_position < 8) {
        std::string reg_name;
        if (is_current_float) {
            reg_name = "fa" + std::to_string(current_arg_position);
        } else {
            reg_name = "a" + std::to_string(current_arg_position);
        }

        unsigned reg_num = ABI::getRegNumFromABIName(reg_name);
        return std::make_unique<RegisterOperand>(
            reg_num, false,
            is_current_float ? RegisterType::Float : RegisterType::Integer);
    }

    // 对于超过8个位置的参数，需要从栈上读取
    if (parent_bb == nullptr) {
        throw std::runtime_error(
            "Cannot generate load instruction for stack argument without "
            "BasicBlock context");
    }

    // 获取当前参数的真实类型信息
    bool is_current_pointer = argument->getType()->isPointerType();

    // 计算栈上的偏移量：栈参数在调用者推入的栈空间中
    // 新的栈布局（从高地址到低地址）：
    // - 调用者的栈参数（最后推入的参数在高地址）
    // - 返回地址 (ra)  ← s0 + 8
    // - 保存的帧指针   ← s0
    // - 被调用函数的局部变量和其他数据  ← sp
    //
    // 栈参数从最后一个开始，按倒序排列
    // 第arg_position个栈参数位于：s0 + 8 + (total_stack_args - (arg_position -
    // 8)) * 8

    // 计算栈参数的累积偏移（与调用端逻辑保持一致）
    int64_t arg_offset = 0;

    // 计算当前栈参数之前的所有栈参数的累积大小
    for (auto arg_iter = function->arg_begin(); arg_iter != function->arg_end();
         ++arg_iter) {
        const auto* arg = arg_iter->get();
        int arg_pos = std::distance(function->arg_begin(), arg_iter);

        // TODO: 修正计算
        if (arg_pos < 8) {
            continue;  // 跳过寄存器参数
        }

        if (arg == argument) {
            break;  // 找到当前参数，停止累积
        }

        // 计算参数大小并累积偏移
        size_t arg_size = getArgSize(arg->getType());
        // RISC-V栈参数需要8字节对齐
        arg_offset += (arg_size + 7) & ~7;
    }

    // 根据参数类型分配正确的寄存器类型
    std::unique_ptr<RegisterOperand> arg_reg;
    if (is_current_float) {
        arg_reg = codeGen_->allocateFloatReg();
    } else {
        arg_reg = codeGen_->allocateReg();
    }

    // 根据参数类型选择加载指令
    Opcode load_opcode;
    if (is_current_float) {
        load_opcode = Opcode::FLW;
    } else if (is_current_pointer) {
        load_opcode = Opcode::LD;  // 指针使用64位加载
    } else {
        load_opcode = Opcode::LW;  // 普通整数使用32位加载
    }

    // 使用新的辅助函数生成加载指令
    generateMemoryInstruction(
        load_opcode,
        std::make_unique<RegisterOperand>(
            arg_reg->getRegNum(), arg_reg->isVirtual(),
            is_current_float ? RegisterType::Float : RegisterType::Integer),
        std::make_unique<RegisterOperand>("s0"),  // 使用帧指针
        arg_offset, parent_bb);

    return std::make_unique<RegisterOperand>(
        arg_reg->getRegNum(), arg_reg->isVirtual(),
        is_current_float ? RegisterType::Float : RegisterType::Integer);
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

// TODO: 拆分函数
std::unique_ptr<MachineOperand> Visitor::visit(const midend::Value* value,
                                               BasicBlock* parent_bb) {
    // 处理值的访问
    // 检查是否已经处理过该值
    const auto foundReg = findRegForValue(value);
    if (foundReg.has_value()) {
        // 调试输出 - 特别关注PHI节点
        if (value->getName().find("phi") != std::string::npos) {
            std::cout << "DEBUG VISIT: Found register for PHI value "
                      << value->getName() << " -> reg "
                      << foundReg.value()->getRegNum() << std::endl;
        }
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
            // TODO: extract function
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
                // 不进行全局映射，避免跨基本块依赖
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
            // 不进行全局映射，避免跨基本块依赖

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
        // TODO: 复用isValidImmediateOffset(int64_t offset)
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
        // codeGen_->mapValueToReg(value, new_reg->getRegNum(),
        //                         new_reg->isVirtual());
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
// TODO: 还有用吗
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
// TODO: extract function: init array
// TODO: use template<typename T, typename ConstantType>
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
                                    for (const auto& v : value) {
                                        flattened_values.push_back(v);
                                    }
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
                                    for (const auto& v : value) {
                                        flattened_values.push_back(v);
                                    }
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
                // TODO: extract function
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