#include "Visit.h"

#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>

#include "ABI.h"
#include "CodeGen.h"
#include "IR/Module.h"
#include "Instructions/All.h"

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
        visit(bb, func_ptr);
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
void Visitor::visit(const midend::Instruction* inst, BasicBlock* parent_bb) {
    switch (inst->getOpcode()) {
        case midend::Opcode::Add:
        case midend::Opcode::Sub:
        case midend::Opcode::Mul:
        case midend::Opcode::Div:
            // 处理算术指令
            break;
        case midend::Opcode::Load:
        case midend::Opcode::Store:
            // 处理内存操作指令
            break;
        case midend::Opcode::Br:
            break;  // 处理分支指令
        case midend::Opcode::Ret:
            // 处理返回指令
            visitRetInstruction(inst, parent_bb);
            break;
        default:
            // 其他指令类型
            break;
    }
}

// 处理 ret 指令
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
    auto ret_value = visit(ret_inst->getOperand(0), parent_bb);

    switch (ret_value->getType()) {
        case OperandType::Immediate: {
            auto instruction =
                std::make_unique<Instruction>(Opcode::LI, parent_bb);
            auto* const ret_imm =
                dynamic_cast<ImmediateOperand*>(ret_value.get());

            instruction->addOperand(std::make_unique<RegisterOperand>(
                ABI::getRegNumFromABIName("a0")));  // rd
            instruction->addOperand(std::make_unique<ImmediateOperand>(
                ret_imm->getValue()));  // imm
            parent_bb->addInstruction(std::move(instruction));
            // std::cout << "Return immediate value: "
            //      << ret_imm->getValue() << std::endl;
            // assert(parent_bb->getInstructionCount() > 0 &&
            //        "Return instruction should be added to the basic block");
            break;
        }
        case OperandType::Register: {
            auto inst = std::make_unique<Instruction>(Opcode::MV, parent_bb);
            auto* reg_source = dynamic_cast<RegisterOperand*>(ret_value.get());

            inst->addOperand(std::make_unique<RegisterOperand>(
                ABI::getRegNumFromABIName("a0")));  // rd
            inst->addOperand(std::make_unique<RegisterOperand>(
                reg_source->getRegNum()));  // rs
            parent_bb->addInstruction(std::move(inst));
            break;
        }

        default:
            // TODO(rikka): 其他类型的返回值处理
            throw std::runtime_error(
                "Unsupported return value type: " +
                std::to_string(static_cast<int>(ret_value->getType())));
    }
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

std::unique_ptr<MachineOperand> Visitor::visit(const midend::Value* value,
                                               BasicBlock* parent_bb) {
    // 处理值的访问
    // 检查是否已经处理过该值
    const auto foundReg = findRegForValue(value);
    if (foundReg.has_value()) {
        // 直接使用找到的寄存器操作数
        return std::make_unique<RegisterOperand>(foundReg.value()->getRegNum());
    }

    // 是立即数，直接返回
    if (value->getType()->isIntegerType()) {
        return std::make_unique<ImmediateOperand>(
            dynamic_cast<const midend::ConstantInt*>(value)->getValue());
    }

    // 分配一个新的虚拟寄存器
    auto new_reg = codeGen_->allocateReg();
    // TODO(rikka): 其他整数类型处理...
    return std::make_unique<RegisterOperand>(new_reg->getRegNum());
}

// 访问常量
void Visitor::visit(const midend::Constant* /*constant*/) {}

// 访问 global variable
void Visitor::visit(const midend::GlobalVariable* /*var*/) {
    // 其他操作...
}

}  // namespace riscv64