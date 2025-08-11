#include "CodeGen.h"
#include "Visit.h"

namespace riscv64 {

std::unique_ptr<RegisterOperand> Visitor::floatImmToReg(
    std::unique_ptr<ImmediateOperand> imm_operand, BasicBlock* parent_bb) {
    float float_value = imm_operand->getFloatValue();

    // 特殊处理浮点零值
    if (float_value == 0.0f) {
        // 分配浮点寄存器
        auto float_reg = codeGen_->allocateFloatReg();

        // 使用 fcvt.s.w 指令将整数零转换为浮点零
        auto fcvt_inst =
            std::make_unique<Instruction>(Opcode::FCVT_S_W, parent_bb);
        fcvt_inst->addOperand(
            cloneRegister(float_reg.get(), true));  // rd (float)
        fcvt_inst->addOperand(
            std::make_unique<RegisterOperand>("zero"));  // rs1 (int zero)
        parent_bb->addInstruction(std::move(fcvt_inst));

        return cloneRegister(float_reg.get(), true);
    }

    // 分配浮点寄存器
    auto float_reg = codeGen_->allocateFloatReg();

    // 获取或创建浮点常量的标签
    auto* pool = codeGen_->getFloatConstantPool();
    std::string label = pool->getOrCreateFloatConstant(float_value);

    // 分配临时整数寄存器用于地址计算
    auto addr_reg = codeGen_->allocateIntReg();

    // 生成 lui 指令：加载高20位地址
    auto lui_inst = std::make_unique<Instruction>(Opcode::LUI, parent_bb);
    lui_inst->addOperand(cloneRegister(addr_reg.get()));
    lui_inst->addOperand(
        std::make_unique<LabelOperand>(label + "@hi"));  // %hi(label)
    parent_bb->addInstruction(std::move(lui_inst));

    // 生成 addi 指令：加载低12位地址
    auto addi_inst = std::make_unique<Instruction>(Opcode::ADDI, parent_bb);
    addi_inst->addOperand(cloneRegister(addr_reg.get()));
    addi_inst->addOperand(
        std::make_unique<LabelOperand>(label + "@lo"));  // %lo(label)
    parent_bb->addInstruction(std::move(addi_inst));

    // 生成 flw 指令：从内存加载浮点数
    auto flw_inst = std::make_unique<Instruction>(Opcode::FLW, parent_bb);
    flw_inst->addOperand(cloneRegister(float_reg.get(), true));
    flw_inst->addOperand(std::make_unique<MemoryOperand>(
        std::make_unique<RegisterOperand>(addr_reg->getRegNum(),
                                          addr_reg->isVirtual()),
        std::make_unique<ImmediateOperand>(0)));
    parent_bb->addInstruction(std::move(flw_inst));

    return cloneRegister(float_reg.get(), true);
}

// 将立即数存到寄存器中，如果已经是寄存器则直接返回
std::unique_ptr<RegisterOperand> Visitor::immToReg(
    std::unique_ptr<MachineOperand> operand, BasicBlock* parent_bb) {
    if (operand->getType() == OperandType::Register) {
        auto* register_operand = dynamic_cast<RegisterOperand*>(operand.get());
        return cloneRegister(register_operand);
    }

    // 处理 FrameIndex 操作数
    if (operand->getType() == OperandType::FrameIndex) {
        auto* frame_operand = dynamic_cast<FrameIndexOperand*>(operand.get());
        if (frame_operand == nullptr) {
            throw std::runtime_error("Invalid frame index operand type: " +
                                     operand->toString());
        }

        // 生成一个新的寄存器，并使用 FRAMEADDR 指令获取帧地址
        auto new_reg = codeGen_->allocateIntReg();
        auto instruction =
            std::make_unique<Instruction>(Opcode::FRAMEADDR, parent_bb);
        instruction->addOperand(cloneRegister(new_reg.get()));  // rd
        instruction->addOperand(std::make_unique<FrameIndexOperand>(
            frame_operand->getIndex()));  // FI
        parent_bb->addInstruction(std::move(instruction));

        return cloneRegister(new_reg.get());
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
            return floatImmToReg(std::unique_ptr<ImmediateOperand>(imm_operand),
                                 parent_bb);
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
        auto new_reg = codeGen_->allocateIntReg();  // 分配一个新的寄存器
        instruction->addOperand(cloneRegister(new_reg.get()));  // rd
        instruction->addOperand(std::make_unique<ImmediateOperand>(
            imm_operand->getValue()));  // imm
        parent_bb->addInstruction(std::move(instruction));

        return new_reg;
    }

    throw std::runtime_error("Unsupported operand type in immToReg: " +
                             operand->toString());
}

}  // namespace riscv64