#include "ValueReusePass.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "Instructions/All.h"
#include "Pass/Analysis/DominanceInfo.h"

namespace riscv64 {

bool ValueReusePass::runOnFunction(
    Function* riscv_function, const midend::Function* midend_function,
    const midend::AnalysisManager* analysisManager) {
    if (riscv_function == nullptr || riscv_function->empty() ||
        midend_function == nullptr || midend_function->empty()) {
        return false;
    }

    resetState();

    std::cout << "ValueReusePass: Analyzing function "
              << riscv_function->getName()
              << " with LI+LA optimization for debugging" << std::endl;

    bool changed = false;
    int optimizationCount = 0;

    // Get or compute dominance information for the midend function
    const midend::DominanceInfo* dominanceInfo = nullptr;
    std::unique_ptr<midend::DominanceInfoBase<false>> ownedDominanceInfo;

    if (analysisManager != nullptr) {
        // Try to get precomputed dominance analysis
        dominanceInfo =
            const_cast<midend::AnalysisManager*>(analysisManager)
                ->getAnalysis<midend::DominanceInfo>(
                    midend::DominanceAnalysis::getName(),
                    *const_cast<midend::Function*>(midend_function));
        if (dominanceInfo != nullptr) {
            std::cout
                << "  Using precomputed dominance info from AnalysisManager"
                << std::endl;
        }
    }

    const auto* domTree = dominanceInfo->getDominatorTree();
    if (domTree == nullptr || domTree->getRoot() == nullptr) {
        std::cout << "  Invalid dominator tree" << std::endl;
        return false;
    }

    std::cout << "  Dominator tree root: " << domTree->getRoot()->bb->getName()
              << std::endl;

    // Use the existing basic block mapping from the function
    // No need to create our own mapping since Function already maintains it
    std::cout << "  Using existing basic block mapping from Function"
              << std::endl;

    std::cout << "  Starting dominator tree traversal..." << std::endl;

    // Core optimization: DFS traversal of dominator tree with value tracking
    // std::unordered_map<const midend::Value*, RegisterOperand*> valueMap;
    bool modified = false;

    try {
        modified = traverseDominatorTree(domTree->getRoot(), riscv_function);
        std::cout << "  Dominator tree traversal completed successfully"
                  << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Error during dominator tree traversal: " << e.what()
                  << std::endl;
        return false;
    } catch (...) {
        std::cout << "  Unknown error during dominator tree traversal"
                  << std::endl;
        return false;
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
        std::cout << "  Stores processed: " << stats_.storesProcessed
                  << std::endl;
        std::cout << "  Calls processed: " << stats_.callsProcessed
                  << std::endl;
        std::cout << "  Memory invalidations: " << stats_.invalidations
                  << std::endl;
    }

    return modified;
}

bool ValueReusePass::traverseDominatorTree(midend::DominatorTree::Node* node,
                                           Function* riscv_function) {
    if (node == nullptr) {
        std::cout << "  Error: null node passed to traverseDominatorTree"
                  << std::endl;
        return false;
    }

    std::vector<MachineOperand*> definitionsInThisBlock;

    auto* midend_bb = node->bb;  // Get the midend basic block from the node
    auto* riscv_bb = riscv_function->getBasicBlock(midend_bb);
    if (riscv_bb == nullptr) {
        std::cout << "  Error: BasicBlock not found for midend block "
                  << midend_bb->getName() << std::endl;
        return false;
    }

    for (const auto& inst : *riscv_bb) {
        std::cout << "  Processing instruction: " << inst->toString()
                  << std::endl;

        // Process the instruction in the current basic block
        auto modified = modifyInstruction(
            inst.get(), riscv_bb, definitionsInThisBlock, availableValuesMap);

        // if (modified_inst.has_value()) {
        //     std::cout << "  Modified instruction: "
        //               << modified_inst->get()->toString() << std::endl;
        //     // 存入修改后的指令
        //     riscv_bb->replaceInstruction(inst.get(),
        //                                  std::move(modified_inst.value()));
        //     stats_.loadsEliminated++;
        //     stats_.virtualRegsReused++;

        //     // return true;
        // }
        if (modified) {
            std::cout << "  Instruction modified successfully: "
                      << inst->toString() << std::endl;
            stats_.loadsEliminated++;
            stats_.virtualRegsReused++;
        } 
    }

    // 递归下降
    for (auto& child : node->children) {
        if (child != nullptr) {
            std::cout << "  Recursing into child node: " << child->bb->getName()
                      << std::endl;
            traverseDominatorTree(child.get(), riscv_function);
            std::cout << "  Child node processed successfully: "
                      << child->bb->getName() << std::endl;

        } else {
            std::cout << "  Skipping null child node" << std::endl;
        }
    }

    // 状态回溯
    // 退出词法作用域
    for (auto& definition : definitionsInThisBlock) {
        if (definition != nullptr) {
            std::cout << "  Exiting lexical scope for definition: "
                      << definition->toString() << std::endl;
            // 清理当前块的定义
            availableValuesMap.erase(definition);
        } else {
            std::cout << "  Skipping null definition in lexical scope exit"
                      << std::endl;
        }
    }

    return false;
}

auto ValueReusePass::modifyInstruction(
    Instruction* inst, BasicBlock* /*riscv_bb*/,
    std::vector<MachineOperand*>& definitionsInThisBlock,
    std::unordered_map<MachineOperand*, RegisterOperand*>& valueMap) -> bool {
    switch (inst->getOpcode()) {
        case Opcode::LI: {
            auto* dest_reg =
                dynamic_cast<RegisterOperand*>(inst->getOperand(0));
            auto* imm_op = dynamic_cast<ImmediateOperand*>(inst->getOperand(1));
            if (dest_reg == nullptr || imm_op == nullptr) {
                return false;  // 无效的操作数
            }

            auto value = imm_op->getValue();
            std::cout << "  Load immediate: " << value << " -> reg"
                      << dest_reg->getRegNum() << std::endl;

            // 通过立即数值进行查找，而不是指针比较
            auto valueMapIter = valueMap.end();
            for (auto it = valueMap.begin(); it != valueMap.end(); ++it) {
                if (auto* key_imm =
                        dynamic_cast<ImmediateOperand*>(it->first)) {
                    if (key_imm->getValue() == imm_op->getValue()) {  // 比较立即数的值
                        valueMapIter = it;
                        break;
                    }
                }
            }

            if (valueMapIter != valueMap.end()) {
                auto* existing_reg = valueMapIter->second;
                if (existing_reg != nullptr) {
                    std::cout << "  OPTIMIZATION: Found existing register "
                              << existing_reg->toString()
                              << " with same immediate value " << value
                              << ", replacing with MV for "
                              << dest_reg->toString() << std::endl;
                    stats_.optimizationOpportunities++;
                    stats_.virtualRegsReused++;

                    // 保存寄存器信息在清空操作数之前
                    unsigned int dest_reg_num = dest_reg->getRegNum();
                    bool dest_is_virtual = dest_reg->isVirtual();
                    RegisterType dest_reg_type = dest_reg->getRegisterType();
                    
                    unsigned int existing_reg_num = existing_reg->getRegNum();
                    bool existing_is_virtual = existing_reg->isVirtual();
                    RegisterType existing_reg_type = existing_reg->getRegisterType();

                    // 替换为 MV 指令
                    inst->setOpcode(Opcode::MV);
                    inst->clearOperands();
                    inst->addOperand(std::make_unique<RegisterOperand>(
                        dest_reg_num, dest_is_virtual, dest_reg_type));
                    inst->addOperand(std::make_unique<RegisterOperand>(
                        existing_reg_num, existing_is_virtual, existing_reg_type));

                    return true;
                }
                std::cout
                    << "  No existing register found for immediate (nullptr) "
                    << value << ", keeping LI instruction" << std::endl;
                // 如果没有找到现有寄存器，保留 LI 指令
                definitionsInThisBlock.push_back(imm_op);
                valueMap[imm_op] = dest_reg;  // 保存映射关系

            } else {
                std::cout << "  No existing register found for immediate "
                          << value << ", keeping LI instruction" << std::endl;
                // 如果没有找到现有寄存器，保留 LI 指令
                definitionsInThisBlock.push_back(imm_op);
                valueMap[imm_op] = dest_reg;  // 保存映射关系
                std::cout << "  Adding new mapping for immediate "
                          << imm_op->toString() << " to register "
                          << dest_reg->toString() << std::endl;
            }
        } break;

        case Opcode::LW:
        case Opcode::LD:
        case Opcode::FLW: {
        }
        case Opcode::SW:
        case Opcode::SD:
        case Opcode::FSW: {
        }
        case Opcode::CALL: {
        }
        default: {
            // 对于其他指令，直接返回 nullptr，表示没有修改
            return false;
        }
    }
    return false;  // 如果没有修改，返回空
}

bool ValueReusePass::processBasicBlock(
    BasicBlock* riscv_bb, const midend::BasicBlock* midend_bb,
    std::unordered_map<const midend::Value*, RegisterOperand*>& valueMap,
    std::vector<const midend::Value*>& definitionsInThisBlock) {
    if (riscv_bb == nullptr) {
        return false;
    }

    std::cout << "    Processing basic block: " << riscv_bb->getLabel()
              << std::endl;

    bool modified = false;
    std::vector<BasicBlock::iterator> toErase;

    for (auto it = riscv_bb->begin(); it != riscv_bb->end(); ++it) {
        auto* inst = it->get();
        stats_.loadsAnalyzed++;

        if (processInstruction(inst, midend_bb, valueMap,
                               definitionsInThisBlock)) {
            toErase.push_back(it);
            modified = true;
        }
    }

    // Remove optimized instructions
    for (auto iter : toErase) {
        std::cout << "      Removing redundant instruction: "
                  << (*iter)->toString() << std::endl;
        riscv_bb->erase(iter);
        stats_.loadsEliminated++;
    }

    return modified;
}

bool ValueReusePass::processInstruction(
    Instruction* inst, const midend::BasicBlock* midend_bb,
    std::unordered_map<const midend::Value*, RegisterOperand*>& valueMap,
    std::vector<const midend::Value*>& definitionsInThisBlock) {
    Opcode opcode = inst->getOpcode();
    std::cout << "        Processing instruction: " << inst->toString()
              << std::endl;

    // switch (opcode) {
    //     case LI: {
    //         // 立即数加载指令 - 基于常量值进行复用判断
    //         const auto& operands = inst->getOperands();
    //         if (operands.size() >= 2) {
    //             auto* dest_reg =
    //                 dynamic_cast<RegisterOperand*>(operands[0].get());
    //             auto* imm_op =
    //                 dynamic_cast<ImmediateOperand*>(operands[1].get());

    //             if (dest_reg != nullptr && imm_op != nullptr) {
    //                 int64_t value = imm_op->getValue();
    //                 std::cout << "          Load immediate: " << value
    //                           << " -> reg" << dest_reg->getRegNum()
    //                           << std::endl;
    //                 stats_.loadsAnalyzed++;


    //                 // 检查是否已经有寄存器保存了相同的立即数值
    //                 // for (auto& pair : valueMap) {
    //                 //     if (auto* key_imm = dynamic_cast<ImmediateOperand*>(pair.first)) {
    //                 //         if (*key_imm == *imm_op) {
    //                 //             valueMapIter = pair;
    //                 //             break;
    //                 //         }
    //                 //     }
    //                 // }
    //                 // If not found, valueMapIter will be valueMap.end()
    //                 auto valueMapIter = valueMap.end();
    //                 if (valueMapIter != valueMap.end() &&
    //                     valueMapIter->second != nullptr) {
    //                     RegisterOperand* existing_reg = valueMapIter->second;

    //                     // 确保现有寄存器仍然有效且不是当前目标寄存器
    //                     if (existing_reg->getRegNum() !=
    //                         dest_reg->getRegNum()) {
    //                         std::cout << "          OPTIMIZATION: Found "
    //                                      "existing register "
    //                                   << existing_reg->getRegNum()
    //                                   << " with same immediate value " << value
    //                                   << ", replacing with mv for reg"
    //                                   << dest_reg->getRegNum() << std::endl;
    //                         stats_.optimizationOpportunities++;
    //                         stats_.virtualRegsReused++;

    //                         // 保存寄存器号在清空操作数之前
    //                         unsigned int dest_reg_num = dest_reg->getRegNum();
    //                         unsigned int existing_reg_num =
    //                             existing_reg->getRegNum();

    //                         // 实际应用优化：将 LI 指令替换为 MV 指令
    //                         inst->setOpcode(MV);
    //                         // 清空操作数并重新设置
    //                         inst->clearOperands();
    //                         // 创建新的寄存器操作数：目标仍然是原寄存器，源是已有的寄存器
    //                         auto dest_operand =
    //                             std::make_unique<RegisterOperand>(
    //                                 dest_reg_num, false, RegisterType::Integer);
    //                         auto source_operand =
    //                             std::make_unique<RegisterOperand>(
    //                                 existing_reg_num, false,
    //                                 RegisterType::Integer);
    //                         inst->addOperand(std::move(dest_operand));
    //                         inst->addOperand(std::move(source_operand));

    //                         std::cout << "          Applied optimization: MV r"
    //                                   << dest_reg_num << ", r"
    //                                   << existing_reg_num << std::endl;

    //                         // 记录当前定义（dest_reg仍然是有效的目标）
    //                         valueMap[immediateConstant] = dest_reg;
    //                         definitionsInThisBlock.push_back(immediateConstant);
    //                         return true;
    //                     }
    //                 }

    //                 // 记录当前定义
    //                 valueMap[immediateConstant] = dest_reg;
    //                 definitionsInThisBlock.push_back(immediateConstant);
    //                 std::cout << "          Recording immediate " << value
    //                           << " in reg" << dest_reg->getRegNum()
    //                           << std::endl;
    //             }
    //         }
    //         break;
    //     }

    //     case LA: {
    //         // 地址加载指令 - 跟踪地址计算的复用
    //         const auto& operands = inst->getOperands();
    //         if (operands.size() >= 2) {
    //             auto* dest_reg =
    //                 dynamic_cast<RegisterOperand*>(operands[0].get());
    //             if (dest_reg != nullptr) {
    //                 std::cout << "          Load address -> reg"
    //                           << dest_reg->getRegNum() << std::endl;
    //                 stats_.loadsAnalyzed++;

    //                 // 为地址加载建立映射，使用目标标识符
    //                 // 第二个操作数通常是符号或标签
    //                 std::string addressTarget = "unknown";
    //                 if (operands.size() > 1) {
    //                     // 尝试获取目标地址的标识符
    //                     if (auto* labelOp = dynamic_cast<LabelOperand*>(
    //                             operands[1].get())) {
    //                         // 获取标签的名称
    //                         addressTarget = "label:" + labelOp->getLabelName();
    //                     } else {
    //                         // 其他类型的地址操作数
    //                         addressTarget = "address_operand";
    //                     }
    //                 }

    //                 // 创建地址的规范化标识符，用作valueMap的key
    //                 std::string canonicalAddr = "addr:" + addressTarget;
    //                 std::cout
    //                     << "          Address canonical: " << canonicalAddr
    //                     << std::endl;

    //                 // 创建一个虚拟的midend::Value指针作为地址标识符
    //                 static std::unordered_map<std::string,
    //                                           std::unique_ptr<char>>
    //                     addressKeyMap;
    //                 if (addressKeyMap.find(canonicalAddr) ==
    //                     addressKeyMap.end()) {
    //                     addressKeyMap[canonicalAddr] =
    //                         std::make_unique<char>('\0');
    //                 }
    //                 const midend::Value* addressKey =
    //                     reinterpret_cast<const midend::Value*>(
    //                         addressKeyMap[canonicalAddr].get());

    //                 // 查找是否有相同地址已经被加载到寄存器
    //                 auto valueMapIter = valueMap.find(addressKey);
    //                 if (valueMapIter != valueMap.end() &&
    //                     valueMapIter->second != nullptr) {
    //                     RegisterOperand* existing_reg = valueMapIter->second;
    //                     if (existing_reg->getRegNum() !=
    //                         dest_reg->getRegNum()) {
    //                         std::cout << "          OPTIMIZATION: Found "
    //                                      "existing register "
    //                                   << existing_reg->getRegNum()
    //                                   << " with same address " << canonicalAddr
    //                                   << ", replacing with mv for reg"
    //                                   << dest_reg->getRegNum() << std::endl;
    //                         stats_.optimizationOpportunities++;
    //                         stats_.virtualRegsReused++;

    //                         // 保存寄存器号在清空操作数之前
    //                         unsigned int dest_reg_num = dest_reg->getRegNum();
    //                         unsigned int existing_reg_num =
    //                             existing_reg->getRegNum();

    //                         // 实际应用优化：将 LA 指令替换为 MV 指令
    //                         inst->setOpcode(MV);
    //                         inst->clearOperands();
    //                         auto dest_operand =
    //                             std::make_unique<RegisterOperand>(
    //                                 dest_reg_num, false, RegisterType::Integer);
    //                         auto source_operand =
    //                             std::make_unique<RegisterOperand>(
    //                                 existing_reg_num, false,
    //                                 RegisterType::Integer);
    //                         inst->addOperand(std::move(dest_operand));
    //                         inst->addOperand(std::move(source_operand));

    //                         std::cout << "          Applied optimization: MV r"
    //                                   << dest_reg_num << ", r"
    //                                   << existing_reg_num << std::endl;

    //                         // 记录当前定义 - 使用新创建的dest_operand
    //                         // 注意：不要使用原来的dest_reg，因为指令已经被修改
    //                         valueMap[addressKey] =
    //                             dynamic_cast<RegisterOperand*>(
    //                                 inst->getOperands()[0].get());
    //                         definitionsInThisBlock.push_back(addressKey);
    //                         return true;
    //                     }
    //                 }

    //                 // 记录当前地址映射
    //                 valueMap[addressKey] = dest_reg;
    //                 definitionsInThisBlock.push_back(addressKey);
    //                 std::cout << "          Recording address " << canonicalAddr
    //                           << " in reg" << dest_reg->getRegNum()
    //                           << std::endl;
    //             }
    //         }
    //         break;
    //     }

    //     case LW:
    //     case FLW: {
    //         // 内存加载指令 - 基于规范化内存地址进行复用判断
    //         const auto& operands = inst->getOperands();
    //         if (operands.size() >= 2) {
    //             auto* dest_reg =
    //                 dynamic_cast<RegisterOperand*>(operands[0].get());
    //             if (dest_reg != nullptr) {
    //                 std::cout << "          Memory load -> reg"
    //                           << dest_reg->getRegNum() << std::endl;
    //                 stats_.loadsAnalyzed++;

    //                 // 获取规范化的内存地址标识符
    //                 std::string canonicalAddress =
    //                     getCanonicalMemoryAddress(inst);
    //                 if (!canonicalAddress.empty()) {
    //                     // 查找对应的 midend Load 指令
    //                     const midend::Value* correspondingLoad =
    //                         findCorrespondingLoadInstruction(inst, midend_bb,
    //                                                          canonicalAddress);
    //                     if (correspondingLoad != nullptr) {
    //                         // 检查是否已经有寄存器保存了相同内存位置的值
    //                         auto loadMapIter = valueMap.find(correspondingLoad);
    //                         if (loadMapIter != valueMap.end() &&
    //                             loadMapIter->second != nullptr) {
    //                             RegisterOperand* existing_reg =
    //                                 loadMapIter->second;

    //                             if (existing_reg->getRegNum() !=
    //                                 dest_reg->getRegNum()) {
    //                                 std::cout << "          OPTIMIZATION: "
    //                                              "Found existing register "
    //                                           << existing_reg->getRegNum()
    //                                           << " with same memory value from "
    //                                           << canonicalAddress
    //                                           << ", could reuse for reg"
    //                                           << dest_reg->getRegNum()
    //                                           << std::endl;
    //                                 stats_.optimizationOpportunities++;
    //                                 stats_.virtualRegsReused++;
    //                                 // 这里可以实现实际的加载消除优化
    //                             }
    //                         }

    //                         // 记录当前加载
    //                         valueMap[correspondingLoad] = dest_reg;
    //                         definitionsInThisBlock.push_back(correspondingLoad);
    //                         std::cout << "          Recording load from "
    //                                   << canonicalAddress << " in reg"
    //                                   << dest_reg->getRegNum() << std::endl;
    //                     }
    //                 }
    //             }
    //         }
    //         break;
    //     }

    //     case SW:
    //     case FSW: {
    //         std::cout << "          Store instruction" << std::endl;
    //         stats_.storesProcessed++;
    //         // Store指令可能使某些内存值失效
    //         invalidateMemoryValues(valueMap, definitionsInThisBlock);
    //         break;
    //     }

    //     case CALL: {
    //         std::cout << "          Call instruction" << std::endl;
    //         stats_.callsProcessed++;
    //         // 函数调用可能修改内存，使某些值失效
    //         invalidateMemoryValues(valueMap, definitionsInThisBlock);
    //         break;
    //     }

    //     default:
    //         // 其他指令暂不处理
    //         break;
    // }

    // 不删除指令，只做分析和映射建立
    return false;
}

const midend::Value* ValueReusePass::findCorrespondingMidendInstruction(
    Instruction* inst, const midend::BasicBlock* midend_bb) {
    // This is a simplified heuristic approach
    // In a real implementation, we'd need a more sophisticated mapping
    // between RISCV64 instructions and midend values

    // For now, we'll use instruction position as a rough heuristic
    // This assumes instruction selection preserves relative ordering

    if (midend_bb == nullptr) {
        return nullptr;
    }

    // Count RISCV64 instruction position in its block
    auto* riscv_bb = inst->getParent();
    if (riscv_bb == nullptr) {
        return nullptr;
    }

    int inst_position = 0;
    for (auto it = riscv_bb->begin(); it != riscv_bb->end(); ++it) {
        if (it->get() == inst) {
            break;
        }
        inst_position++;
    }

    // Find corresponding midend instruction by position (very rough heuristic)
    int current_position = 0;
    for (auto& midend_inst : *midend_bb) {
        if (current_position == inst_position) {
            // Check if this midend instruction produces a value
            if (midend_inst->getType() &&
                !midend_inst->getType()->isVoidType()) {
                return midend_inst;
            }
        }
        current_position++;
    }

    return nullptr;
}

// 专门用于查找对应常量值的方法
const midend::Value* ValueReusePass::findCorrespondingConstantValue(
    Instruction* /*inst*/, const midend::BasicBlock* midend_bb, int64_t value) {
    if (midend_bb == nullptr) {
        return nullptr;
    }

    // 在 midend 基本块中查找产生相同常量值的指令
    for (const auto& midend_inst : *midend_bb) {
        // 检查是否是常量值
        if (const auto* constant =
                dynamic_cast<const midend::Constant*>(midend_inst)) {
            if (const auto* intConstant =
                    dynamic_cast<const midend::ConstantInt*>(constant)) {
                if (intConstant->getValue() == value) {
                    std::cout << "          Found matching constant " << value
                              << " in midend IR" << std::endl;
                    return constant;
                }
            }
        }

        // 也检查产生常量的指令（例如算术运算的结果）
        // 这里可以扩展更复杂的常量匹配逻辑
    }

    // 如果没有找到对应的常量，返回nullptr
    // 在实际实现中，可能需要更复杂的处理
    return nullptr;
}

// 专门用于查找对应加载指令的方法
const midend::Value* ValueReusePass::findCorrespondingLoadInstruction(
    Instruction* /*inst*/, const midend::BasicBlock* midend_bb,
    const std::string& canonicalAddress) {
    if (midend_bb == nullptr || canonicalAddress.empty()) {
        return nullptr;
    }

    // 在 midend 基本块中查找对应的 Load 指令
    for (const auto& midend_inst : *midend_bb) {
        if (midend_inst->getOpcode() == midend::Opcode::Load) {
            // 获取 Load 指令的源地址
            if (midend_inst->getNumOperands() > 0) {
                const midend::Value* sourceAddr = midend_inst->getOperand(0);

                // 将 midend 地址转换为规范化地址进行比较
                std::string midendCanonicalAddr =
                    getMidendCanonicalAddress(sourceAddr);
                if (midendCanonicalAddr == canonicalAddress) {
                    std::cout << "          Found matching load from "
                              << canonicalAddress << " in midend IR"
                              << std::endl;
                    return midend_inst;
                }
            }
        }
    }

    return nullptr;
}

// 获取 RISC-V 指令的规范化内存地址
std::string ValueReusePass::getCanonicalMemoryAddress(Instruction* inst) {
    const auto& operands = inst->getOperands();

    // 对于 LW/FLW 指令，操作数格式通常是: dest_reg, offset(base_reg)
    if (operands.size() >= 2) {
        // 第二个操作数应该是内存操作数
        if (auto* memOp = dynamic_cast<MemoryOperand*>(operands[1].get())) {
            auto* baseReg = memOp->getBaseReg();
            auto* offsetOp = memOp->getOffset();

            if (baseReg != nullptr && offsetOp != nullptr) {
                int64_t offset = offsetOp->getValue();

                // 构造规范化地址字符串
                std::string canonical = "reg" +
                                        std::to_string(baseReg->getRegNum()) +
                                        ":offset" + std::to_string(offset);

                // 如果是栈帧相关的寄存器，使用更具体的标识
                const int SP_REG_NUM = 2;  // sp 寄存器
                const int FP_REG_NUM = 8;  // fp 寄存器
                if (baseReg->getRegNum() == SP_REG_NUM) {
                    canonical = "stack:offset" + std::to_string(offset);
                } else if (baseReg->getRegNum() == FP_REG_NUM) {
                    canonical = "frame:offset" + std::to_string(offset);
                }

                std::cout << "          Canonical address: " << canonical
                          << std::endl;
                return canonical;
            }
        }
    }

    return "";
}

// 辅助方法：将 midend 地址值转换为规范化地址
std::string ValueReusePass::getMidendCanonicalAddress(
    const midend::Value* addr) {
    if (addr == nullptr) {
        return "";
    }

    // 检查是否是全局变量
    if (const auto* globalVar =
            dynamic_cast<const midend::GlobalVariable*>(addr)) {
        return "global:" + globalVar->getName();
    }

    // 检查是否是 Alloca 指令（栈变量）
    if (const auto* allocaInst =
            dynamic_cast<const midend::Instruction*>(addr)) {
        if (allocaInst->getOpcode() == midend::Opcode::Alloca) {
            // 使用指令的地址作为唯一标识符（避免reinterpret_cast）
            static size_t allocaCounter = 0;
            return "alloca:" + std::to_string(allocaCounter++);
        }
    }

    // 检查是否是 GEP 指令 - 简化处理，避免递归
    if (const auto* gepInst = dynamic_cast<const midend::Instruction*>(addr)) {
        if (gepInst->getOpcode() == midend::Opcode::GetElementPtr) {
            // 简化处理：只返回基本的GEP标识
            static size_t gepCounter = 0;
            return "gep:" + std::to_string(gepCounter++);
        }
    }

    // 默认情况：使用简单的计数器标识符
    static size_t valueCounter = 0;
    return "value:" + std::to_string(valueCounter++);
}

std::unordered_map<const midend::BasicBlock*, BasicBlock*>
ValueReusePass::createBasicBlockMapping(
    Function* riscv_function, const midend::Function* midend_function) {
    std::unordered_map<const midend::BasicBlock*, BasicBlock*> mapping;

    // Simple mapping based on position in function
    // This assumes that the basic block order is preserved during instruction
    // selection
    auto midend_it = midend_function->begin();
    auto riscv_it = riscv_function->begin();

    while (midend_it != midend_function->end() &&
           riscv_it != riscv_function->end()) {
        mapping[*midend_it] = riscv_it->get();
        ++midend_it;
        ++riscv_it;
    }

    std::cout << "  Created mapping for " << mapping.size() << " basic blocks"
              << std::endl;
    return mapping;
}

bool ValueReusePass::replaceWithMove(Instruction* inst,
                                     RegisterOperand* source_reg) {
    // This is a placeholder - in a real implementation, we'd generate a move
    // instruction
    (void)inst;
    (void)source_reg;
    return false;
}

void ValueReusePass::resetState() {
    stats_.loadsAnalyzed = 0;
    stats_.optimizationOpportunities = 0;
    stats_.valuesReused = 0;
    stats_.loadsEliminated = 0;
    stats_.virtualRegsReused = 0;
    stats_.storesProcessed = 0;
    stats_.callsProcessed = 0;
    stats_.invalidations = 0;
}

void ValueReusePass::invalidateMemoryValues(
    std::unordered_map<const midend::Value*, RegisterOperand*>& valueMap,
    std::vector<const midend::Value*>& definitionsInThisBlock) {
    // Store和Call指令会使内存相关的值失效
    // 我们需要保守地清除所有可能受影响的值

    std::vector<const midend::Value*> toInvalidate;

    // 识别需要失效的值（地址加载、内存加载等）
    for (const auto& pair : valueMap) {
        const midend::Value* key = pair.first;
        // 检查是否是我们创建的特殊键（地址或内存加载）
        // 由于我们使用了伪造的指针，这里做简单的范围检查
        if (key != nullptr) {
            toInvalidate.push_back(key);
        }
    }

    std::cout << "        Conservative invalidation of " << toInvalidate.size()
              << " tracked values" << std::endl;

    // 移除失效的值，但不要添加到definitionsInThisBlock
    // 因为这些不是在当前块中定义的新值
    for (const auto* key : toInvalidate) {
        valueMap.erase(key);
    }

    stats_.invalidations++;
}

}  // namespace riscv64
