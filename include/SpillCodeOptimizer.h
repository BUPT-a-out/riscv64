#include "Instructions/BasicBlock.h"
#include "Instructions/Function.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <map>

namespace riscv64 {

class SpillCodeOptimizer {
public:
    static void optimizeSpillCode(Function* function);
    
private:
    // 寄存器状态：描述寄存器中保存的是什么
    enum class RegisterState {
        UNKNOWN,        // 未知状态
        FI_ADDRESS,     // 保存FI的地址
        FI_VALUE,       // 保存FI对应内存的值
        OTHER          // 其他值
    };
    
    // 寄存器信息
    struct RegisterInfo {
        RegisterState state = RegisterState::UNKNOWN;
        unsigned fiId = 0;        // 如果是FI相关，对应的FI ID
        int lastDefPos = -1;      // 最后被定义的位置
        
        RegisterInfo() = default;
        RegisterInfo(RegisterState s, unsigned fi, int pos) 
            : state(s), fiId(fi), lastDefPos(pos) {}
    };
    
    // FI信息结构 - 修改为支持多个寄存器
    struct FIInfo {
        // 存储多个地址寄存器及其定义位置
        std::map<unsigned, int> addressRegs;
        // 存储多个值寄存器及其定义位置  
        std::map<unsigned, int> valueRegs;
        
        FIInfo() = default;
        
        // 清除所有地址寄存器
        void clearAllAddressRegs() {
            addressRegs.clear();
        }
        
        // 清除所有值寄存器
        void clearAllValueRegs() {
            valueRegs.clear();
        }
        
        // 检查是否有有效的地址寄存器
        bool hasAddressReg() const {
            return !addressRegs.empty();
        }
        
        // 检查是否有有效的值寄存器
        bool hasValueReg() const {
            return !valueRegs.empty();
        }
    };
    
    // 基本块的状态跟踪器
    struct BasicBlockTracker {
        std::unordered_map<unsigned, RegisterInfo> regMap;  // 寄存器 -> 信息
        std::unordered_map<unsigned, FIInfo> fiMap;         // FI ID -> 信息
        int currentPos = 0;                                 // 当前指令位置
        
        void updateRegister(unsigned reg, RegisterState state, unsigned fiId, int pos);
        void clearRegister(unsigned reg);
        void clearFIRegister(unsigned fiId, unsigned reg);
        bool canReuseFIAddress(unsigned fiId, unsigned& outReg);
        bool canReuseFIValue(unsigned fiId, unsigned& outReg);

        void addFIRegisterAssociation(unsigned fiId, unsigned reg, RegisterState state, int pos);
        void removeFIRegisterAssociation(unsigned fiId, unsigned reg, RegisterState state);

        std::vector<unsigned> getFIAddressRegs(unsigned fiId);
        std::vector<unsigned> getFIValueRegs(unsigned fiId);
    };
    
    // 优化单个基本块
    static void optimizeBasicBlock(BasicBlock* bb);
    
    // 处理不同类型的指令
    static void handleFrameAddr(BasicBlockTracker& tracker, Instruction* inst);
    static void handleLoad(BasicBlockTracker& tracker, Instruction* inst);
    static void handleStore(BasicBlockTracker& tracker, Instruction* inst);
    static void handleRegisterDef(BasicBlockTracker& tracker, Instruction* inst);
    
    // 辅助函数
    static bool isFrameAddrInst(Instruction* inst);
    static bool isLoadFromFI(Instruction* inst);
    static bool isStoreToFI(Instruction* inst);
    static unsigned getFrameIndex(Instruction* inst);
    static unsigned getTargetRegister(Instruction* inst);
    static unsigned getSourceRegister(Instruction* inst);
    static bool canOptimizeInstruction(Instruction* inst, BasicBlockTracker& tracker);
    static void applyOptimization(Instruction* inst, BasicBlockTracker& tracker);

    static void replaceWithMoveInstruction(Instruction* inst, 
                                                   unsigned targetReg, 
                                                   unsigned sourceReg, 
                                                   bool isFloat);
};

} // namespace riscv64