#include <string>
#include <vector>

namespace riscv64 {

class BasicBlock;  // 前向声明

class Function {
   public:
    explicit Function(const std::string& name) : name(name) {}

    void addBasicBlock(BasicBlock* block) { basic_blocks.push_back(block); }

    // 迭代器，便于遍历基本块
    using iterator = std::vector<BasicBlock*>::iterator;
    iterator begin() { return basic_blocks.begin(); }
    iterator end() { return basic_blocks.end(); }

    const std::string& getName() const { return name; }

    // 管理函数的栈帧信息
    void calculateStackFrame() {
        // ... 计算需要多大的栈空间，哪些寄存器需要保存等
    }

   private:
    std::string name;
    std::vector<BasicBlock*> basic_blocks;
    // ... 还可以包含栈帧信息 (StackFrameInfo), 保存的寄存器列表等
};

}  // namespace riscv64