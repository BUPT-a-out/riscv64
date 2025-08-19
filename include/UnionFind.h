// 下面是一个针对 unsigned int 类型的虚拟寄存器编号的并查集 C++
// 实现，并附有详细的用法解释。这个实现包含了两个关键的性能优化：路径压缩（Path
// Compression） 和 按大小合并（Union by Size）。

// #include <iostream>
#include <map>
#include <numeric>
#include <vector>

namespace riscv64 {

// 为虚拟寄存器（由 unsigned int 标识）设计的并查集类
class UnionFind {
   public:
    // 构造函数
    UnionFind() = default;

    // 向并查集中添加一个新的虚拟寄存器。
    // 每个新寄存器初始时自成一个集合。
    void add(unsigned int vreg) {
        // 如果vreg已经存在，则不执行任何操作
        if (parent.count(vreg) != 0U) {
            return;
        }
        // 每个新元素都是自己的父节点
        parent[vreg] = vreg;
        // 初始时，每个集合的大小为1
        sz[vreg] = 1;
    }

    // 查找vreg所在集合的“代表元”（或称为“根节点”）
    // 这个实现包含了“路径压缩”优化。
    unsigned int find(unsigned int vreg) {
        // 如果vreg不存在，先添加它
        if (parent.count(vreg) == 0) {
            add(vreg);
        }

        // 如果当前节点的父节点就是它自己，那么它就是根
        if (parent[vreg] == vreg) {
            return vreg;
        }

        // 递归地查找根节点，并将路径上所有节点的父节点直接指向根
        // 这就是路径压缩，它会“压平”树的结构，加速后续查找
        return parent[vreg] = find(parent[vreg]);
    }

    // 合并包含vreg1和vreg2的两个集合
    // 这个实现包含了“按大小合并”优化。
    void unite(unsigned int vreg1, unsigned int vreg2) {
        // 1. 找到两个元素各自所在集合的根节点
        unsigned int root1 = find(vreg1);
        unsigned int root2 = find(vreg2);

        // 2. 如果它们已经在同一个集合中，则无需操作
        if (root1 == root2) {
            return;
        }

        // 3. 按大小合并：总是将较小的树合并到较大的树上
        //    这能有效防止树退化成链表，保持树的平衡
        if (sz[root1] < sz[root2]) {
            std::swap(root1, root2);  // 保证root1是较大树的根
        }

        // 将小树的根（root2）连接到大树的根（root1）上
        parent[root2] = root1;
        // 更新大树的大小
        sz[root1] += sz[root2];
    }

    void uniteWithDirection(unsigned int v_child, unsigned int v_parent) {
        unsigned int root_child = find(v_child);
        unsigned int root_parent = find(v_parent);

        if (root_child == root_parent) {
            return;
        }

        // 直接将 child 的根指向 parent 的根，忽略大小
        parent[root_child] = root_parent;
        // 更新大小信息
        sz[root_parent] += sz[root_child];
    }

    // 检查两个寄存器是否在同一个集合中（即是否等价）
    bool connected(unsigned int vreg1, unsigned int vreg2) {
        return find(vreg1) == find(vreg2);
    }

   private:
    // 使用 std::map 来存储父节点指针，这样就不需要预先知道寄存器的最大数量
    // parent[i] 表示元素 i 的父节点
    std::map<unsigned int, unsigned int> parent;

    // sz[i] 表示以 i 为根的集合的大小（元素数量）
    // 仅当 i 是根节点时，sz[i]才有意义
    std::map<unsigned int, unsigned int> sz;
};

}  // namespace riscv64