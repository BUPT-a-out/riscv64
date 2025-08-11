#pragma once

#include <map>
#include <vector>
#include <memory>
#include <iostream>
#include <cassert>
#include <limits>
#include "RAGreedy/LiveIntervals.h"
#include "RAGreedy/SlotIndexes.h"

namespace riscv64 {

/// 活跃区间联合 - 多个虚拟寄存器活跃段的联合
class LiveIntervalUnion {
  // 使用STL map替代IntervalMap
  using LiveSegments = std::map<SlotIndex, std::shared_ptr<const LiveInterval>>;

public:
  using SegmentIter = LiveSegments::iterator;
  using ConstSegmentIter = LiveSegments::const_iterator;

private:
  unsigned Tag = 0;
  LiveSegments Segments;

public:
  LiveIntervalUnion() = default;
  ~LiveIntervalUnion() = default;

  // 移动构造和赋值
  LiveIntervalUnion(LiveIntervalUnion &&) = default;
  LiveIntervalUnion &operator=(LiveIntervalUnion &&) = default;

  // 禁用拷贝
  LiveIntervalUnion(const LiveIntervalUnion &) = delete;
  LiveIntervalUnion &operator=(const LiveIntervalUnion &) = delete;

  // 迭代器接口
  SegmentIter begin();
  SegmentIter end();
  SegmentIter find(SlotIndex x);
  ConstSegmentIter begin() const;
  ConstSegmentIter end() const;
  ConstSegmentIter find(SlotIndex x) const;

  bool empty() const;
  SlotIndex startIndex() const;
  SlotIndex endIndex() const;

  unsigned getTag() const;
  bool changedSince(unsigned tag) const;

  // 合并活跃区间
  void unify(std::shared_ptr<const LiveInterval> VirtReg, const LiveInterval &Interval);
  
  // 移除活跃区间
  void extract(std::shared_ptr<const LiveInterval> VirtReg, const LiveInterval &Interval);
  
  // 清空所有区间
  void clear();

  // 打印联合信息
  void print(std::ostream &OS) const;

  // 获取任意一个虚拟寄存器
  std::shared_ptr<const LiveInterval> getOneVReg() const;

  /// 查询类 - 检查单个活跃虚拟寄存器与联合的干扰
  class Query {
    const LiveIntervalUnion *LiveUnion = nullptr;
    const LiveInterval *LI = nullptr;
    unsigned SegmentIdx = 0;  // 当前处理的segment索引
    ConstSegmentIter LiveUnionI;
    std::vector<std::shared_ptr<const LiveInterval>> InterferingVRegs;
    bool CheckedFirstInterference = false;
    bool SeenAllInterferences = false;
    unsigned Tag = 0;
    unsigned UserTag = 0;

    unsigned collectInterferingVRegs(unsigned MaxInterferingRegs);
    bool isSeenInterference(std::shared_ptr<const LiveInterval> VirtReg) const;

  public:
    Query() = default;
    Query(const LiveInterval &li, const LiveIntervalUnion &liu);
    
    Query(const Query &) = delete;
    Query &operator=(const Query &) = delete;

    void reset(unsigned NewUserTag, const LiveInterval &NewLI,
               const LiveIntervalUnion &NewLiveUnion);

    void init(unsigned NewUserTag, const LiveInterval &NewLI,
              const LiveIntervalUnion &NewLiveUnion);

    bool checkInterference();

    const std::vector<std::shared_ptr<const LiveInterval>> &interferingVRegs(
        unsigned MaxInterferingRegs = std::numeric_limits<unsigned>::max());
  };

  /// 联合数组类
  class Array {
    std::vector<std::unique_ptr<LiveIntervalUnion>> LIUs;

  public:
    Array() = default;
    ~Array() = default;

    Array(Array &&Other) = default;
    Array &operator=(Array &&Other) = default;
    Array(const Array &) = delete;
    Array &operator=(const Array &) = delete;

    void init(unsigned size);
    unsigned size() const;
    void clear();

    LiveIntervalUnion& operator[](unsigned idx);
    const LiveIntervalUnion& operator[](unsigned idx) const;
  };
};

} // end namespace riscv64
