#pragma once

#include <iostream>
#include <unordered_map>
#include <unordered_set>

#include "Instructions/All.h"

namespace riscv64 {

/**
 * @brief Value Reuse Pass for RISCV64 Backend - Simplified Version
 *
 * This pass operates on the RISCV64 IR and eliminates redundant load
 * instructions by tracking values in virtual registers.
 */
class ValueReusePass {
   public:
    ValueReusePass() = default;
    ~ValueReusePass() = default;

    /**
     * @brief Run the value reuse optimization on a RISCV64 function
     * @param riscv_function - The RISCV64 function to optimize
     * @return True if any optimizations were performed
     */
    bool runOnFunction(Function* riscv_function);

    /**
     * @brief Get statistics about the optimization
     */
    struct Statistics {
        int loadsAnalyzed = 0;
        int optimizationOpportunities = 0;
        int loadsEliminated = 0;
        int virtualRegsReused = 0;
    };

    const Statistics& getStatistics() const { return stats_; }

   private:
    Statistics stats_;

    // Mapping from immediate values to their first register
    std::unordered_map<int64_t, unsigned> immediateToFirstReg_;

        // Unified mapping from canonical memory addresses to their first loaded register
    // Key: canonical memory address string (e.g., "global:g_val.26" or "frame:FI#1:offset0")
    // Value: register number that holds the loaded value
    std::unordered_map<std::string, unsigned> memoryToFirstReg_;

    // Track instructions to be removed after optimization
    std::vector<Instruction*> instructionsToRemove_;

    /**
     * @brief Build mapping of immediate values to registers
     * @param riscv_function - Function to analyze
     */
    void buildImmediateMap(Function* riscv_function);

    /**
     * @brief Process a single RISCV64 basic block
     * @param riscv_bb - RISCV64 basic block to analyze
     * @return True if block was modified
     */
    bool processBasicBlock(BasicBlock* riscv_bb);

    /**
     * @brief Process a single RISCV64 basic block with global liveness tracking
     * @param riscv_bb - RISCV64 basic block to analyze
     * @param globalLiveRegs - Global set of live registers across blocks
     * @return True if block was modified
     */
    bool processBasicBlockGlobal(BasicBlock* riscv_bb,
                                 std::unordered_set<unsigned>& globalLiveRegs);

    /**
     * @brief Process a single RISCV64 instruction
     * @param riscv_inst - RISCV64 instruction to analyze
     * @param liveRegs - Set of currently live registers
     * @return True if instruction was modified
     */
    bool processInstruction(Instruction* riscv_inst,
                            std::unordered_set<unsigned>& liveRegs);

    /**
     * @brief Optimize load immediate instruction
     * @param inst - The LI instruction to optimize
     * @param liveRegs - Set of currently live registers
     * @return True if optimization was applied
     */
    bool optimizeLoadImmediate(Instruction* inst,
                               std::unordered_set<unsigned>& liveRegs);

    /**
     * @brief Optimize memory load instruction
     * @param inst - The memory load instruction to optimize
     * @param liveRegs - Set of currently live registers
     * @return True if optimization was applied
     */
    bool optimizeMemoryLoad(Instruction* inst,
                            std::unordered_set<unsigned>& liveRegs);

    /**
     * @brief Replace instruction with register move
     * @param inst - Instruction to replace
     * @param sourceReg - Source register number
     * @param destReg - Destination register number
     * @return True if replacement was successful
     */
    bool replaceWithMove(Instruction* inst, unsigned sourceReg,
                         unsigned destReg);

    /**
     * @brief Update register liveness information
     * @param inst - Current instruction
     * @param liveRegs - Set of currently live registers to update
     */
    void updateLiveness(Instruction* inst,
                        std::unordered_set<unsigned>& liveRegs);

    /**
     * @brief Check if instruction should be removed
     * @param inst - Instruction to check
     * @return True if instruction should be removed
     */
    bool shouldRemoveInstruction(Instruction* inst);

    /**
     * @brief Reset state for a new function
     */
    void resetState();

    /**
     * @brief Extract global variable name from LA instruction
     * @param inst - The LA instruction
     * @return Global variable name, or empty string if not a global var LA
     */
    std::string extractGlobalVarFromLA(Instruction* inst);

    /**
     * @brief Find the LA instruction that loads address for this LW instruction
     * @param lw_inst - The LW instruction
     * @param bb - Basic block containing the instruction
     * @return Pointer to the LA instruction, or nullptr if not found
     */
    Instruction* findCorrespondingLA(Instruction* lw_inst, BasicBlock* bb);
    
    /**
     * @brief Create a canonical memory address key for any type of memory access
     * @param lw_inst - The LW instruction to analyze
     * @param bb - Basic block containing the instruction
     * @return String representing the canonical memory address, or empty if not recognized
     */
    std::string createCanonicalMemoryKey(Instruction* lw_inst, BasicBlock* bb);
    
    /**
     * @brief Find the frameaddr instruction that loads address for this LW instruction
     * @param lw_inst - The LW instruction
     * @param bb - Basic block containing the instruction
     * @return Pointer to the frameaddr instruction, or nullptr if not found
     */
    Instruction* findCorrespondingFrameAddr(Instruction* lw_inst, BasicBlock* bb);
};

}  // namespace riscv64
