# Mythos · Deep Looping Reasoning Skill

## Introduction

Mythos is an AI skill based on looped deep reasoning mechanisms, designed to achieve more efficient and in-depth problem-solving through the following core technologies:

1. **Parcae LTI Stability Mechanism** - Ensures training and inference stability of looped models
2. **Adaptive Computation Time (ACT)** - Dynamically allocates computational resources, handling simple problems quickly while providing more thinking time for complex ones
3. **Mixture of Experts (MoE)** - Provides breadth and support across multiple domains and perspectives

## Core Design Philosophy

### Perfect Combination of Breadth and Depth

```
MoE (Mixture of Experts) provides Breadth
  ↓
  Multi-domain knowledge coverage
  Different experts handle different types of tasks
  Parallel multi-perspective thinking

+ Looping Reasoning provides Depth
  ↓
  Gradual in-depth reasoning process
  Each loop builds on the previous step's output
  Self-correction and optimization

= Mythos
  ↓
  Comprehensive and in-depth reasoning capability
```

## File Structure

```
mythos-skill/
├── SKILL.md                          # Main skill definition file
├── README.md                         # This file (Chinese)
├── README_EN.md                      # English version
├── references/
│   ├── research/
│   │   ├── 01-introduction.md          # Skill overview
│   │   ├── 02-parcae-stability.md    # Parcae LTI stability mechanism details
│   │   ├── 03-act-mechanism.md       # ACT mechanism details
│   │   ├── 04-moe-reasoning.md       # MoE application in deep reasoning
│   │   └── 05-core-algorithms.md     # Core algorithm implementation
│   └── sources/                       # Original papers and materials
└── scripts/                           # Related scripts
```

## Usage

### Trigger Words

- "深度推理" (Deep reasoning)
- "逐步分析" (Step-by-step analysis)
- "多角度思考" (Multi-perspective thinking)
- "循环推理" (Looping reasoning)
- "从多个维度分析" (Analyze from multiple dimensions)
- Similar complex problem-solving requests

### Usage Examples

#### Example 1: Complex Mathematical Proof

```
User: Use deep reasoning to help me prove that for all n ≥ 1, 1 + 2 + ... + n = n(n+1)/2

Mythos:
I discuss this problem with you using the Mythos deep looping reasoning framework,
based on Parcae stability, ACT dynamic computation, and MoE multi-expert architecture.

Let me analyze this problem from multiple perspectives...

[Breadth Analysis]
- Math Expert: This is a summation formula, should use mathematical induction
- Logic Expert: Need to verify the logical rigor of each step
- Structure Expert: Build a clear proof structure

[Depth Reasoning]
First Round: Identify proof method
This is a classic summation formula, mathematical induction is the most suitable method...

Second Round: Build proof structure
Base Step: Verify formula holds for n=1
Inductive Hypothesis: Assume formula holds for n=k
Inductive Step: Prove formula holds for n=k+1

Third Round: Detailed derivation
[Detailed mathematical derivation process]

Fourth Round: Verify boundary cases
For n=0, both sides equal 0, formula also holds

[Final Conclusion]
Proved: The summation formula holds for all non-negative integers n

Core insight of the reasoning path: The key to mathematical induction lies in cleverly
using the inductive hypothesis to transform the k+1 case into the k case.
```

#### Example 2: Multi-perspective System Design

```
User: Use deep reasoning to help me design a highly available distributed cache system

Mythos:
I discuss this problem with you using the Mythos deep looping reasoning framework...

[Breadth Analysis]
- Architecture Expert: Adopt layered architecture, use consistent hashing
- Code Expert: Use Redis, implement LRU eviction strategy
- Math Expert: Capacity = QPS × Size × Time
- Innovation Expert: Introduce AI prediction mechanism

[Depth Reasoning]
[Multiple rounds of design iteration...]

[Final Solution]
[Complete architecture design and innovation points]
```

## Technical Details

### Parcae Architecture

Parcae uses a three-block structure:

- **Prelude Block (P)**: Initializes computation, processes inputs and initial states
- **Recurrent Block (R)**: Executes looping reasoning, gradually deepens analysis
- **Coda Block (C)**: Finalizes sequences, produces output

### Core Algorithms

See `references/research/05-core-algorithms.md` for details, including:

1. Parcae LTI stability mechanism implementation
2. ACT dynamic stopping mechanism implementation
3. DeepSeek-style MoE implementation
4. Complete reasoning loop architecture
5. Application examples and evaluation methods

## Use Cases

### 1. Complex Mathematical Reasoning
- Step-by-step proof construction
- Multi-perspective mathematical problem analysis
- Adaptive computation depth

### 2. Algorithm Design and Analysis
- Algorithm complexity analysis
- Comparison of multiple algorithm solutions
- Step-by-step implementation and verification

### 3. System Architecture Design
- Multi-dimensional architecture evaluation
- Progressive design approach
- Risk and benefit analysis

### 4. Cross-domain Problem Solving
- Technology + business comprehensive analysis
- Multi-disciplinary knowledge integration
- Progressive solution building

## Skill Features

### Adaptive Computation
- Simple problems: 1-2 steps, quick response
- Complex problems: Automatically expand to more steps
- Resource efficiency optimization

### Stability Guarantee
- Spectral radius constraints based on Parcae
- Stable gradient flow
- Predictable training behavior

### Multi-expert Collaboration
- Code expert, math expert, logic expert, etc.
- Dynamic routing selection
- Load balancing mechanism

## Key Advantages

| Feature | Traditional Methods | Mythos |
|---------|---------------------|--------|
| Computation Depth | Fixed | Adaptive |
| Domain Coverage | Single | Multi-expert |
| Stability | Potentially unstable | Constructive guarantee |
| Resource Efficiency | Wasteful | Dynamic optimization |
| Reasoning Quality | Single perspective | Multi-perspective synthesis |

## References

1. Parcae — Scaling Laws for Stable Looped Language Models
2. Graves, A. (2016). "Adaptive Computation Time for Recurrent Neural Networks"
3. Dehghani, M., et al. (2019). "Universal Transformers"
4. DeepSeekMoE: Towards Ultimate Expert Utilization in Mixture-of-Experts Models

## Version History

- **v1.0** (2026-04-23)
  - Initial version
  - Complete SKILL.md definition
  - Core algorithm implementation documentation
  - Detailed research documentation for three technical mechanisms

## Contributions

This skill is based on the following papers and research:

- Parcae paper: Scaling Laws for Stable Looped Language Models
- Universal Transformers paper
- DeepSeekMoE related research

## Acknowledgments

Special thanks to **kyegomez** for open-sourcing the **Claude Mythos** project, which provided valuable inspiration and technical reference for deep looping reasoning systems.

## License

[To be added]

## Contact

For questions or suggestions, please provide feedback via GitHub Issues.
