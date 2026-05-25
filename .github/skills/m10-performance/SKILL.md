---
name: m10-performance
description: "Mastering C++ Performance. Triggers: cache locality, heap allocation, inlining, SIMD, standard layout, string_view, reserve, pmr."
---

# C++ Performance

## Core Question

**Where is the data?**

- **Contiguous?** (Cache friendly) → `std::vector`.
- **Scattered?** (Cache miss) → `std::list`, `std::map`.

## Thinking Prompt

1.  **Count the Allocations.**
    - Can they be fused? (`reserve`).
    - Can they be removed? (Stack / SSO).

2.  **Measure, Don't Guess.**
    - Use Google Benchmark or Quick-Bench.

3.  **Data Layout.**
    - `struct` padding? Reorder members largest to smallest.
    - Pointer chasing? Flatten the graph.

## Quick Reference

| Technique         | Benefit        | Implementation                         |
| ----------------- | -------------- | -------------------------------------- |
| **`reserve()`**   | Avoid reallocs | Call before loop.                      |
| **`string_view`** | No copy string | Params.                                |
| **`std::vector`** | Cache locality | Default container.                     |
| **PMR**           | Custom alloc   | `std::pmr::monotonic_buffer_resource`. |
