---
name: m15-anti-pattern
description: "Common C++ Anti-Patterns. Triggers: global variables, new/delete, C-style cast, macros, void*."
---

# C++ Anti-Patterns

## Core Question

**Is this C or C++?**

- **C-Style**: `malloc`, `free`, `(int)x`, `void*`.
- **C++ Style**: `std::vector`, `std::unique_ptr`, `static_cast`, templates.

## Error → Design Question

| Issue                | Design Question                   |
| -------------------- | --------------------------------- |
| **Hard to refactor** | Are you using Macros?             |
| **Leak**             | Are you using `new`?              |
| **UB**               | Are you using `reinterpret_cast`? |

## Thinking Prompt

1.  **Can I delete this `new`?**
    - Use `make_unique`.

2.  **Can I remove this macro?**
    - Use `constexpr` or templates.

## Quick Reference

| Anti-Pattern       | Modern Fix                |
| ------------------ | ------------------------- |
| **`new T`**        | **`make_unique<T>`**      |
| **`T*` ownership** | **`unique_ptr<T>`**       |
| **`(T)ptr`**       | **`static_cast<T>(ptr)`** |
| **`#define`**      | **`constexpr`**           |
