---
name: m04-zero-cost
description: "Mastering C++ Polymorphism: Templates, Concepts, and Virtual Functions. Triggers: templates, concepts, vtable, CRTP, static polymorphism, dynamic polymorphism, SFINAE."
---

# C++ Zero-Cost Abstractions

## Core Question

**When do we determine the type?**

- **Compile Time** (Static): Templates, Concepts, CRTP. Zero runtime cost, larger binary.
- **Runtime** (Dynamic): Virtual functions, `std::any`, `std::variant`. Flexible, vtable overhead.

## Error → Design Question

| Issue              | Design Question                                |
| ------------------ | ---------------------------------------------- |
| **Template spew**  | Are you missing Concepts constraints?          |
| **Linker error**   | Did you define template in .cpp instead of .h? |
| **Object slicing** | Did you assign Derived to Base value?          |
| **Slow build**     | Are you overusing headers/templates?           |

## Thinking Prompt

1.  **Is the set of types known at compile time?**
    - Yes? → Templates or `std::variant`.
    - No? → Inheritance (Virtual functions).

2.  **Do I need to store them in a list?**
    - Homogeneous? → `std::vector<T>`.
    - Heterogeneous? → `std::vector<std::unique_ptr<Base>>` or `std::vector<std::variant<...>>`.

3.  **Does the interface match exactly?**
    - Duck typing needed? → Templates (Concepts).
    - Strict hierarchy? → Inheritance.

## Trace Up / Down

-   **Trace Up**:
    -   *Issue*: "Virtual function call is too slow in tight loop."
    -   *Cause*: Indirect branch misprediction.
    -   *Fix*: Switch to Static Polymorphism (CRTP or Templates) if types are known.

-   **Trace Down**:
    -   *Intent*: "I want a function that accepts anything that has `.draw()`."
    -   *Code (C++20)*: `void render(Drawable auto& item) { item.draw(); }`

## Quick Reference

| Pattern             | Dispatch         | Cost                | Use When                     |
| ------------------- | ---------------- | ------------------- | ---------------------------- |
| **`virtual`**       | Dynamic          | Vtable + Cache miss | Plugins, Runtime extensions. |
| **Template**        | Static           | Code bloat          | High perf, Type deduction.   |
| **`std::function`** | Dynamic          | Alloc + Indirect    | Storing callbacks.           |
| **`std::variant`**  | Static branching | Branch switch       | Closed set of types.         |
| **CRTP**            | Static           | Zero                | Static inheritance.          |
