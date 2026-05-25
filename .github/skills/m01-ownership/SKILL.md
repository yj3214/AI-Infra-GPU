---
name: m01-ownership
description: "Mastering C++ Ownership: Move Semantics, RAII, and Reference Safety. Triggers: std::move, use-after-free, double-free, dangling pointer, copy elision, RVO, reference qualifiers."
---

# C++ Ownership (Move Semantics & RAII)

## Core Question

**Who owns this resource, and does it need to move?**

In C++, ownership is a discipline, not just a compiler check. You must decide:
- **Scope-bound?** Use Stack (RAII).
- **Exclusive?** Use `std::unique_ptr`.
- **Shared?** Use `std::shared_ptr`.
- **View?** Use `T*` or `T&` (non-owning).

## Error → Design Question

| Compiler/Sanitizer Error | Design Question                                            |
| ------------------------ | ---------------------------------------------------------- |
| **Double Free**          | Who owns this? Did you copy a raw pointer?                 |
| **Use After Free**       | Did a reference outlive its owner?                         |
| **Memory Leak**          | Where is the destructor called? (Did you use `new`?)       |
| **Object slicing**       | Why are you passing by value instead of pointer/reference? |
| **Moved-from usage**     | Why are you touching a variable after `std::move`?         |

## Thinking Prompt

1.  **Does it need a heap allocation?**
    - No? → Stack value (Rule of Zero).
    - Yes? → `std::unique_ptr` (Rule of Zero).
    - DO NOT write a destructor unless managing a non-RAII C handle.

2.  **Is it a Transfer or a Copy?**
    - Transfer? → `std::move`.
    - Copy? → Copy Constructor (or `clone()` pattern if polymorphic).

3.  **Is this a View?**
    - Yes? → `std::string_view`, `std::span`, or `const T&`.
    - Never pass `std::shared_ptr` unless transferring shared ownership.

## Trace Up / Down

-   **Trace Up** (Diagnosing):
    -   *Error*: "Segfault / Heap Corruption"
    -   *Ask*: Is there a raw pointer (`T*`) managing a resource?
    -   *Fix*: Wrap in `unique_ptr` or `vector`.

-   **Trace Down** (Implementing):
    -   *Intent*: "I want to pass this big object to a function and never see it again."
    -   *Code*: `void consume(BigThing t);` called with `consume(std::move(thing));`

## Quick Reference

| Pattern           | Cost           | Use When                               |
| ----------------- | -------------- | -------------------------------------- |
| **Value (Stack)** | Zero           | Default choice. Small/Medium objects.  |
| **`unique_ptr`**  | Zero           | Heap needed. Exclusive ownership.      |
| **`shared_ptr`**  | Atomic Inc/Dec | Cyclic graph or true shared ownership. |
| **`T&` (Ref)**    | Zero           | Parameter passing (non-null).          |
| **`T*` (Ptr)**    | Zero           | Parameter passing (nullable).          |
| **`std::move`**   | Zero           | Transferring ownership.                |
