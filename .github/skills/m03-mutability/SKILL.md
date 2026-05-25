---
name: m03-mutability
description: "Mastering C++ Const Correctness and Mutability. Triggers: const, mutable, logically const, bitwise const, data race, mutex, shared_mutex, thread safety."
---

# C++ Mutability & Const Correctness

## Core Question

**Who is allowed to change this state?**

C++ defaults to **Mutable**. You must opt-in to safety with `const`.
- **API Contract**: `const` methods promise not to change visible state.
- **Physical Constness**: The compiler ensures bits don't change.
- **Logical Constness**: You use `mutable` to change hidden state (like caches) inside `const` methods.

## Error → Design Question

| Issue                     | Design Question                                                         |
| ------------------------- | ----------------------------------------------------------------------- |
| **Discarded qualifiers**  | Are you trying to modify member data in a `const` function?             |
| **Data Race**             | Did you modify a `mutable` field from multiple threads without locking? |
| **Iterator Invalidation** | Are you modifying a container while reading it?                         |

## Thinking Prompt

1.  **Should this be `const`?**
    - Variables: Yes, default to `const` or `constexpr`.
    - Methods: Yes, unless it *must* modify state.
    - Pointers: `const T*` (ptr to const) vs `T* const` (const ptr).

2.  **Is internal state 'invariant' or 'cache'?**
    - Cache/Memoization? → Use `mutable` + `std::mutex`.
    - Actual state? → Non-const method.

3.  **Is it thread-safe?**
    - `const` methods implies safe for concurrent readers.
    - `mutable` members MUST be protected by synchronization in `const` methods.

## Trace Up / Down

-   **Trace Up**:
    -   *Issue*: "Undefined behavior in multithreaded app."
    -   *Cause*: A `const` method modified a `mutable` cache without a lock, assuming `const` meant safe.
    -   *Fix*: Add `std::mutex` to the class, lock it in the `const` method.

-   **Trace Down**:
    -   *Intent*: "I want a thread-safe lookup table."
    -   *Code*: `std::shared_mutex` allowing multiple readers (`shared_lock`) and one writer (`unique_lock`).

## Quick Reference

| Keyword                 | Meaning               | Use When                         |
| ----------------------- | --------------------- | -------------------------------- |
| **`const`**             | Read-only access      | Parameters, local vars, getters. |
| **`constexpr`**         | Compile-time constant | Constants, array sizes.          |
| **`mutable`**           | Modifiable in const   | Caches, Mutexes within a class.  |
| **`std::mutex`**        | Exclusive Lock        | Protecting mutable state.        |
| **`std::shared_mutex`** | Read/Write Lock       | Rare updates, frequent reads.    |
