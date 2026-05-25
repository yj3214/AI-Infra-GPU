---
name: m02-resource
description: "Mastering C++ Smart Pointers: unique_ptr, shared_ptr, weak_ptr. Triggers: memory management, cycles, enable_shared_from_this, make_unique, make_shared, custom deleters."
---

# C++ Smart Pointers (Resource Management)

## Core Question

**How many owners does this resource to have?**

- **One**: `std::unique_ptr` (90% of cases).
- **Many**: `std::shared_ptr`.
- **Observer**: `std::weak_ptr`.

## Error → Design Question

| Issue                    | Design Question                                                  |
| ------------------------ | ---------------------------------------------------------------- |
| **Memory Leak (Cycles)** | Do you have `shared_ptr` pointing to each other? Cycle = Leak.   |
| **Dangling Pointer**     | Did you store a `weak_ptr` or raw pointer but check it too late? |
| **Double Free**          | Did you make two `unique_ptr`s from one raw pointer?             |
| **Performance**          | Are you copying `shared_ptr` unnecessarily? (Atomic ops).        |

## Thinking Prompt

1.  **Can I use `unique_ptr`?**
    - Always start here. It has zero overhead (size = raw pointer).
    - It forces you to think about ownership transfer (`std::move`).

2.  **Is this a Cycle?**
    - Parent -> Child (`shared_ptr`).
    - Child -> Parent (`weak_ptr`).
    - If Child holds `shared_ptr` to Parent, they never die.

3.  **Do I need a custom deleter?**
    - Managing C-APIs (`FILE*`, `SDL_Surface*`)?
    - `unique_ptr<FILE, DeclType(&fclose)>` handles this perfectly.

## Trace Up / Down

-   **Trace Up**:
    -   *Issue*: "Application memory usage grows endlessly but no 'leaks' reported by Valgrind."
    -   *Cause*: `shared_ptr` cycle. The reachable memory is technically "owned", just mutually.
    -   *Fix*: Break the cycle with `weak_ptr`.

-   **Trace Down**:
    -   *Intent*: "I need a list of heterogeneous objects."
    -   *Code*: `std::vector<std::unique_ptr<Base>>`

## Quick Reference

| Pattern                       | Cost          | Use When                                          |
| ----------------------------- | ------------- | ------------------------------------------------- |
| **`make_unique`**             | Zero          | Creating new heap objects.                        |
| **`make_shared`**             | 1 Alloc       | Creating access-controlled shared objects.        |
| **`weak_ptr`**                | Control Block | Breaking cycles, Caching.                         |
| **`enable_shared_from_this`** | Zero          | Need `shared_from_this()` inside member function. |
