---
name: m12-lifecycle
description: "Mastering C++ Lifecycle: RAII, Destructors, Static Initialization, Rule of 5."
---

# C++ Resource Lifecycle

## Core Question

**When does this die?**

- **Stack**: End of scope `}`.
- **Heap**: When `delete` (or smart pointer drop) occurs.
- **Static**: At program exit (reverse init order).

## Error → Design Question

| Issue              | Design Question                                                     |
| ------------------ | ------------------------------------------------------------------- |
| **Resource Leak**  | Did you manually `open()` without a wrapper class?                  |
| **Use After Free** | Did you capture a reference to a local variable in a lambda/thread? |
| **Static Fiasco**  | Do statics depend on each other? (Use Meyers Singleton).            |

## Thinking Prompt

1.  **Does it have a destructor?**
    - Yes → RAII. Good.
    - No → Wrap it.

2.  **Does it copy?**
    - `FILE*` cannot copy. Delete copy constructor.

## Quick Reference

| Pattern          | Use Case                                   |
| ---------------- | ------------------------------------------ |
| **RAII Wrapper** | `FileHandle`, `LockGuard`.                 |
| **Scope Guard**  | `std::scope_exit` (Cleanup callback).      |
| **Rule of 5**    | Copy/Move/Destructor implementation logic. |
