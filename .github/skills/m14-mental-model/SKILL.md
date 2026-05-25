---
name: m14-mental-model
description: "C++ Mental Models: Pointer vs Reference, Initialization, Undefined Behavior."
---

# C++ Mental Models

## Core Question

**What happens in memory?**

- **Value**: Does it own the bytes?
- **Reference**: Is it an alias?
- **Pointer**: Is it nullable?

## Thinking Prompt

1.  **Is it a Copy or Reference?**
    - `auto x = y` (Copy).
    - `auto& x = y` (Reference).

2.  **Does it dangle?**
    - Returning `&local` is always wrong.

## Quick Reference

| Concept         | Mental Model                       |
| --------------- | ---------------------------------- |
| **`T&`**        | Guaranteed Alias.                  |
| **`T*`**        | Nullable Address (Requires check). |
| **`std::move`** | Cast to rvalue (Prepare to steal). |
