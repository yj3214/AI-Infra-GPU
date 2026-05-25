---
name: m06-error-handling
description: "Mastering C++ Error Handling. Triggers: exceptions, try-catch, noexcept, std::expected, std::optional, error codes, assert, terminate."
---

# C++ Error Handling

## Core Question

**Is this error recoverable?**

- **Yes (Local)**: `std::expected` or return code.
- **Yes (Distant)**: Exceptions (`throw`).
- **No (Bug)**: `assert` or `std::terminate`.

## Error → Design Question

| Issue                  | Design Question                                      |
| ---------------------- | ---------------------------------------------------- |
| **Uncaught Exception** | Did you forget to catch, or throw in `noexcept`?     |
| **Silent Failure**     | Did you ignore a return code? (Use `[[nodiscard]]`). |
| **Destructor Throw**   | Never throw from destructor (`std::terminate`).      |

## Thinking Prompt

1.  **Is absence valid?**
    - Yes? → `std::optional<T>`.

2.  **Does caller need details?**
    - Yes? → `std::expected<T, E>` or Exception.
    - No? → `bool` or `std::optional`.

3.  **Is it a Logic Error (Bug)?**
    - Yes? → `assert()` or `std::terminate()`. Do not throw for bugs in C++ (Contract Violation).

## Trace Up / Down

-   **Trace Up**:
    -   *Issue*: "Program aborted with 'terminate called recursively'."
    -   *Cause*: An exception was thrown while stack unwinding (in a destructor).
    -   *Fix*: Fix the destructor. Ensure RAII cleanup is `noexcept`.

## Quick Reference

| Mechanism           | Cost (Happy)   | Cost (Sad) | Use When                     |
| ------------------- | -------------- | ---------- | ---------------------------- |
| **`std::optional`** | Branch         | Branch     | Return may be empty.         |
| **`std::expected`** | Branch         | Branch     | Recoverable error (Parsing). |
| **Exception**       | Zero           | Huge       | Rare IO/Resource errors.     |
| **Assert**          | Zero (Release) | Abort      | Logic bugs / Invariants.     |
