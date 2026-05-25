---
name: m13-domain-error
description: "Mastering C++ Domain Errors: Exception Hierarchies, System Errors, and Expected."
---

# C++ Domain Errors

## Core Question

**Who catches this?**

- **Exception**: Inherit `std::runtime_error`.
- **System Error**: `std::error_code` (OS codes).

## Thinking Prompt

1.  **Is it a Domain Event?**
    - `InsufficientFunds` is an exception type.

2.  **Does it have context?**
    - "File not found" needs "Which file?".
    - `struct FileError : std::runtime_error { std::filesystem::path p; ... }`

## Quick Reference

| Pattern                  | Use Case                         |
| ------------------------ | -------------------------------- |
| **`std::runtime_error`** | Base for most failures.          |
| **`std::logic_error`**   | Bug (violation of precondition). |
| **`std::expected`**      | Visible failure path.            |
