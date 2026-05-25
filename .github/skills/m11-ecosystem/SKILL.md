---
name: m11-ecosystem
description: "Mastering C++ Ecosystem: CMake, vcpkg, Conan, Sanitizers, Tooling."
---

# C++ Ecosystem

## Core Question

**How do I build and maintain this?**

- **Build**: CMake (Standard).
- **Deps**: vcpkg or Conan (Don't manual install).
- **Quality**: Clang-Tidy + AddressSanitizer.

## Thinking Prompt

1.  **Is the build reproducible?**
    - Yes? → CMake + Manifest (vcpkg.json).
    - No? → Manual paths (Bad).

2.  **Are you checking for bugs?**
    - `fsanitize=address` (ASan) catches 90% of memory errors.

## Quick Reference

| Tool           | Purpose                                  |
| -------------- | ---------------------------------------- |
| **CMake**      | Build System Generator.                  |
| **vcpkg**      | MSFT Package Manager (Source based).     |
| **Conan**      | Python Package Manager (Binary caching). |
| **ASan**       | Address Sanitizer (Memory bugs).         |
| **Clang-Tidy** | Static Analysis / Linter.                |
