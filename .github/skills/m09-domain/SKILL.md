---
name: m09-domain
description: "Mastering C++ Domain Modeling (DDD). Triggers: Entity, Value Object, Aggregate, Repository, Pimpl, Class Design, Invariant."
---

# C++ Domain Modeling

## Core Question

**Identity or Value?**

- **Value Object**: Defined by attributes (`Color`, `Money`). Equality via comparison. Copyable.
- **Entity**: Defined by Identity (`User`, `Socket`). Equality via ID. Non-copyable (usually).

## Error → Design Question

| Issue                  | Design Question                                      |
| ---------------------- | ---------------------------------------------------- |
| **Data Inconsistency** | Are public fields allowing invalid states?           |
| **Object Slicing**     | Are you passing polymorphic Entities by Value?       |
| **Header Hell**        | Are you leaking implementation details? (Use Pimpl). |

## Thinking Prompt

1.  **Is it copyable?**
    - Yes? → Value Type (Rule of Zero, defaults).
    - No? → Entity (Delete copy ctor, enable move).

2.  **Does it have invariants?**
    - Yes? → `class` with private data + public methods.
    - No? → `struct` (POD).

3.  **Does it own others?**
    - Aggregate Root? → Owns children via `std::vector` / `unique_ptr`.

## Trace Up / Down

-   **Trace Down**:
    -   *Intent*: "User has a Name and an Address."
    -   *Code*: `class User` (Entity) holds `Name` (Value) and `Address` (Value).

## Quick Reference

| Pattern          | C++ Implementation                          |
| ---------------- | ------------------------------------------- |
| **Value Object** | `struct` + `operator<=>`.                   |
| **Entity**       | `class` + Deleted Copy + ID field.          |
| **Repository**   | Pure Virtual Interface (`virtual ... = 0`). |
| **Aggregate**    | Parent class owning children.               |
