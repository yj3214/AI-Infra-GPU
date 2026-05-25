---
name: m05-type-driven
description: "Mastering C++ Type-Driven Design. Triggers: strong types, phantom types, type state pattern, builder pattern, invalid state unrepresentable."
---

# C++ Type-Driven Design

## Core Question

**Can I make this bug a compile error?**

- **Primitive Obsession**: Using `int` for IDs, `double` for Money. Bad.
- **Strong Types**: `struct UserId`, `struct Money`. Good.
- **Type State**: `Connection<OFF>` vs `Connection<ON>`.

## Error → Design Question

| Issue                 | Design Question                                               |
| --------------------- | ------------------------------------------------------------- |
| **Swapped arguments** | Did you pass `width` to `height`? (Use Strong Types).         |
| **Invalid State**     | Did you call `read()` on closed file? (Use Type State).       |
| **Unit confusion**    | Did you mix Meters and Feet? (Use `std::chrono` style units). |

## Thinking Prompt

1.  **Is this `int` unique?**
    - Yes? → Wrap in `struct`.
    - `struct UserId { int val; };` prevents `process(OrderId)`.

2.  **Does valid usage depend on order?**
    - Yes? → Encode state in type.
    - `Builder::port()` returns `BuilderWithPort`.

3.  **Are units compatible?**
    - No? → Template tag. `Dist<Meters>` + `Dist<Feet>`.

## Trace Up / Down

-   **Trace Up**:
    -   *Issue*: "Rocket crashed because of Metric vs Imperial confusion."
    -   *Cause*: `double calculate_trajectory(double dist)` accepted any number.
    -   *Fix*: `Dist<Meters> calculate(Dist<Meters> d)`. Compilation fails if you pass Feet.

-   **Trace Down**:
    -   *Intent*: "Ensure file is open before reading."
    -   *Code*: `File<Open> f = File<Closed>().open(); f.read();`

## Quick Reference

| Pattern            | Cost | Use When                                |
| ------------------ | ---- | --------------------------------------- |
| **Struct Wrapper** | Zero | Distinct IDs, coordinates.              |
| **Enum Class**     | Zero | Type-safe flags (no implicit int conv). |
| **Phantom Type**   | Zero | Tracking state without storage.         |
| **User Literal**   | Zero | `10_m`, `50_s`.                         |
