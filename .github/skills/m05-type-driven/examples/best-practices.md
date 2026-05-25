# Best Practices: Type Safety

## 1. Strong Types (The "Struct" trick)

Don't use `typedef` or `using` for ID safety. They are weak aliases.

```cpp
// Bad
using Width = int;
using Height = int;
void resize(Width w, Height h); // Compiler sees (int, int)

// Good
struct Width { int v; };
struct Height { int v; };
void resize(Width w, Height h); // Compiler enforcement
```

## 2. Scoped Enums (`enum class`)

Always prefer `enum class` over `enum`.

```cpp
// Bad
enum Color { RED, BLUE }; // Pollutes namespace, implicit int

// Good
enum class Color : uint8_t { Red, Blue }; // Type safe, scoped
```

## 3. User-Defined Literals

Make units readable.

```cpp
auto operator""_kg(long double x) { return Mass{x}; }
auto operator""_lb(long double x) { return Mass{x * 0.453}; }

Mass m = 10.0_kg;
```

## 4. `[[nodiscard]]`

Force users to check return values (like Errors).

```cpp
[[nodiscard]] ErrorCode connect();
```
