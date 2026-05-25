# Best Practices: Error Handling

## 1. Exceptions are for Exceptional Circumstances

Don't use exceptions for control flow (e.g., "End of File"). Use them for things that shouldn't happen but might (Network Down, Disk Full).

## 2. Strong Exception Safety (RAII)

If an exception is thrown, your program state must be consistent (no leaks, no half-modified objects).

-   **Bad**: `new A; new B;` (If B fails, A leaks).
-   **Good**: `make_unique<A>(); make_unique<B>();`.

## 3. `noexcept`

Mark functions `noexcept` if they cannot throw (especially Move Constructors and Destructors). This allows `std::vector` to use optimization (using move instead of copy during resize).

```cpp
class Widget {
public:
    Widget(Widget&&) noexcept; // Critical for performance!
};
```

## 4. `std::expected` (C++23)

For parsing or business logic where failure is common.

```cpp
std::expected<int, std::string> parse(std::string_view s) {
    if (s.empty()) return std::unexpected("Empty");
    return std::stoi(std::string(s));
}
```
