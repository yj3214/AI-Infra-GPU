# Best Practices: Polymorphism & Templates

## 1. Use C++20 Concepts instead of SFINAE

**Old C++ (SFINAE)**: Unreadable `enable_if` macros.
**Modern C++**: Clean constraints.

```cpp
// Bad
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
void process(T t);

// Good
void process(std::integral auto t);
```

## 2. Prefer Composition over Inheritance

Inheritance is brittle. Standard library uses templates (Iterators) instead of inheritance.

- **Bad**: `class MyVector : public std::vector<int>` (No virtual destructor!).
- **Good**: `class MyVector { std::vector<int> internal; }`.

## 3. `virtual` Destructors

If you have *any* virtual functions, you MUST have a virtual destructor.

```cpp
struct Base {
    virtual void foo() = 0;
    virtual ~Base() = default; // Essential!
};
```
Without this, `delete base_ptr` will leak Derived resources.

## 4. `std::variant` (Sum Types)

For closed sets of types, `variant` is safer and faster than inheritance (data locality).

```cpp
using Shape = std::variant<Circle, Square>;
std::vector<Shape> shapes;

for (auto& s : shapes) {
    std::visit([](auto&& arg) { arg.draw(); }, s);
}
```
