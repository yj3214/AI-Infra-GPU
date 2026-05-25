# Best Practices: Ownership & RAII

## 1. The Rule of Zero

**Idiomatic C++**: Do not write destructors, copy/move constructors, or assignment operators. Use types that handle it for you (`std::vector`, `std::unique_ptr`, `std::string`).

```cpp
// Bad
class Handle {
    int* ptr;
public:
    Handle() : ptr(new int(0)) {}
    ~Handle() { delete ptr; } // Buggy: missing copy/move ops
};

// Good
class Handle {
    std::unique_ptr<int> ptr = std::make_unique<int>(0);
    // Compiler generates correct move ops. Copy is disabled automatically.
};
```

## 2. Sink Parameters

If a function needs to own the data, take it by value and `std::move` into place.

```cpp
class User {
    std::string name;
public:
    // Good: One overload, works for lvalues (copy) and rvalues (move)
    void setName(std::string new_name) {
        name = std::move(new_name);
    }
};
```

## 3. Avoid `std::shared_ptr` abuse

`shared_ptr` is the "Global Variable" of lifetimes. It makes ownership reasoning hard.
- **Default**: `unique_ptr`.
- **View**: `T*` or `T&`.
- **Only usage**: Complex cyclic graphs or async lifetimes where owner is unknown.
