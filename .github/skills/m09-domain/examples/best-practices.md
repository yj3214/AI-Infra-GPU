# Best Practices: Domain Design

## 1. Pimpl Idiom (Compilation Firewall)

Detailed implementation logic in headers slows down builds and exposes deps. Use Pimpl.

```cpp
// User.h
class User {
public:
    User();
    ~User(); // Needed for unique_ptr forward decl
    void doWork();
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};
```

## 2. Strong Typing for IDs

Don't use `int`.

```cpp
struct UserId { 
    uint64_t val; 
    auto operator<=>(const UserId&) const = default; 
};
```

## 3. Class vs Struct

- Use `struct` for passive data (Value Objects, DTOs).
- Use `class` for active objects with invariants (Entities, Services).
