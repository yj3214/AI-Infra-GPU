# Best Practices: Const & Mutability

## 1. M&M Rule: Mutable needs Mutex

If you use the `mutable` keyword on a member variable, it is **your responsibility** to ensure thread safety.
A `const` function is callable from multiple threads simultaneously. If it modifies a `mutable` member without locking, you have a race.

```cpp
class Widget {
    mutable int cache_ = -1;
    mutable std::mutex mtx_; // Mutex must also be mutable!
public:
    int get() const {
        std::lock_guard lock(mtx_);
        if (cache_ == -1) cache_ = compute();
        return cache_;
    }
};
```

## 2. East Const vs West Const

It doesn't matter functionally, but be consistent.
- **West**: `const int* p` (Standard convention).
- **East**: `int const* p` ("Logical" convention: applies to what's on the left).

What matters is pointer qualifiers:
```cpp
const int* p;       // Pointer to CONST int (Data is immutable)
int* const p;       // CONST pointer to int (Pointer is immutable)
const int* const p; // CONST pointer to CONST int (Everything immutable)
```

## 3. Propagation of Const
C++ `const` does not propagate through pointers!

```cpp
class Owner {
    int* data;
public:
    // This is valid compilation, but logically questionable
    void clear() const {
        *data = 0; // The POINTER is const, but the data it points to is NOT.
    }
};
```
**Fix**: Use `std::experimental::propagate_const` or be careful.
