# Best Practices: Domain Errors

## 1. Inherit Standard Exceptions

Don't throw `int` or `std::string`.

```cpp
struct MyError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
```

## 2. Catch by Reference

Always catch `const std::exception& e`. Slicing loses info.

## 3. Don't use Exceptions for Control Flow

Exceptions are slow (stack walking). Do not use them for valid business logic paths (e.g. "User Not Found" in a search API might be `expected` or `optional`).
