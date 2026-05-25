# Best Practices: Anti-Patterns

## 1. No C-Style Casts

They are nuclear options that try `static`, `const`, and `reinterpret` casts blindly.

## 2. No Output Parameters

Return values by value. RVO optimization makes it free.

```cpp
// Bad
void get(int& out);

// Good
int get();
```

## 3. No Singleton Managers

Global state makes testing impossible. Pass dependencies.
