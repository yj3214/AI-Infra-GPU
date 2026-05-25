# Best Practices: Lifecycle

## 1. Rule of Zero

If you can avoid writing a destructor, do it. Use `unique_ptr` members.

## 2. Rule of Five

If you write a Destructor, you usually need:
1. Copy Ctor
2. Copy Assign
3. Move Ctor
4. Move Assign
5. Destructor

(Or delete Copy if unique).

## 3. Meyers Singleton

Thread-safe, lazy-init global.

```cpp
Is &get() {
    static Is instance;
    return instance;
}
```
