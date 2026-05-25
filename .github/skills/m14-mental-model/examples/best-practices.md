# Best Practices: Mental Models

## 1. Init List Order

Members init in DECLARATION order, not list order.

```cpp
struct Foo {
    int a;
    int b;
    // Bad if flipped in list
    Foo() : a(10), b(a + 1) {} 
};
```

## 2. RAII is King

Stop trying to assume `goto cleanup` style logic. Use destructors.
