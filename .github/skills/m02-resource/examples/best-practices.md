# Best Practices: Smart Pointers

## 1. Always use `make_unique` / `make_shared`

Do not use `new`.

```cpp
// Bad
std::unique_ptr<Widget> p(new Widget()); // Not exception safe in older C++ versions, verbose

// Good
auto p = std::make_unique<Widget>();
```

**Exception Safety**: In `foo(unique_ptr(new A), unique_ptr(new B))`, if `new B` throws, `new A` matches leak. `make_unique` fixes this.

## 2. Breaking Cycles with `weak_ptr`

Common pattern in Graphs or Parent/Child strctures.

```cpp
struct Node {
    std::vector<std::shared_ptr<Node>> children;
    std::weak_ptr<Node> parent; // NOT shared_ptr!
};
```

## 3. Custom Deleters for C interoperability

Use `unique_ptr` to modernize C libraries.

```cpp
struct FILE_deleter {
    void operator()(FILE* f) const { fclose(f); }
};

using FilePtr = std::unique_ptr<FILE, FILE_deleter>;

FilePtr open_file(const char* name) {
    return FilePtr(fopen(name, "r")); // Auto-closes on return/exception
}
```
