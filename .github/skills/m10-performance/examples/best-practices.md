# Best Practices: Performance

## 1. Stop using `std::list` / `std::map`

Linked nodes are cache destroyers.
- Replace `std::list` with `std::vector`.
- Replace `std::map` with `std::vector` (sorted) or `boost::flat_map`.
- Use `std::unordered_map` only if needed.

## 2. SSO (Small String Optimization)

`std::string` doesn't allocate for small strings (usually < 15-22 chars).
Keep strings short to stay on stack.

## 3. Polymorphic Allocators (`pmr`)

C++17 feature to swap allocators at runtime.

```cpp
char buf[1024];
std::pmr::monotonic_buffer_resource pool(buf, 1024);
std::pmr::vector<int> v{&pool}; // No heap alloc!
```
