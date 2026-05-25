# Best Practices: Concurrency

## 1. Prefer `std::jthread` (C++20)

`std::thread` terminates if destroyed without join/detach. `std::jthread` automatically cancels and joins.

```cpp
void func() {
    std::jthread t([]{ ... }); 
} // Safe: t joins here.
```

## 2. Lock Guards (RAII)

Never call `.lock()` and `.unlock()` manually.

```cpp
std::mutex mtx;
{
    std::lock_guard lock(mtx); // C++11
    // or
    std::scoped_lock lock(mtx1, mtx2); // C++17 (Deadlock free multi-lock)
    // critical section
} // Auto-unlock
```

## 3. `volatile` is NOT Atomic

`volatile` in C++ does NOT imply thread safety. It implies "Hardware MMIO".
Always use `std::atomic<T>`.

## 4. Coroutines via Library

C++20 Coroutines (`co_await`) are "assembly language". You need a runtime library:
- **cppcoro** / **libunifex**
- **asio** (C++20 support)
- **folly**

Do not try to write your own promise_type/awaitables unless you are a library author.
