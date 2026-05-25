---
name: m07-concurrency
description: "Mastering C++ Concurrency. Triggers: std::thread, jthread, atomic, mutex, deadlock, race condition, memory model, coroutine, async."
---

# C++ Concurrency

## Core Question

**How do threads communicate?**

- **Shared State**: Mutexes (`std::mutex`) or Atomics (`std::atomic`).
- **Coordination**: Condition Variables (`std::condition_variable`) or Latches/Semaphores (C++20).
- **Tasks**: `std::async` or Coroutines (C++20).

## Error → Design Question

| Issue             | Design Question                                                    |
| ----------------- | ------------------------------------------------------------------ |
| **Data Race**     | Are two threads accessing memory without a Happens-Before edge?    |
| **Deadlock**      | Did you lock Mutex A then B, while another thread locked B then A? |
| **False Sharing** | Are independent atomics sitting on the same cache line?            |
| **Live Lock**     | Are threads spinning without progress?                             |

## Thinking Prompt

1.  **Do I need a thread?**
    - Short task? → `std::async` or Thread Pool.
    - Long background? → `std::jthread`.

2.  **Is it shared state?**
    - Yes? → Protect with `std::mutex`.
    - Is it small/primitive? → `std::atomic`.

3.  **Locks or Atomics?**
    - Logic requiring multiple steps? → Mutex (Atomic is only for single instructions).
    - Simple flag/counter? → Atomic.

## Trace Up / Down

-   **Trace Up**:
    -   *Issue*: "Random update values in counter."
    -   *Cause*: `counter++` is Read-Modify-Write, not atomic. Data race.
    -   *Fix*: `std::atomic<int>`.

## Quick Reference

| Tool                    | C++ Version | Use When                      |
| ----------------------- | ----------- | ----------------------------- |
| **`std::jthread`**      | C++20       | Standard thread (auto-join).  |
| **`std::atomic`**       | C++11       | Lock-free counters/flags.     |
| **`std::mutex`**        | C++11       | Locking critical sections.    |
| **`std::shared_mutex`** | C++17       | Read-heavy workloads.         |
| **`std::latch`**        | C++20       | Waiting for N tasks to start. |
| **`co_await`**          | C++20       | Async I/O (requires library). |
