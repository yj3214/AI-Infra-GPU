# Best Practices: Ecosystem

## 1. Modern CMake

Don't use `include_directories`. Use Targets.

```cmake
# Bad
include_directories(${Boost_INCLUDE_DIRS})

# Good
target_link_libraries(MyApp PRIVATE Boost::filesystem)
```

## 2. Always use Sanitizers

In Debug builds, enable ASan and UBSan.

```cmake
add_compile_options(-fsanitize=address,undefined)
add_link_options(-fsanitize=address,undefined)
```

## 3. Package Management

Stop using `sudo apt-get install libboost-dev`.
Use `vcpkg.json` manifest mode. It ensures all devs use the exact same library versions.
