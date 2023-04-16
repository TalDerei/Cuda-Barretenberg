# Vanilla Plonk 
add_executable(vanilla_plonk_bench vanilla_plonk.bench.cpp)

target_link_libraries(
  vanilla_plonk_bench
  stdlib_primitives
  env
  benchmark
)

add_custom_target(
    run_vanilla_plonk_bench
    COMMAND vanilla_plonk_bench
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Turboplonk
add_executable(turbo_plonk_bench turbo_plonk.bench.cpp)

target_link_libraries(
  turbo_plonk_bench
  stdlib_primitives
  env
  benchmark
)

add_custom_target(
    run_turbo_plonk_bench
    COMMAND turbo_plonk_bench
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Ultraplonk
add_executable(ultra_plonk_bench ultra_plonk.bench.cpp)

target_link_libraries(
  ultra_plonk_bench
  stdlib_primitives
  env
  benchmark
)

add_custom_target(
    run_ultra_plonk_bench
    COMMAND ultra_plonk_bench
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)


