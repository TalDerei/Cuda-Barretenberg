add_executable(
    keygen
    main.cpp
)

target_link_libraries(
    keygen
    PRIVATE
    rollup_proofs_root_verifier
    env
)