if(NOT WASM)
  link_libraries(leveldb)
  add_subdirectory(db_cli)
  add_subdirectory(keygen)
  add_subdirectory(rollup_cli)
  add_subdirectory(tx_factory)
endif()

add_subdirectory(proofs)